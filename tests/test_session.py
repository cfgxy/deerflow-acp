"""会话注册与 turn 生命周期测试。"""

from __future__ import annotations

import asyncio
import threading

import pytest

from conftest import FakeBackend

from deerflow_acp.backend import BackendUnavailableError
from deerflow_acp.config import BridgeConfig
from deerflow_acp.session import (
    SessionIdError,
    SessionRegistry,
    TurnAlreadyRunningError,
    UnknownSessionError,
    new_session_id,
    validate_session_id,
)


def registry(backend: FakeBackend, **cfg) -> SessionRegistry:
    return SessionRegistry(backend, BridgeConfig(**cfg))


async def collect(reg: SessionRegistry, session, message: str):
    seen: list[tuple[str, dict]] = []

    async def on_event(event_type, data):
        seen.append((event_type, data))

    outcome = await reg.run_turn(session, message, on_event)
    return outcome, seen


# ----------------------------------------------------------------------
# sessionId 约束
# ----------------------------------------------------------------------


def test_generated_session_id_satisfies_deerflow_thread_id_constraint():
    for _ in range(20):
        assert validate_session_id(new_session_id())


@pytest.mark.parametrize("bad", ["", "a" * 65, "has space", "斜杠/不行", "点.不行", None, 123])
def test_invalid_session_ids_are_rejected(bad):
    with pytest.raises(SessionIdError):
        validate_session_id(bad)


# ----------------------------------------------------------------------
# 恢复语义
# ----------------------------------------------------------------------


def test_resume_unknown_thread_raises_instead_of_creating():
    reg = registry(FakeBackend(threads={}))
    with pytest.raises(UnknownSessionError):
        reg.resume("df-nonexistent", "/tmp")
    # 关键：拒绝后不得留下任何会话，否则后续 prompt 会静默从空白开始
    with pytest.raises(UnknownSessionError):
        reg.get("df-nonexistent")


def test_resume_existing_thread_creates_session():
    reg = registry(FakeBackend(threads={"df-known": [{"type": "human", "content": "hi"}]}))
    session = reg.resume("df-known", "/tmp")
    assert session.session_id == "df-known"
    assert reg.get("df-known") is session


def test_resume_in_process_session_is_idempotent():
    backend = FakeBackend(threads={"df-known": []})
    reg = registry(backend)
    created = reg.create("/tmp")
    assert reg.resume(created.session_id, "/tmp") is created


def test_resume_propagates_backend_failure_not_unknown_session():
    """checkpointer 故障必须与「会话不存在」分账。"""
    reg = registry(FakeBackend(thread_lookup_error=BackendUnavailableError("db down")))
    with pytest.raises(BackendUnavailableError):
        reg.resume("df-known", "/tmp")


def test_close_removes_session():
    reg = registry(FakeBackend())
    session = reg.create("/tmp")
    reg.close(session.session_id)
    with pytest.raises(UnknownSessionError):
        reg.get(session.session_id)


def test_cancel_unknown_session_is_silent():
    reg = registry(FakeBackend())
    reg.cancel("df-nope")  # 不抛异常：cancel 是 JSON-RPC 通知，没有响应通道


# ----------------------------------------------------------------------
# turn 执行
# ----------------------------------------------------------------------


async def test_turn_streams_events_and_reports_usage():
    backend = FakeBackend(
        [
            ("messages-tuple", {"type": "ai", "content": "hi", "id": "m1"}),
            ("end", {"usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}}),
        ]
    )
    reg = registry(backend)
    session = reg.create("/tmp")
    outcome, seen = await collect(reg, session, "问题")

    assert outcome.stop_reason == "end_turn"
    assert outcome.usage_payload == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
    assert [t for t, _ in seen] == ["messages-tuple", "end"]
    # sessionId 原样用作 thread_id
    assert backend.stream_calls == [("问题", session.session_id)]


async def test_backend_error_surfaces_in_outcome():
    reg = registry(FakeBackend(raise_on_stream=RuntimeError("模型挂了")))
    session = reg.create("/tmp")
    outcome, _ = await collect(reg, session, "问题")
    assert outcome.stop_reason == "refusal"
    assert isinstance(outcome.error, RuntimeError)


async def test_concurrent_turn_on_same_session_is_rejected():
    backend = FakeBackend([("messages-tuple", {"type": "ai", "content": "x", "id": "m"})])
    backend.gate = threading.Event()
    reg = registry(backend)
    session = reg.create("/tmp")

    first = asyncio.create_task(collect(reg, session, "a"))
    await asyncio.sleep(0.05)
    with pytest.raises(TurnAlreadyRunningError):
        await collect(reg, session, "b")

    backend.gate.set()
    await first


async def test_cancel_closes_generator_and_stops_early():
    events = [("messages-tuple", {"type": "ai", "content": str(i), "id": "m"}) for i in range(200)]
    backend = FakeBackend(events)
    reg = registry(backend, cancel_grace_seconds=2.0)
    session = reg.create("/tmp")

    delivered: list[tuple[str, dict]] = []

    async def on_event(event_type, data):
        delivered.append((event_type, data))
        if len(delivered) == 3:
            reg.cancel(session.session_id)

    outcome = await reg.run_turn(session, "问题", on_event)

    assert outcome.stop_reason == "cancelled"
    assert outcome.escalated is False
    # 协作式取消真的把生成器关掉了，而不是靠丢弃事件伪装
    assert backend.closed.wait(timeout=2)
    assert backend.emitted < len(events)


async def test_cancel_before_any_event_still_yields_cancelled():
    backend = FakeBackend([("messages-tuple", {"type": "ai", "content": "x", "id": "m"})] * 50)
    backend.gate = threading.Event()
    reg = registry(backend, cancel_grace_seconds=2.0)
    session = reg.create("/tmp")

    async def on_event(event_type, data):
        return None

    task = asyncio.create_task(reg.run_turn(session, "问题", on_event))
    await asyncio.sleep(0.05)
    reg.cancel(session.session_id)
    backend.gate.set()
    outcome = await task
    assert outcome.stop_reason == "cancelled"


async def test_session_not_marked_running_after_turn():
    backend = FakeBackend([("end", {"usage": {}})])
    reg = registry(backend)
    session = reg.create("/tmp")
    await collect(reg, session, "问题")
    assert session.running is False
    assert reg.active_sessions == []


async def test_active_sessions_reports_in_flight_turn():
    backend = FakeBackend([("messages-tuple", {"type": "ai", "content": "x", "id": "m"})])
    backend.gate = threading.Event()
    reg = registry(backend)
    session = reg.create("/tmp")

    task = asyncio.create_task(collect(reg, session, "a"))
    await asyncio.sleep(0.05)
    assert reg.active_sessions == [session]
    backend.gate.set()
    await task
