"""取消生命周期：后端在 yield 之前就阻塞时的行为。

这是协作式取消最难的一种情形——DeerFlow 的模型调用或工具执行可能长时间卡在
``next(generator)`` 内部，此时工作线程根本到不了下一个 yield 边界，既观察不到
取消标志，也发不出完成信号。桥必须在这种情况下**仍然**在宽限期内让
``session/prompt`` 返回，并且不能让这个卡住的旧 worker 影响同一 session 的后续
turn；否则客户端会永久挂起，或者后续每一次 prompt 都被判为「turn 正在执行」。
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from deerflow_acp.config import BridgeConfig
from deerflow_acp.session import SessionRegistry


class StallingBackend:
    """首次 ``stream()`` 在产出任何事件之前无限阻塞的后端。

    ``release`` 置位后阻塞解除；测试必须在收尾时置位，否则会留下泄漏线程。
    每次 stream 调用产出的事件都带上调用序号，用于证明旧 turn 的事件不会串到
    新 turn 上。
    """

    def __init__(self, *, events_per_turn: int = 3) -> None:
        self.release = threading.Event()
        self.entered = threading.Event()
        self.events_per_turn = events_per_turn
        self.calls = 0
        self.closed = threading.Event()

    def stream(self, message: str, *, thread_id: str) -> Iterator[tuple[str, dict[str, Any]]]:
        self.calls += 1
        call_index = self.calls

        def gen() -> Iterator[tuple[str, dict[str, Any]]]:
            try:
                if call_index == 1:
                    # 关键：阻塞发生在第一个 yield **之前**
                    self.entered.set()
                    self.release.wait(timeout=30)
                for i in range(self.events_per_turn):
                    yield ("messages-tuple", {"type": "ai", "content": f"c{call_index}-{i}", "id": "m"})
            except GeneratorExit:
                self.closed.set()
                raise

        return gen()

    def thread_exists(self, thread_id: str) -> bool:
        return False

    def history(self, thread_id: str) -> list[dict[str, Any]]:
        return []


def registry(backend: StallingBackend, **cfg: Any) -> SessionRegistry:
    return SessionRegistry(backend, BridgeConfig(**cfg))


async def _noop(event_type: str, data: dict) -> None:
    return None


async def test_cancel_returns_within_grace_when_backend_blocks_before_first_yield():
    backend = StallingBackend()
    reg = registry(backend, cancel_grace_seconds=0.3)
    session = reg.create("/tmp")
    try:
        task = asyncio.create_task(reg.run_turn(session, "问题", _noop))
        assert await asyncio.to_thread(backend.entered.wait, 2)
        reg.cancel(session.session_id)

        # 宽限期 0.3s；给足调度余量，但远小于后端 30s 的阻塞
        outcome = await asyncio.wait_for(task, timeout=5)

        assert outcome.stop_reason == "cancelled"
        assert outcome.escalated is True, "后端未在宽限期内协作退出，必须如实升级"
        assert session.running is False, "必须释放 running，否则后续 prompt 永远被拒"
    finally:
        backend.release.set()


async def test_session_accepts_new_turn_after_escalated_cancel():
    backend = StallingBackend()
    reg = registry(backend, cancel_grace_seconds=0.3)
    session = reg.create("/tmp")
    try:
        task = asyncio.create_task(reg.run_turn(session, "第一问", _noop))
        assert await asyncio.to_thread(backend.entered.wait, 2)
        reg.cancel(session.session_id)
        first = await asyncio.wait_for(task, timeout=5)
        assert first.stop_reason == "cancelled"

        # 放行旧 worker，让它有机会去干扰后续 turn——正确实现下它干扰不了
        backend.release.set()

        seen: list[tuple[str, dict]] = []

        async def on_event(event_type: str, data: dict) -> None:
            seen.append((event_type, data))

        second = await asyncio.wait_for(
            reg.run_turn(session, "第二问", on_event), timeout=5
        )

        assert second.stop_reason == "end_turn"
        assert second.escalated is False
        contents = [d["content"] for _, d in seen]
        assert contents == ["c2-0", "c2-1", "c2-2"], f"旧 turn 的事件串进了新 turn：{contents}"
    finally:
        backend.release.set()


async def test_escalated_cancel_does_not_leak_stale_cancel_flag():
    """升级取消后，新 turn 不得继承上一个 turn 的取消标志。"""
    backend = StallingBackend()
    reg = registry(backend, cancel_grace_seconds=0.2)
    session = reg.create("/tmp")
    try:
        task = asyncio.create_task(reg.run_turn(session, "第一问", _noop))
        assert await asyncio.to_thread(backend.entered.wait, 2)
        reg.cancel(session.session_id)
        await asyncio.wait_for(task, timeout=5)
        backend.release.set()

        second = await asyncio.wait_for(reg.run_turn(session, "第二问", _noop), timeout=5)
        assert second.stop_reason == "end_turn", "新 turn 被上一轮的取消标志误伤"
    finally:
        backend.release.set()


async def test_normal_turn_is_not_slowed_by_cancel_polling():
    """轮询实现不得拖慢正常 turn——没有取消时应当立即随事件推进。"""
    backend = StallingBackend(events_per_turn=50)
    backend.release.set()
    reg = registry(backend, cancel_grace_seconds=5.0)
    session = reg.create("/tmp")
    seen: list[tuple[str, dict]] = []

    async def on_event(event_type: str, data: dict) -> None:
        seen.append((event_type, data))

    loop = asyncio.get_running_loop()
    started = loop.time()
    outcome = await asyncio.wait_for(reg.run_turn(session, "问题", on_event), timeout=5)
    elapsed = loop.time() - started

    assert outcome.stop_reason == "end_turn"
    assert len(seen) == 50
    assert elapsed < 1.0, f"50 个事件耗时 {elapsed:.2f}s，轮询间隔拖慢了正常路径"


@pytest.mark.parametrize("grace", [0.1, 0.4])
async def test_cancel_respects_configured_grace(grace: float):
    backend = StallingBackend()
    reg = registry(backend, cancel_grace_seconds=grace)
    session = reg.create("/tmp")
    try:
        task = asyncio.create_task(reg.run_turn(session, "问题", _noop))
        assert await asyncio.to_thread(backend.entered.wait, 2)

        loop = asyncio.get_running_loop()
        reg.cancel(session.session_id)
        started = loop.time()
        outcome = await asyncio.wait_for(task, timeout=5)
        elapsed = loop.time() - started

        assert outcome.escalated is True
        assert elapsed >= grace * 0.5, f"{elapsed:.2f}s 明显短于宽限期 {grace}s，没有真的等"
        assert elapsed < grace + 2.0, f"{elapsed:.2f}s 远超宽限期 {grace}s"
    finally:
        backend.release.set()
