"""思考开关穿透链测试：会话覆盖 → runner payload / worker 字段解析 / backend stream kwargs。

agent 层的广播与 set_config_option 行为见 ``test_agent.py``；本文件只覆盖
覆盖值从会话层到达 ``DeerFlowClient.stream`` 的三段管线。
"""

from __future__ import annotations

import io
import json

from deerflow_acp import backend as backend_module
from deerflow_acp.backend import EmbeddedDeerFlowBackend
from deerflow_acp.config import BridgeConfig
from deerflow_acp.runner import SubprocessTurnRunner
from deerflow_acp import worker as worker_module


# ----------------------------------------------------------------------
# SubprocessTurnRunner：job payload 顶层携带每轮覆盖
# ----------------------------------------------------------------------


def test_job_payload_carries_thinking_override():
    runner = SubprocessTurnRunner(BridgeConfig(thinking_enabled=True))
    payload = json.loads(runner._job_payload("df-1", "hi", model_name=None, thinking_enabled=False))

    assert payload["thinking_enabled"] is False
    # 静态默认仍在 config 段：worker 侧顶层覆盖优先，缺失时回退静态值
    assert payload["config"]["thinking_enabled"] is True


def test_job_payload_without_override_is_none():
    runner = SubprocessTurnRunner(BridgeConfig(thinking_enabled=True))
    payload = json.loads(runner._job_payload("df-1", "hi", model_name=None, thinking_enabled=None))

    assert payload["thinking_enabled"] is None


# ----------------------------------------------------------------------
# worker：字段解析 + 传递给 backend.stream（类型污染显式归 None）
# ----------------------------------------------------------------------


class _RecordingBackend:
    def __init__(self) -> None:
        self.kwargs: list[dict] = []

    def stream(self, message, *, thread_id, model_name=None, thinking_enabled=None):
        self.kwargs.append(
            {"model_name": model_name, "thinking_enabled": thinking_enabled},
        )
        return iter([])


def _run_worker_job(job: dict) -> _RecordingBackend:
    backend = _RecordingBackend()
    original = worker_module._load_backend
    worker_module._load_backend = lambda config: backend
    try:
        assert worker_module.run(job, io.BytesIO()) == 0
    finally:
        worker_module._load_backend = original
    return backend


def test_worker_threads_thinking_override_into_backend():
    backend = _run_worker_job(
        {"config": {}, "message": "hi", "thread_id": "df-1", "thinking_enabled": False},
    )
    assert backend.kwargs == [{"model_name": None, "thinking_enabled": False}]


def test_worker_without_thinking_override_passes_none():
    backend = _run_worker_job({"config": {}, "message": "hi", "thread_id": "df-1"})
    assert backend.kwargs == [{"model_name": None, "thinking_enabled": None}]


def test_worker_rejects_non_bool_thinking_value():
    """JSON 里 1/0 不是合法开关：类型污染显式归 None（回退静态默认），不猜。"""
    backend = _run_worker_job(
        {"config": {}, "message": "hi", "thread_id": "df-1", "thinking_enabled": 1},
    )
    assert backend.kwargs == [{"model_name": None, "thinking_enabled": None}]


# ----------------------------------------------------------------------
# EmbeddedDeerFlowBackend：覆盖值进入 client.stream kwargs，None 不传
# ----------------------------------------------------------------------


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def stream(self, message, *, thread_id=None, **kwargs):
        self.calls.append({"thread_id": thread_id, **kwargs})
        return iter([])


def _backend_with_client(monkeypatch=None) -> tuple[EmbeddedDeerFlowBackend, _FakeClient]:
    backend = EmbeddedDeerFlowBackend(BridgeConfig(thinking_enabled=True))
    client = _FakeClient()
    backend._client = client
    if monkeypatch is not None:
        # direct_env_secrets 注入取决于进程环境（存在即附带），与本组测试
        # 验证的 thinking kwargs 线程化正交；固定为空保证断言不受环境影响。
        monkeypatch.setattr(backend_module, "_collect_direct_env_secrets", dict)
    return backend, client


def test_backend_streams_thinking_override(monkeypatch):
    backend, client = _backend_with_client(monkeypatch)
    list(backend.stream("hi", thread_id="df-1", thinking_enabled=False))
    assert client.calls == [{"thread_id": "df-1", "thinking_enabled": False}]


def test_backend_streams_without_override_sends_no_thinking_kwarg(monkeypatch):
    """None 时不传 kwargs：DeerFlow 沿用 client 构造时的静态默认。"""
    backend, client = _backend_with_client(monkeypatch)
    list(backend.stream("hi", thread_id="df-1", thinking_enabled=None))
    assert client.calls == [{"thread_id": "df-1"}]
