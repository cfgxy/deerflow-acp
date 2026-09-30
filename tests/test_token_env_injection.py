"""验证 ``MULTICA_TOKEN`` 经 request-scoped secrets 通道逐 turn 注入 DeerFlow。

RUYI-300：DeerFlow 沙箱默认剥离名字含 TOKEN 的环境变量，multica CLI 在 bash
工具下拿不到任务凭据。修复形态：桥的 ``EmbeddedDeerFlowBackend.stream()`` 每
次 turn 从桥进程环境收集 ``MULTICA_TOKEN``（worker 子进程继承同一环境），经
DeerFlow 的 ``direct_env_secrets`` 信任通道随请求注入——DeerFlow 侧由
``test_direct_env_secrets.py`` 覆盖 bash 工具的送达与剥离不弱化。

本文件只覆盖桥侧三段：环境收集、逐 turn 请求构造、无凭据时零注入。
断言全部落在对 ``DeerFlowClient.stream`` 请求构造上；测试值一律为假 token。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from deerflow_acp.backend import EmbeddedDeerFlowBackend
from deerflow_acp.config import BridgeConfig

#: 假 token——任何真实凭据值都不得出现在测试里。
FAKE_TOKEN = "fake-multica-token-for-tests"


class _RecordingClient:
    """记录 stream() 收到的 kwargs，产出一条可消费事件。"""

    def __init__(self) -> None:
        self.stream_calls: list[dict[str, Any]] = []

    def stream(self, message: str, *, thread_id: str, **kwargs: Any) -> Iterator[Any]:
        self.stream_calls.append({"thread_id": thread_id, **kwargs})
        return iter([_FakeEvent("end", {"usage": {"input_tokens": 0}})])


class _FakeEvent:
    def __init__(self, type: str, data: dict[str, Any]) -> None:
        self.type = type
        self.data = data


def _backend_with_client(monkeypatch: pytest.MonkeyPatch) -> tuple[EmbeddedDeerFlowBackend, _RecordingClient]:
    backend = EmbeddedDeerFlowBackend(BridgeConfig())
    client = _RecordingClient()
    monkeypatch.setattr(backend, "_ensure_client", lambda: client)
    return backend, client


def test_stream_injects_multica_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """桥进程环境带 MULTICA_TOKEN 时，每次 turn 请求携带 direct_env_secrets。"""
    monkeypatch.setenv("MULTICA_TOKEN", FAKE_TOKEN)
    backend, client = _backend_with_client(monkeypatch)

    list(backend.stream("你好", thread_id="sess-1"))

    assert len(client.stream_calls) == 1
    call = client.stream_calls[0]
    assert call["thread_id"] == "sess-1"
    assert call["direct_env_secrets"] == {"MULTICA_TOKEN": FAKE_TOKEN}


def test_stream_omits_injection_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """环境无 MULTICA_TOKEN（如本地裸跑）时不注入——不传空映射，保持 DeerFlow
    请求形态与未打补丁前一致。"""
    monkeypatch.delenv("MULTICA_TOKEN", raising=False)
    backend, client = _backend_with_client(monkeypatch)

    list(backend.stream("你好", thread_id="sess-1"))

    assert "direct_env_secrets" not in client.stream_calls[0]


def test_stream_ignores_blank_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """空串视同缺失：不注入空值凭据。"""
    monkeypatch.setenv("MULTICA_TOKEN", "")
    backend, client = _backend_with_client(monkeypatch)

    list(backend.stream("你好", thread_id="sess-1"))

    assert "direct_env_secrets" not in client.stream_calls[0]


def test_model_override_and_token_coexist(monkeypatch: pytest.MonkeyPatch) -> None:
    """与 RUYI-283 的逐轮模型覆盖同请求共存：两个通道互不排挤。"""
    monkeypatch.setenv("MULTICA_TOKEN", FAKE_TOKEN)
    backend, client = _backend_with_client(monkeypatch)

    list(backend.stream("你好", thread_id="sess-1", model_name="glm-5.3-flash"))

    call = client.stream_calls[0]
    assert call["model_name"] == "glm-5.3-flash"
    assert call["direct_env_secrets"] == {"MULTICA_TOKEN": FAKE_TOKEN}


def test_each_turn_recollects_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """逐 turn 收集：前一轮有、后一轮无时，第二轮请求不得再携带——token 的
    生命周期跟随请求，不做进程级缓存。"""
    backend, client = _backend_with_client(monkeypatch)

    monkeypatch.setenv("MULTICA_TOKEN", FAKE_TOKEN)
    list(backend.stream("第一轮", thread_id="sess-1"))
    monkeypatch.delenv("MULTICA_TOKEN")
    list(backend.stream("第二轮", thread_id="sess-1"))

    assert client.stream_calls[0]["direct_env_secrets"] == {"MULTICA_TOKEN": FAKE_TOKEN}
    assert "direct_env_secrets" not in client.stream_calls[1]
