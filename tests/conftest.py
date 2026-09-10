"""测试公共装置：一个可编排的假 DeerFlow 后端。

假后端只替换 `DeerFlowBackend` 这一个协议面（stream / thread_exists / history），
桥的协议层、归一化层、会话层、线程与取消逻辑全部是真实代码路径——
测试验证的是桥的行为，不是 mock 的自洽。
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

import pytest

from deerflow_acp.backend import BackendUnavailableError


class FakeBackend:
    """可编排的 DeerFlow 后端替身。"""

    def __init__(
        self,
        events: list[tuple[str, dict[str, Any]]] | None = None,
        *,
        threads: dict[str, list[dict[str, Any]]] | None = None,
        raise_on_stream: BaseException | None = None,
        thread_lookup_error: BaseException | None = None,
    ) -> None:
        self.events = events or []
        self.threads = threads or {}
        self.raise_on_stream = raise_on_stream
        self.thread_lookup_error = thread_lookup_error
        #: 每次 yield 前等待此事件（用于制造「turn 正在运行」的窗口）
        self.gate: threading.Event | None = None
        #: 生成器被 close() 时置位，用于断言协作式取消真的生效
        self.closed = threading.Event()
        #: 实际产出的事件数，用于断言取消提前中断了流
        self.emitted = 0
        self.stream_calls: list[tuple[str, str]] = []

    def stream(self, message: str, *, thread_id: str) -> Iterator[tuple[str, dict[str, Any]]]:
        self.stream_calls.append((message, thread_id))
        if self.raise_on_stream is not None:
            raise self.raise_on_stream

        def gen() -> Iterator[tuple[str, dict[str, Any]]]:
            try:
                for event in self.events:
                    if self.gate is not None:
                        self.gate.wait(timeout=10)
                    self.emitted += 1
                    yield event
            except GeneratorExit:
                self.closed.set()
                raise

        return gen()

    def thread_exists(self, thread_id: str) -> bool:
        if self.thread_lookup_error is not None:
            raise self.thread_lookup_error
        return thread_id in self.threads

    def history(self, thread_id: str) -> list[dict[str, Any]]:
        if self.thread_lookup_error is not None:
            raise self.thread_lookup_error
        return list(self.threads.get(thread_id, []))


class RecordingConnection:
    """记录 session/update 通知的假连接。"""

    def __init__(self) -> None:
        self.updates: list[tuple[str, Any]] = []

    async def session_update(self, *, session_id: str, update: Any) -> None:
        self.updates.append((session_id, update))

    def update_kinds(self) -> list[str]:
        return [getattr(u, "session_update", None) for _, u in self.updates]

    def texts(self, kind: str) -> list[str]:
        out = []
        for _, update in self.updates:
            if getattr(update, "session_update", None) == kind:
                out.append(update.content.text)
        return out


@pytest.fixture
def backend_unavailable() -> BackendUnavailableError:
    return BackendUnavailableError("测试用后端故障")


@pytest.fixture
def connection() -> RecordingConnection:
    return RecordingConnection()
