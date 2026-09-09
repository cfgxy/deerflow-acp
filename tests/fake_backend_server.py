"""契约测试用的 ACP server 进程入口。

用真实的 `deerflow-acp` CLI 服务栈（真实 stdio、真实 JSON-RPC 编解码、
真实 SDK router），只把 DeerFlow 后端换成脚本化的假后端——脚本从
`DEERFLOW_ACP_FAKE_SCRIPT` 指向的 JSON 文件读取。

这样契约测试跑的是桥自己的协议行为，不需要真实模型调用。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections.abc import Iterator
from typing import Any


class ScriptedBackend:
    """按脚本产出事件的假 DeerFlow 后端。

    脚本格式::

        {
          "events": [["messages-tuple", {...}], ["end", {...}]],
          "threads": {"df-known": [{"type": "ai", "content": "..."}]},
          "delay_ms": 0,           # 每个事件之间的间隔，用于制造取消窗口
          "repeat": 1,             # events 重复次数
          "raise": "RuntimeError"  # 若设置，stream 立即抛该异常
        }
    """

    def __init__(self, script: dict[str, Any]) -> None:
        self._events = [tuple(item) for item in script.get("events", [])]
        self._threads = script.get("threads", {})
        self._delay = float(script.get("delay_ms", 0)) / 1000.0
        self._repeat = int(script.get("repeat", 1))
        self._raise = script.get("raise")
        # 通过一个文件标记生成器是否被 close()，让父进程可以断言协作式取消
        self._closed_marker = os.environ.get("DEERFLOW_ACP_FAKE_CLOSED_MARKER")

    def stream(self, message: str, *, thread_id: str) -> Iterator[tuple[str, dict[str, Any]]]:
        if self._raise:
            raise RuntimeError(self._raise)

        def gen() -> Iterator[tuple[str, dict[str, Any]]]:
            try:
                for _ in range(self._repeat):
                    for event_type, data in self._events:
                        if self._delay:
                            time.sleep(self._delay)
                        yield event_type, dict(data)
            except GeneratorExit:
                if self._closed_marker:
                    with open(self._closed_marker, "w", encoding="utf-8") as fh:
                        fh.write("closed")
                raise

        return gen()

    def thread_exists(self, thread_id: str) -> bool:
        return thread_id in self._threads

    def history(self, thread_id: str) -> list[dict[str, Any]]:
        return list(self._threads.get(thread_id, []))


def main() -> int:
    from deerflow_acp.agent import DeerFlowAgent
    from deerflow_acp.cli import isolate_stdout
    from deerflow_acp.config import BridgeConfig
    from deerflow_acp.logging_setup import configure_logging, redirect_root_logging_to_stderr

    configure_logging("WARNING")
    redirect_root_logging_to_stderr("WARNING")

    script_path = os.environ["DEERFLOW_ACP_FAKE_SCRIPT"]
    with open(script_path, encoding="utf-8") as fh:
        script = json.load(fh)

    backend = ScriptedBackend(script)
    config = BridgeConfig.from_env()

    protocol_stdout = isolate_stdout()

    # 故意在协议启动后往 fd 1 写垃圾：验证 stdout 隔离确实有效。
    if os.environ.get("DEERFLOW_ACP_FAKE_POLLUTE"):
        print("这行垃圾绝不能出现在 JSON-RPC 通道里")
        sys.stdout.write("再来一行\n")
        sys.stdout.flush()
        os.write(1, "裸 write 也不行\n".encode())

    async def run() -> int:
        import acp

        from deerflow_acp.cli import _stdio_streams

        reader, writer = await _stdio_streams(protocol_stdout)
        await acp.run_agent(
            lambda conn: DeerFlowAgent(conn, config=config, backend=backend),
            input_stream=writer,
            output_stream=reader,
            use_unstable_protocol=True,
        )
        return 0

    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
