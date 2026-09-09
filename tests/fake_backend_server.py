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
          "raise": "RuntimeError", # 若设置，stream 立即抛该异常（消息为此字符串）
          "raise_backend_unavailable": "...",  # 改抛 BackendUnavailableError
          "stall_before_first_yield_s": 0      # 在第一个 yield 之前阻塞这么久
        }
    """

    def __init__(self, script: dict[str, Any]) -> None:
        self._events = [tuple(item) for item in script.get("events", [])]
        self._threads = script.get("threads", {})
        self._delay = float(script.get("delay_ms", 0)) / 1000.0
        self._repeat = int(script.get("repeat", 1))
        self._raise = script.get("raise")
        self._raise_unavailable = script.get("raise_backend_unavailable")
        # 模拟「模型/工具调用卡在 next(generator) 内部」：阻塞点在第一个 yield 之前，
        # 工作线程既看不到取消标志，也发不出完成信号
        self._stall = float(script.get("stall_before_first_yield_s", 0))
        # 通过一个文件标记生成器是否被 close()，让父进程可以断言协作式取消
        self._closed_marker = os.environ.get("DEERFLOW_ACP_FAKE_CLOSED_MARKER")
        # 往 fd 1 打垃圾，验证「后端污染 stdout」不会撕裂承载在 fd 1 上的通道
        self._pollute_stdout = bool(script.get("pollute_stdout"))

    def stream(self, message: str, *, thread_id: str) -> Iterator[tuple[str, dict[str, Any]]]:
        if self._pollute_stdout:
            print("这行垃圾绝不能出现在 IPC 通道里")
            sys.stdout.write("再来一行\n")
            sys.stdout.flush()
            os.write(1, "裸 write 也不行\n".encode())
        if self._raise_unavailable:
            from deerflow_acp.backend import BackendUnavailableError

            raise BackendUnavailableError(self._raise_unavailable)
        if self._raise:
            raise RuntimeError(self._raise)

        def gen() -> Iterator[tuple[str, dict[str, Any]]]:
            try:
                if self._stall:
                    time.sleep(self._stall)
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
        if self._raise_unavailable:
            from deerflow_acp.backend import BackendUnavailableError

            raise BackendUnavailableError(self._raise_unavailable)
        return thread_id in self._threads

    def history(self, thread_id: str) -> list[dict[str, Any]]:
        return list(self._threads.get(thread_id, []))


def build_backend(config: Any) -> ScriptedBackend:
    """worker 子进程侧的脚本后端工厂。

    通过 ``DEERFLOW_ACP_WORKER_BACKEND=fake_backend_server:build_backend`` 注入。
    脚本对象跨不过进程边界，所以 worker 里重新读一遍同一个脚本文件。
    """
    script_path = os.environ["DEERFLOW_ACP_FAKE_SCRIPT"]
    with open(script_path, encoding="utf-8") as fh:
        backend = ScriptedBackend(json.load(fh))

    # 把 worker 自己的 pid/pgid 落盘：契约测试据此断言取消超时后进程组真的退出了。
    # 这两个数字在 JSON-RPC 报文里不存在，只能由 worker 侧自报。
    pid_path = os.environ.get("DEERFLOW_ACP_FAKE_WORKER_PID")
    if pid_path:
        with open(pid_path, "w", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()} {os.getpgid(0)}")
    return backend


def main() -> int:
    from deerflow_acp.cli import isolate_stdout, serve
    from deerflow_acp.config import BridgeConfig
    from deerflow_acp.logging_setup import configure_logging, redirect_root_logging_to_stderr

    configure_logging("WARNING")
    redirect_root_logging_to_stderr("WARNING")

    script_path = os.environ["DEERFLOW_ACP_FAKE_SCRIPT"]
    with open(script_path, encoding="utf-8") as fh:
        script = json.load(fh)

    backend = ScriptedBackend(script)
    config = BridgeConfig.from_env()

    # 默认走进程内执行（脚本后端是本进程里的对象）。设了这个变量则改走真实的
    # worker 子进程路径——契约测试用它验证生产链路上的进程隔离与取消。
    runner = None
    if os.environ.get("DEERFLOW_ACP_FAKE_USE_WORKER"):
        from deerflow_acp.runner import SubprocessTurnRunner

        runner = SubprocessTurnRunner(config)

    protocol_stdout = isolate_stdout()

    # 故意在协议启动后往 fd 1 写垃圾：验证 stdout 隔离确实有效。
    if os.environ.get("DEERFLOW_ACP_FAKE_POLLUTE"):
        print("这行垃圾绝不能出现在 JSON-RPC 通道里")
        sys.stdout.write("再来一行\n")
        sys.stdout.flush()
        os.write(1, "裸 write 也不行\n".encode())

    # 走真实的 cli.serve：信号处理、宽限期与关停时的 worker 回收都在那里，
    # 自己另起一套 run_agent 就把这些行为排除在契约测试之外了。
    return asyncio.run(serve(config, protocol_stdout, backend=backend, runner=runner))


if __name__ == "__main__":
    raise SystemExit(main())
