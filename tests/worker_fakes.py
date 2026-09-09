"""worker 子进程里加载的假 DeerFlow 后端。

通过 ``DEERFLOW_ACP_WORKER_BACKEND=worker_fakes:build`` 注入，脚本从
``DEERFLOW_ACP_FAKE_STATE`` 指向的目录读写状态。**它跑在真实的 worker 子进程里**，
因此对它做的一切（阻塞、写副作用、写 checkpoint）都是真实的跨进程行为，
不是父进程里的 mock 自洽。

它必须能模拟被终止的旧执行体会造成的两类真实伤害：

* **工具副作用**：每隔一段时间往 ``side_effects.log`` 追加一行。旧 worker 若没死，
  取消之后这个文件还会继续长——这是「取消没有真正生效」的物证。
* **checkpoint 写入**：往 ``checkpoints/<thread_id>.json`` 追加条目，模拟 DeerFlow
  以 thread_id 为主键写状态。两个 worker 并发写同一 thread 就会在这里撞上。
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def state_dir() -> Path:
    return Path(os.environ["DEERFLOW_ACP_FAKE_STATE"])


class StatefulFakeBackend:
    """带持久状态与可观测副作用的假后端。

    行为由 ``DEERFLOW_ACP_FAKE_STATE/script.json`` 控制::

        {
          "events": [["messages-tuple", {...}]],
          "stall_before_first_yield_s": 0,   # 第一个 yield 之前阻塞（模拟卡在模型调用里）
          "side_effect_interval_s": 0.2,     # 阻塞期间每隔多久写一次副作用
          "raise": "RuntimeError: ...",      # stream 立即抛该异常
          "raise_on_close": "..."            # 生成器 close() 时抛该异常（消息含假秘密）
        }
    """

    def __init__(self, config: Any) -> None:
        self._dir = state_dir()
        script_path = self._dir / "script.json"
        self._script: dict[str, Any] = json.loads(script_path.read_text(encoding="utf-8")) if script_path.exists() else {}
        self._events = [tuple(item) for item in self._script.get("events", [])]
        self._stall = float(self._script.get("stall_before_first_yield_s", 0))
        self._side_effect_interval = float(self._script.get("side_effect_interval_s", 0.2))
        self._raise = self._script.get("raise")
        self._raise_on_close = self._script.get("raise_on_close")
        self._checkpoint_interval = float(self._script.get("checkpoint_interval_s", 0.2))

    # ------------------------------------------------------------------

    def _append_side_effect(self, thread_id: str) -> None:
        """模拟一次不可撤销的工具副作用（写外部文件）。"""
        with open(self._dir / "side_effects.log", "a", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()} {thread_id} {time.time():.4f}\n")
            fh.flush()

    def _append_checkpoint(self, thread_id: str) -> None:
        """模拟 DeerFlow 以 thread_id 为主键写 checkpoint。"""
        cp_dir = self._dir / "checkpoints"
        cp_dir.mkdir(exist_ok=True)
        path = cp_dir / f"{thread_id}.json"
        entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        entries.append({"pid": os.getpid(), "ts": time.time()})
        path.write_text(json.dumps(entries), encoding="utf-8")

    # ------------------------------------------------------------------

    def stream(self, message: str, *, thread_id: str) -> Iterator[tuple[str, dict[str, Any]]]:
        if self._raise:
            raise RuntimeError(self._raise)

        if self._script.get("pollute_stdout"):
            # 模拟 DeerFlow / LangGraph / C 扩展往 fd 1 打东西。
            # worker 已把 fd 1 抢走，这些必须全部落到 stderr，
            # 否则会把 ndJSON 消息流撕成畸形行。
            print("这行垃圾绝不能出现在 IPC 通道里")
            sys.stdout.write("再来一行\n")
            sys.stdout.flush()
            os.write(1, "裸 write 也不行\n".encode("utf-8"))

        script = self._script
        stall = self._stall
        side_interval = self._side_effect_interval
        cp_interval = self._checkpoint_interval
        raise_on_close = self._raise_on_close

        def gen() -> Iterator[tuple[str, dict[str, Any]]]:
            try:
                if stall:
                    # 关键：阻塞点在第一个 yield **之前**，且期间持续制造副作用。
                    # 这正是 DeerFlow 卡在模型调用或工具执行里的形态——生成器
                    # 交不出控制权，协作式取消够不着它。
                    deadline = time.time() + stall
                    next_side = time.time()
                    next_cp = time.time()
                    while time.time() < deadline:
                        now = time.time()
                        if side_interval and now >= next_side:
                            self._append_side_effect(thread_id)
                            next_side = now + side_interval
                        if cp_interval and now >= next_cp:
                            self._append_checkpoint(thread_id)
                            next_cp = now + cp_interval
                        time.sleep(0.02)
                for event_type, data in self._events:
                    payload = dict(data)
                    payload.setdefault("pid", os.getpid())
                    self._append_checkpoint(thread_id)
                    yield event_type, payload
            except GeneratorExit:
                if raise_on_close:
                    # 构造的假秘密：验证 close() 阶段的异常也不会泄露到 stderr
                    raise RuntimeError(raise_on_close) from None
                raise

        _ = script
        return gen()

    def thread_exists(self, thread_id: str) -> bool:
        return (self._dir / "checkpoints" / f"{thread_id}.json").exists()

    def history(self, thread_id: str) -> list[dict[str, Any]]:
        return []


def build(config: Any) -> StatefulFakeBackend:
    return StatefulFakeBackend(config)
