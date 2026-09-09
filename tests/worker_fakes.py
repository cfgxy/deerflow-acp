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
import subprocess
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
          "raise_on_close": "...",           # 生成器 close() 时抛该异常（消息含假秘密）
          "tool_child_lifetime_s": 0         # 派生一个「工具子进程」，活这么久后自己退出
        }

    ``tool_child_lifetime_s`` 模拟 DeerFlow 的工具派生出的孙进程：它留在 worker
    的**同一个进程组**里，但**不是**桥的子进程，因此 ``proc.wait()`` 看不到它。
    worker 主进程先退出、这个孙进程还在跑，正是「主 PID 退出 ≠ 进程组排空」的
    真实形态；它持续写副作用，让「会话被提前释放」变成可观测的物证。
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
        self._tool_child_lifetime = float(self._script.get("tool_child_lifetime_s", 0))

    # ------------------------------------------------------------------

    def _append_side_effect(self, thread_id: str) -> None:
        """模拟一次不可撤销的工具副作用（写外部文件）。"""
        with open(self._dir / "side_effects.log", "a", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()} {thread_id} {time.time():.4f}\n")
            fh.flush()

    def _append_checkpoint(self, thread_id: str) -> None:
        """模拟 DeerFlow 以 thread_id 为主键写 checkpoint。

        写入必须**原子**：这个后端随时会被 SIGKILL 打断，若直接覆写目标文件，
        强杀落在「截断」与「写入」之间就会留下 0 字节文件，把「旧执行体是否
        改写了既有 checkpoint」这个观测点毁掉。先写同目录临时文件再 ``os.replace``，
        保证任何时刻读到的都是某个完整版本。
        """
        cp_dir = self._dir / "checkpoints"
        cp_dir.mkdir(exist_ok=True)
        path = cp_dir / f"{thread_id}.json"
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        entries = json.loads(raw) if raw.strip() else []
        entries.append({"pid": os.getpid(), "ts": time.time()})
        tmp = cp_dir / f"{thread_id}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(entries), encoding="utf-8")
        os.replace(tmp, path)

    # ------------------------------------------------------------------

    def _spawn_tool_child(self) -> int:
        """派生一个留在同一进程组里的「工具子进程」。

        不传 ``start_new_session``：它必须继承 worker 的 PGID，否则就测不到
        「主 PID 退出但进程组仍有活口」。它每 50ms 往 ``tool_child.log`` 追加一行，
        因此「桥是否真的等到进程组排空」在文件里直接可见。
        """
        log = self._dir / "tool_child.log"
        code = (
            "import os,time,sys\n"
            f"deadline=time.time()+{self._tool_child_lifetime!r}\n"
            "while time.time()<deadline:\n"
            f"    open({str(log)!r},'a').write(f'{{os.getpid()}} {{time.time():.4f}}\\n')\n"
            "    time.sleep(0.05)\n"
        )
        proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        (self._dir / "tool_child.pid").write_text(str(proc.pid), encoding="utf-8")
        return proc.pid

    def stream(self, message: str, *, thread_id: str) -> Iterator[tuple[str, dict[str, Any]]]:
        if self._raise:
            raise RuntimeError(self._raise)

        if self._tool_child_lifetime:
            self._spawn_tool_child()

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
