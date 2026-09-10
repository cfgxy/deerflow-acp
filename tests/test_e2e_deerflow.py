"""与本机 DeerFlow 的端到端测试。

默认跳过——需要真实模型凭据与本机 DeerFlow 部署，不能进 CI 常规流水线。
显式开启：

    DEERFLOW_ACP_E2E=1 pytest tests/test_e2e_deerflow.py -q

跑的是完整链路：真实 `deerflow-acp acp` 子进程 → 真实 `DeerFlowClient`
→ 真实模型调用 → 真实 LangGraph checkpointer。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("DEERFLOW_ACP_E2E") != "1",
    reason="需要本机 DeerFlow 与真实模型凭据；设 DEERFLOW_ACP_E2E=1 开启",
)

# 真实模型调用比契约测试慢一个数量级
TURN_TIMEOUT = float(os.environ.get("DEERFLOW_ACP_E2E_TIMEOUT", "300"))

# DeerFlow 只按**当前工作目录**查找 config.yaml（构造函数的 config_path 参数
# 在当前版本并不改变查找根），因此桥子进程必须在 DeerFlow 部署根下启动。
DEERFLOW_ROOT = os.environ.get("DEERFLOW_ACP_E2E_CWD", "/home/guxy/srv/deerflow")


class Bridge:
    def __init__(self, proc: subprocess.Popen) -> None:
        self.proc = proc
        self._next_id = 0

    def send(self, method: str, params: dict | None = None, *, notification: bool = False) -> int | None:
        message: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        if not notification:
            self._next_id += 1
            message["id"] = self._next_id
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()
        return message.get("id")

    def read_message(self, timeout: float) -> dict:
        assert self.proc.stdout is not None
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                stderr = self.proc.stderr.read() if self.proc.stderr else ""
                raise AssertionError(f"桥进程提前退出，stderr 尾部：{stderr[-2000:]}")
            line = line.strip()
            if line:
                return json.loads(line)
        raise AssertionError("读取超时")

    def await_response(self, request_id: int, timeout: float) -> dict:
        notifications: list[dict] = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            message = self.read_message(timeout=max(0.1, deadline - time.time()))
            if message.get("id") == request_id:
                message["_notifications"] = notifications
                return message
            if "id" not in message:
                notifications.append(message)
        raise AssertionError("等待响应超时")

    def initialize(self) -> dict:
        rid = self.send(
            "initialize",
            {"protocolVersion": 1, "clientInfo": {"name": "deerflow-acp-e2e", "version": "0"}, "clientCapabilities": {}},
        )
        return self.await_response(rid, timeout=60)

    def new_session(self) -> str:
        rid = self.send("session/new", {"cwd": os.getcwd(), "mcpServers": []})
        return self.await_response(rid, timeout=60)["result"]["sessionId"]

    def close(self) -> int:
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)
            raise AssertionError("桥进程未在 stdin 断连后退出")
        return self.proc.returncode


@pytest.fixture
def bridge():
    procs: list[Bridge] = []

    def factory(env_extra: dict | None = None) -> Bridge:
        proc = subprocess.Popen(
            [sys.executable, "-m", "deerflow_acp.cli", "acp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=DEERFLOW_ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1", **(env_extra or {})},
        )
        b = Bridge(proc)
        procs.append(b)
        return b

    yield factory

    for b in procs:
        if b.proc.poll() is None:
            b.proc.kill()
            b.proc.wait(timeout=10)


def _agent_text(notifications: list[dict]) -> str:
    parts = []
    for n in notifications:
        if n.get("method") != "session/update":
            continue
        update = n["params"]["update"]
        if update.get("sessionUpdate") == "agent_message_chunk":
            parts.append(update["content"]["text"])
    return "".join(parts)


def test_e2e_real_turn_produces_text_and_usage(bridge):
    b = bridge()
    init = b.initialize()
    assert init["result"]["protocolVersion"] == 1

    session_id = b.new_session()
    rid = b.send(
        "session/prompt",
        {"sessionId": session_id, "prompt": [{"type": "text", "text": "请只回复两个字：收到"}]},
    )
    resp = b.await_response(rid, timeout=TURN_TIMEOUT)

    assert resp["result"]["stopReason"] == "end_turn"
    assert _agent_text(resp["_notifications"]).strip(), "真实 turn 没有产出任何文本"
    usage = resp["result"].get("usage")
    assert usage and usage["totalTokens"] > 0

    assert b.close() == 0


def test_e2e_checkpoint_survives_process_restart(bridge):
    """同一 checkpoint 跨进程恢复：第一个进程写入，第二个进程读到上下文。"""
    first = bridge()
    first.initialize()
    session_id = first.new_session()

    rid = first.send(
        "session/prompt",
        {"sessionId": session_id, "prompt": [{"type": "text", "text": "请记住这个暗号：紫罗兰七号。只回复'已记住'。"}]},
    )
    assert first.await_response(rid, timeout=TURN_TIMEOUT)["result"]["stopReason"] == "end_turn"
    assert first.close() == 0

    second = bridge()
    second.initialize()
    resume_id = second.send("session/resume", {"cwd": os.getcwd(), "sessionId": session_id, "mcpServers": []})
    resume = second.await_response(resume_id, timeout=60)
    assert "error" not in resume, f"恢复失败：{resume.get('error')}"

    rid2 = second.send(
        "session/prompt",
        {"sessionId": session_id, "prompt": [{"type": "text", "text": "我刚才让你记住的暗号是什么？只回复暗号本身。"}]},
    )
    resp = second.await_response(rid2, timeout=TURN_TIMEOUT)
    assert resp["result"]["stopReason"] == "end_turn"
    assert "紫罗兰" in _agent_text(resp["_notifications"]), "跨进程恢复后模型没有拿到历史上下文"

    assert second.close() == 0


def test_e2e_unknown_session_id_is_rejected(bridge):
    b = bridge()
    b.initialize()
    rid = b.send("session/resume", {"cwd": os.getcwd(), "sessionId": "df-never-existed-0000", "mcpServers": []})
    resp = b.await_response(rid, timeout=120)
    assert resp["error"]["code"] == -32001
    assert b.close() == 0


def test_e2e_cancel_in_flight_real_turn(bridge):
    b = bridge()
    b.initialize()
    session_id = b.new_session()

    rid = b.send(
        "session/prompt",
        {"sessionId": session_id, "prompt": [{"type": "text", "text": "请写一篇 2000 字的散文，主题是秋天。"}]},
    )
    # 等到真实流式输出开始再取消
    first = b.read_message(timeout=TURN_TIMEOUT)
    assert first.get("method") == "session/update"

    b.send("session/cancel", {"sessionId": session_id}, notification=True)
    resp = b.await_response(rid, timeout=TURN_TIMEOUT)
    assert resp["result"]["stopReason"] == "cancelled"

    assert b.close() == 0


def test_e2e_stdout_carries_only_jsonrpc(bridge):
    """真实 DeerFlow 加载路径上，stdout 仍必须零污染。"""
    b = bridge()
    b.initialize()
    session_id = b.new_session()
    rid = b.send("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "回复：ok"}]})
    b.await_response(rid, timeout=TURN_TIMEOUT)

    assert b.proc.stdin is not None
    b.proc.stdin.close()
    remaining = b.proc.stdout.read() if b.proc.stdout else ""
    b.proc.wait(timeout=60)

    for line in remaining.splitlines():
        if not line.strip():
            continue
        assert json.loads(line).get("jsonrpc") == "2.0"


def _worker_children(bridge_pid: int) -> list[int]:
    """桥进程直接派生的 worker 子进程 PID 列表。"""
    out = subprocess.run(["pgrep", "-P", str(bridge_pid)], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def _pgid_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def test_e2e_cancel_kills_real_worker_process_group(bridge):
    """真实 DeerFlow 卡在模型调用里时，取消超时必须终止整个 worker 进程组。

    宽限期压到 0.5s，确保走的是强制终止分支而非协作退出。
    """
    b = bridge({"DEERFLOW_ACP_CANCEL_GRACE_SECONDS": "0.5"})
    b.initialize()
    session_id = b.new_session()

    rid = b.send(
        "session/prompt",
        {"sessionId": session_id, "prompt": [{"type": "text", "text": "请写一篇 3000 字的散文，主题是秋天的黄昏。"}]},
    )
    # 等 worker 起来并拿到它的进程组
    deadline = time.time() + 60
    workers: list[int] = []
    while time.time() < deadline and not workers:
        workers = _worker_children(b.proc.pid)
        if not workers:
            time.sleep(0.1)
    assert workers, "真实路径没有派生 worker 子进程"
    worker_pid = workers[0]
    worker_pgid = os.getpgid(worker_pid)
    assert worker_pgid != os.getpgid(b.proc.pid), "worker 没有独立进程组，killpg 会误伤桥自己"

    # 等真实流式输出开始，确保 DeerFlow 已经进入模型调用
    first = b.read_message(timeout=TURN_TIMEOUT)
    assert first.get("method") == "session/update"

    b.send("session/cancel", {"sessionId": session_id}, notification=True)
    resp = b.await_response(rid, timeout=TURN_TIMEOUT)
    assert resp["result"]["stopReason"] == "cancelled"

    # 响应返回时进程组就必须已经被回收
    assert not _pgid_alive(worker_pgid), f"worker 进程组 {worker_pgid}（pid={worker_pid}）在取消返回后仍存活"

    assert b.close() == 0


def test_e2e_resume_same_session_after_forced_kill(bridge):
    """强制终止旧 worker 后，同一 session 必须能从已落 checkpoint 继续。

    这是进程隔离方案的核心风险点：SIGKILL 可能撕裂 checkpointer 写入。
    """
    b = bridge({"DEERFLOW_ACP_CANCEL_GRACE_SECONDS": "0.5"})
    b.initialize()
    session_id = b.new_session()

    rid = b.send(
        "session/prompt",
        {"sessionId": session_id, "prompt": [{"type": "text", "text": "请记住这个暗号：紫罗兰七号。然后写一篇 3000 字的散文，主题是秋天的黄昏。"}]},
    )
    first = b.read_message(timeout=TURN_TIMEOUT)
    assert first.get("method") == "session/update"

    workers = _worker_children(b.proc.pid)
    assert workers
    worker_pgid = os.getpgid(workers[0])

    b.send("session/cancel", {"sessionId": session_id}, notification=True)
    assert b.await_response(rid, timeout=TURN_TIMEOUT)["result"]["stopReason"] == "cancelled"
    assert not _pgid_alive(worker_pgid)

    # 同 session 立刻新一轮：既要能起来，也要拿到强杀前落下的上下文
    rid2 = b.send(
        "session/prompt",
        {"sessionId": session_id, "prompt": [{"type": "text", "text": "我刚才让你记住的暗号是什么？只回复暗号本身。"}]},
    )
    resp = b.await_response(rid2, timeout=TURN_TIMEOUT)
    assert "error" not in resp, f"强杀后同 session 无法继续：{resp.get('error')}"
    assert resp["result"]["stopReason"] == "end_turn"
    assert "紫罗兰" in _agent_text(resp["_notifications"]), "强制终止破坏了 checkpoint 上下文"

    assert b.close() == 0


def test_e2e_no_orphan_worker_after_sigterm(bridge):
    """桥收到 SIGTERM 时，正在跑的真实 worker 进程组不得残留。"""
    b = bridge({"DEERFLOW_ACP_CANCEL_GRACE_SECONDS": "0.5"})
    b.initialize()
    session_id = b.new_session()
    b.send(
        "session/prompt",
        {"sessionId": session_id, "prompt": [{"type": "text", "text": "请写一篇 3000 字的散文，主题是秋天的黄昏。"}]},
    )
    first = b.read_message(timeout=TURN_TIMEOUT)
    assert first.get("method") == "session/update"

    workers = _worker_children(b.proc.pid)
    assert workers
    worker_pgid = os.getpgid(workers[0])

    b.proc.send_signal(signal.SIGTERM)
    b.proc.wait(timeout=60)

    # worker 与桥在同一会话树下：桥退出后进程组必须随之消失
    deadline = time.time() + 15
    while time.time() < deadline and _pgid_alive(worker_pgid):
        time.sleep(0.2)
    assert not _pgid_alive(worker_pgid), f"桥被 SIGTERM 后 worker 进程组 {worker_pgid} 成了孤儿"


def test_e2e_no_orphan_processes_after_exit(bridge):
    b = bridge()
    b.initialize()
    session_id = b.new_session()
    rid = b.send("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "回复：ok"}]})
    b.await_response(rid, timeout=TURN_TIMEOUT)
    pid = b.proc.pid
    assert b.close() == 0

    children = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True)
    assert children.stdout.strip() == ""
