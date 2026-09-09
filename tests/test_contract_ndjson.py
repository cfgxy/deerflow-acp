"""双端 ndJSON 契约测试。

用真实子进程跑 ACP server，父进程手写 JSON-RPC over ndJSON 报文，
既验证桥回复的报文形状，也验证 stdout 通道的零污染。

这一层刻意不使用 ACP SDK 的 client：只有手写报文才能覆盖畸形 JSON、
未知方法、未知 session 这些 SDK 客户端根本发不出去的输入。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
SERVER = TESTS_DIR / "fake_backend_server.py"

TEXT_EVENTS = [
    ["messages-tuple", {"type": "ai", "content": "你好", "id": "m1"}],
    ["end", {"usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}}],
]


class Peer:
    """一个跑在子进程里的 ACP server，父进程直接读写 ndJSON。"""

    def __init__(self, proc: subprocess.Popen, closed_marker: Path) -> None:
        self.proc = proc
        self.closed_marker = closed_marker
        self._next_id = 0

    def send_raw(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def send(self, method: str, params: dict | None = None, *, notification: bool = False) -> int | None:
        message: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        if notification:
            self.send_raw(json.dumps(message))
            return None
        self._next_id += 1
        message["id"] = self._next_id
        self.send_raw(json.dumps(message))
        return self._next_id

    def read_message(self, timeout: float = 20.0) -> dict:
        assert self.proc.stdout is not None
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise AssertionError("server 提前关闭了 stdout")
            line = line.strip()
            if not line:
                continue
            # 契约要求：stdout 每一行都必须是合法 JSON-RPC，没有例外
            return json.loads(line)
        raise AssertionError("读取 server 消息超时")

    def await_response(self, request_id: int, timeout: float = 20.0) -> dict:
        """读到指定 id 的响应，沿途收集 notification。"""
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
            {
                "protocolVersion": 1,
                "clientInfo": {"name": "contract-test", "version": "0"},
                "clientCapabilities": {},
            },
        )
        return self.await_response(rid)

    def new_session(self, cwd: str = "/tmp") -> str:
        rid = self.send("session/new", {"cwd": cwd, "mcpServers": []})
        resp = self.await_response(rid)
        return resp["result"]["sessionId"]

    def close(self) -> tuple[int, str]:
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
            raise AssertionError("stdin 断连后 server 未在 15s 内退出")
        stderr = self.proc.stderr.read() if self.proc.stderr else ""
        return self.proc.returncode, stderr


@pytest.fixture
def peer(tmp_path):
    procs: list[Peer] = []

    def factory(script: dict, *, env_extra: dict | None = None) -> Peer:
        script_path = tmp_path / f"script-{len(procs)}.json"
        script_path.write_text(json.dumps(script), encoding="utf-8")
        closed_marker = tmp_path / f"closed-{len(procs)}.marker"

        env = dict(os.environ)
        env["DEERFLOW_ACP_FAKE_SCRIPT"] = str(script_path)
        env["DEERFLOW_ACP_FAKE_CLOSED_MARKER"] = str(closed_marker)
        env["PYTHONUNBUFFERED"] = "1"
        env.update(env_extra or {})

        proc = subprocess.Popen(
            [sys.executable, str(SERVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        p = Peer(proc, closed_marker)
        procs.append(p)
        return p

    yield factory

    for p in procs:
        if p.proc.poll() is None:
            p.proc.kill()
            p.proc.wait(timeout=5)


# ----------------------------------------------------------------------
# 主链路
# ----------------------------------------------------------------------


def test_full_turn_initialize_new_prompt_update_end(peer):
    p = peer({"events": TEXT_EVENTS})

    init = p.initialize()
    assert init["jsonrpc"] == "2.0"
    assert init["result"]["protocolVersion"] == 1
    assert init["result"]["agentInfo"]["name"] == "deerflow-acp"
    assert init["result"]["agentCapabilities"]["loadSession"] is True

    session_id = p.new_session()

    rid = p.send("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "你好"}]})
    resp = p.await_response(rid)

    assert resp["result"]["stopReason"] == "end_turn"
    assert resp["result"]["usage"]["totalTokens"] == 7

    updates = [n for n in resp["_notifications"] if n.get("method") == "session/update"]
    assert [u["params"]["update"]["sessionUpdate"] for u in updates] == ["agent_message_chunk"]
    assert updates[0]["params"]["sessionId"] == session_id
    assert updates[0]["params"]["update"]["content"]["text"] == "你好"

    code, _ = p.close()
    assert code == 0


def test_tool_call_lifecycle_over_the_wire(peer):
    p = peer(
        {
            "events": [
                ["messages-tuple", {"type": "ai", "content": "", "id": "m1", "tool_calls": [{"name": "web_search", "args": {"q": "x"}, "id": "t1"}]}],
                ["messages-tuple", {"type": "tool", "content": "命中", "name": "web_search", "tool_call_id": "t1", "id": "m2"}],
                ["end", {"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}],
            ]
        }
    )
    p.initialize()
    session_id = p.new_session()
    rid = p.send("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "查"}]})
    resp = p.await_response(rid)

    updates = [n["params"]["update"] for n in resp["_notifications"] if n.get("method") == "session/update"]
    assert [u["sessionUpdate"] for u in updates] == ["tool_call", "tool_call_update"]
    assert updates[0]["toolCallId"] == "t1"
    assert updates[0]["kind"] == "fetch"
    assert updates[0]["status"] == "in_progress"
    assert updates[1]["status"] == "completed"
    p.close()


def test_thought_chunk_over_the_wire(peer):
    p = peer(
        {
            "events": [
                ["messages-tuple", {"type": "ai", "content": "答", "id": "m1", "additional_kwargs": {"reasoning_content": "想"}}],
                ["end", {"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}],
            ]
        }
    )
    p.initialize()
    session_id = p.new_session()
    rid = p.send("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "q"}]})
    resp = p.await_response(rid)
    kinds = [n["params"]["update"]["sessionUpdate"] for n in resp["_notifications"] if n.get("method") == "session/update"]
    assert kinds == ["agent_thought_chunk", "agent_message_chunk"]
    p.close()


# ----------------------------------------------------------------------
# 恢复
# ----------------------------------------------------------------------


def test_session_resume_known_thread_succeeds(peer):
    p = peer({"events": TEXT_EVENTS, "threads": {"df-known": [{"type": "ai", "content": "旧"}]}})
    p.initialize()
    rid = p.send("session/resume", {"cwd": "/tmp", "sessionId": "df-known", "mcpServers": []})
    resp = p.await_response(rid)
    assert "error" not in resp

    # 恢复后可以直接继续 prompt
    rid2 = p.send("session/prompt", {"sessionId": "df-known", "prompt": [{"type": "text", "text": "继续"}]})
    assert p.await_response(rid2)["result"]["stopReason"] == "end_turn"
    p.close()


def test_session_resume_unknown_thread_is_rejected(peer):
    p = peer({"events": TEXT_EVENTS, "threads": {}})
    p.initialize()
    rid = p.send("session/resume", {"cwd": "/tmp", "sessionId": "df-missing", "mcpServers": []})
    resp = p.await_response(rid)
    assert resp["error"]["code"] == -32001
    assert resp["error"]["data"]["sessionId"] == "df-missing"

    # 被拒后不得留下可用会话
    rid2 = p.send("session/prompt", {"sessionId": "df-missing", "prompt": [{"type": "text", "text": "x"}]})
    assert p.await_response(rid2)["error"]["code"] == -32001
    p.close()


def test_session_load_replays_history(peer):
    p = peer(
        {
            "events": TEXT_EVENTS,
            "threads": {"df-known": [{"type": "human", "content": "问"}, {"type": "ai", "content": "答"}]},
        }
    )
    p.initialize()
    rid = p.send("session/load", {"cwd": "/tmp", "sessionId": "df-known", "mcpServers": []})
    resp = p.await_response(rid)
    assert "error" not in resp
    kinds = [n["params"]["update"]["sessionUpdate"] for n in resp["_notifications"] if n.get("method") == "session/update"]
    assert kinds == ["user_message_chunk", "agent_message_chunk"]
    p.close()


# ----------------------------------------------------------------------
# 取消
# ----------------------------------------------------------------------


def test_cancel_in_flight_turn_returns_cancelled(peer):
    p = peer({"events": [["messages-tuple", {"type": "ai", "content": "x", "id": "m"}]], "repeat": 500, "delay_ms": 10})
    p.initialize()
    session_id = p.new_session()

    rid = p.send("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "长任务"}]})
    # 等到确实开始流式输出，再取消——避免竞态导致取消发在 turn 开始前
    first = p.read_message()
    assert first["method"] == "session/update"

    p.send("session/cancel", {"sessionId": session_id}, notification=True)
    resp = p.await_response(rid, timeout=30)

    assert resp["result"]["stopReason"] == "cancelled"
    # 协作式取消真的关闭了生成器
    deadline = time.time() + 5
    while time.time() < deadline and not p.closed_marker.exists():
        time.sleep(0.05)
    assert p.closed_marker.exists(), "DeerFlow 生成器未被 close()"

    code, _ = p.close()
    assert code == 0


def test_cancel_race_before_turn_starts_is_harmless(peer):
    """取消竞态：cancel 早于 prompt 到达，不得让后续 turn 永久失效。"""
    p = peer({"events": TEXT_EVENTS})
    p.initialize()
    session_id = p.new_session()

    p.send("session/cancel", {"sessionId": session_id}, notification=True)
    rid = p.send("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]})
    resp = p.await_response(rid)
    # run_turn 开始时会清掉陈旧的取消标志，因此本轮应正常结束
    assert resp["result"]["stopReason"] == "end_turn"
    p.close()


def test_cancel_for_unknown_session_does_not_crash_server(peer):
    p = peer({"events": TEXT_EVENTS})
    p.initialize()
    p.send("session/cancel", {"sessionId": "df-nope"}, notification=True)

    session_id = p.new_session()
    rid = p.send("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]})
    assert p.await_response(rid)["result"]["stopReason"] == "end_turn"
    p.close()


# ----------------------------------------------------------------------
# 协议层错误
# ----------------------------------------------------------------------


def test_malformed_json_gets_parse_error_and_connection_survives(peer):
    p = peer({"events": TEXT_EVENTS})
    p.initialize()

    p.send_raw("{ 这不是 JSON")
    error = p.read_message()
    assert error["error"]["code"] == -32700

    # 连接必须存活：后续正常请求仍要能跑通
    session_id = p.new_session()
    rid = p.send("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]})
    assert p.await_response(rid)["result"]["stopReason"] == "end_turn"
    p.close()


def test_unknown_method_returns_method_not_found(peer):
    p = peer({"events": TEXT_EVENTS})
    p.initialize()
    rid = p.send("session/no_such_method", {})
    assert p.await_response(rid)["error"]["code"] == -32601
    p.close()


def test_prompt_unknown_session_returns_unknown_session_error(peer):
    p = peer({"events": TEXT_EVENTS})
    p.initialize()
    rid = p.send("session/prompt", {"sessionId": "df-nope", "prompt": [{"type": "text", "text": "hi"}]})
    resp = p.await_response(rid)
    assert resp["error"]["code"] == -32001
    p.close()


def test_prompt_with_missing_params_returns_invalid_params(peer):
    p = peer({"events": TEXT_EVENTS})
    p.initialize()
    rid = p.send("session/prompt", {"sessionId": "df-nope"})
    assert p.await_response(rid)["error"]["code"] == -32602
    p.close()


def test_non_text_content_block_is_rejected(peer):
    p = peer({"events": TEXT_EVENTS})
    p.initialize()
    session_id = p.new_session()
    rid = p.send(
        "session/prompt",
        {"sessionId": session_id, "prompt": [{"type": "image", "data": "aGk=", "mimeType": "image/png"}]},
    )
    resp = p.await_response(rid)
    assert resp["error"]["code"] == -32602
    p.close()


def test_backend_failure_becomes_internal_error_without_leaking_message(peer):
    p = peer({"events": [], "raise": "sk-secret-should-not-leak"})
    p.initialize()
    session_id = p.new_session()
    rid = p.send("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]})
    resp = p.await_response(rid)

    assert resp["error"]["code"] == -32603
    serialized = json.dumps(resp, ensure_ascii=False)
    assert "sk-secret-should-not-leak" not in serialized
    assert resp["error"]["data"]["errorType"] == "RuntimeError"
    p.close()


def test_authenticate_is_method_not_found(peer):
    p = peer({"events": TEXT_EVENTS})
    p.initialize()
    rid = p.send("authenticate", {"methodId": "x"})
    assert p.await_response(rid)["error"]["code"] == -32601
    p.close()


# ----------------------------------------------------------------------
# stdout 纪律与进程生命周期
# ----------------------------------------------------------------------


def test_stdout_carries_only_jsonrpc_even_when_backend_prints(peer):
    p = peer({"events": TEXT_EVENTS}, env_extra={"DEERFLOW_ACP_FAKE_POLLUTE": "1"})

    p.initialize()
    session_id = p.new_session()
    rid = p.send("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]})
    p.await_response(rid)

    code, stderr = p.close()
    assert code == 0
    # 垃圾输出必须落到 stderr
    assert "这行垃圾绝不能出现在 JSON-RPC 通道里" in stderr
    assert "裸 write 也不行" in stderr


def test_every_stdout_line_is_valid_jsonrpc(peer):
    """把整段 stdout 收下来逐行解析，任何一行不是 JSON-RPC 即失败。"""
    p = peer({"events": TEXT_EVENTS}, env_extra={"DEERFLOW_ACP_FAKE_POLLUTE": "1"})
    p.initialize()
    session_id = p.new_session()
    rid = p.send("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]})
    p.await_response(rid)

    assert p.proc.stdin is not None
    p.proc.stdin.close()
    remaining = p.proc.stdout.read() if p.proc.stdout else ""
    p.proc.wait(timeout=15)

    for line in remaining.splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line)
        assert parsed.get("jsonrpc") == "2.0"


def test_stdin_close_exits_process_cleanly(peer):
    p = peer({"events": TEXT_EVENTS})
    p.initialize()
    code, _ = p.close()
    assert code == 0


def test_stdin_close_during_turn_still_exits(peer):
    """stdin 在 turn 进行中断连：进程必须退出，不得留守。"""
    p = peer({"events": [["messages-tuple", {"type": "ai", "content": "x", "id": "m"}]], "repeat": 300, "delay_ms": 10})
    p.initialize()
    session_id = p.new_session()
    p.send("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "长任务"}]})
    p.read_message()  # 确认 turn 已开始

    assert p.proc.stdin is not None
    p.proc.stdin.close()
    if p.proc.stdout is not None:
        p.proc.stdout.read()
    p.proc.wait(timeout=30)
    assert p.proc.returncode is not None


def test_sigterm_terminates_without_orphan_children(peer):
    import signal

    p = peer({"events": [["messages-tuple", {"type": "ai", "content": "x", "id": "m"}]], "repeat": 500, "delay_ms": 10})
    p.initialize()
    session_id = p.new_session()
    p.send("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "长任务"}]})
    p.read_message()

    p.proc.send_signal(signal.SIGTERM)
    p.proc.wait(timeout=30)
    assert p.proc.returncode is not None

    # 进程组里不应残留子进程
    children = subprocess.run(
        ["pgrep", "-P", str(p.proc.pid)], capture_output=True, text=True
    )
    assert children.stdout.strip() == ""


# ----------------------------------------------------------------------
# 取消：后端卡在第一个 yield 之前（真子进程）
# ----------------------------------------------------------------------

# 构造的假秘密：形态逼真但完全无效
CONTRACT_FAKE_KEY = "sk-proj-Ab3xQ9zK7mNpR2vT5wY8cE1dF4gH6jL0oP"
CONTRACT_FAKE_DSN = "postgres://dfuser:Sup3rS3cretPw@127.0.0.1:5432/deerflow"


def test_cancel_returns_within_grace_when_backend_stalls_before_first_yield(peer):
    """最难的一种取消：后端卡在 next(generator) 内部，永远到不了 yield 边界。

    没有事件循环侧计时的话，session/prompt 会永久挂起——这正是宽限期存在的理由。
    """
    p = peer(
        {"events": TEXT_EVENTS, "stall_before_first_yield_s": 30},
        env_extra={"DEERFLOW_ACP_CANCEL_GRACE_SECONDS": "0.5"},
    )
    p.initialize()
    session_id = p.new_session()

    rid = p.send("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "你好"}]})
    time.sleep(0.3)  # 让 turn 真的进到后端阻塞里
    p.send("session/cancel", {"sessionId": session_id}, notification=True)

    started = time.time()
    resp = p.await_response(rid, timeout=10)
    elapsed = time.time() - started

    assert resp["result"]["stopReason"] == "cancelled"
    assert elapsed < 8, f"取消后 {elapsed:.1f}s 才返回，宽限期没有生效"

    p.proc.kill()
    p.proc.wait(timeout=5)


def test_session_reusable_after_stalled_cancel(peer):
    """被弃用的旧 worker 不得让同一 session 的后续 prompt 一直吃 -32011。"""
    p = peer(
        {"events": TEXT_EVENTS, "stall_before_first_yield_s": 3},
        env_extra={"DEERFLOW_ACP_CANCEL_GRACE_SECONDS": "0.5"},
    )
    p.initialize()
    session_id = p.new_session()

    rid = p.send("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "第一问"}]})
    time.sleep(0.3)
    p.send("session/cancel", {"sessionId": session_id}, notification=True)
    first = p.await_response(rid, timeout=10)
    assert first["result"]["stopReason"] == "cancelled"

    # 同一 session 立刻再来一轮：不得被判为「turn 正在执行」
    rid2 = p.send("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "第二问"}]})
    second = p.await_response(rid2, timeout=20)
    assert "error" not in second, f"后续 turn 被旧 worker 拖累：{second.get('error')}"
    assert second["result"]["stopReason"] == "end_turn"

    p.proc.kill()
    p.proc.wait(timeout=5)


# ----------------------------------------------------------------------
# 秘密不出进程：JSON-RPC 与 stderr 双通道（真子进程）
# ----------------------------------------------------------------------


def test_secret_never_reaches_jsonrpc_or_stderr_on_backend_error(peer):
    p = peer({"events": TEXT_EVENTS, "raise": f"provider 拒绝：{CONTRACT_FAKE_KEY}"})
    p.initialize()
    session_id = p.new_session()

    rid = p.send("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "你好"}]})
    resp = p.await_response(rid)

    wire = json.dumps(resp, ensure_ascii=False)
    assert CONTRACT_FAKE_KEY not in wire
    assert "RuntimeError" in wire, "脱敏不得把错误分类也抹掉，否则不可诊断"

    _code, stderr = p.close()
    assert CONTRACT_FAKE_KEY not in stderr, "秘密从 stderr 漏出去了"


def test_secret_never_reaches_jsonrpc_on_backend_unavailable(peer):
    p = peer({"events": TEXT_EVENTS, "raise_backend_unavailable": f"连不上 {CONTRACT_FAKE_DSN}"})
    p.initialize()
    session_id = p.new_session()

    rid = p.send("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "你好"}]})
    resp = p.await_response(rid)

    wire = json.dumps(resp, ensure_ascii=False)
    assert "Sup3rS3cretPw" not in wire
    assert resp["error"]["code"] == -32010

    _code, stderr = p.close()
    assert "Sup3rS3cretPw" not in stderr


def test_secret_in_custom_event_never_reaches_client(peer):
    """custom 事件里的 error/reason 会变成客户端可见文本，必须先脱敏。"""
    p = peer(
        {
            "events": [
                ["custom", {"type": "llm_retry", "attempt": 1, "max_attempts": 3,
                            "reason": f"401，Authorization: Bearer {CONTRACT_FAKE_KEY}"}],
                ["end", {"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}],
            ]
        }
    )
    p.initialize()
    session_id = p.new_session()

    rid = p.send("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "你好"}]})
    resp = p.await_response(rid)

    wire = json.dumps(resp, ensure_ascii=False)
    assert CONTRACT_FAKE_KEY not in wire
    # 重试次数仍要看得见
    assert "1/3" in wire or ("1" in wire and "3" in wire)

    _code, stderr = p.close()
    assert CONTRACT_FAKE_KEY not in stderr
