"""worker 子进程本身的行为：IPC 通道隔离、信号语义、异常分类。

这些用例直接起 ``python -m deerflow_acp.worker``，不经过 runner——验证的是
worker 单独作为一个进程时的契约，任何 runner 实现都依赖它。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from deerflow_acp import ipc

REPO_TESTS = Path(__file__).parent


def spawn_worker(state: Path, thread_id: str, message: str = "问题") -> subprocess.Popen:
    env = {
        **os.environ,
        "DEERFLOW_ACP_WORKER_BACKEND": "worker_fakes:build",
        "DEERFLOW_ACP_FAKE_STATE": str(state),
        "PYTHONPATH": os.pathsep.join([str(REPO_TESTS), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep),
        "PYTHONUNBUFFERED": "1",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "deerflow_acp.worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=True,
    )
    job = {"session_id": thread_id, "message": message, "thread_id": thread_id, "config": {}}
    assert proc.stdin is not None
    proc.stdin.write((json.dumps(job) + "\n").encode())
    proc.stdin.flush()
    proc.stdin.close()
    return proc


def read_messages(proc: subprocess.Popen, *, limit: int = 50, timeout: float = 30) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    deadline = time.time() + timeout
    assert proc.stdout is not None
    while len(out) < limit and time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        parsed = ipc.decode(line)
        if parsed is not None:
            out.append(parsed)
    return out


@pytest.fixture
def state(tmp_path: Path) -> Path:
    (tmp_path / "checkpoints").mkdir()
    return tmp_path


def write_script(state: Path, script: dict[str, Any]) -> None:
    (state / "script.json").write_text(json.dumps(script), encoding="utf-8")


def test_worker_emits_ready_then_events_then_done(state: Path):
    write_script(
        state,
        {"events": [["messages-tuple", {"type": "ai", "content": "你好", "id": "m"}], ["end", {"usage": {"total_tokens": 5}}]]},
    )
    proc = spawn_worker(state, "df-a")
    try:
        messages = read_messages(proc)
    finally:
        proc.wait(timeout=30)

    kinds = [m["t"] for m in messages]
    assert kinds == [ipc.MSG_READY, ipc.MSG_EVENT, ipc.MSG_EVENT, ipc.MSG_DONE]
    assert messages[0]["pid"] == proc.pid
    assert messages[-1]["cancelled"] is False
    assert proc.returncode == 0


def test_worker_ipc_channel_is_immune_to_stdout_pollution(state: Path):
    """后端往 stdout 打印的内容必须落到 stderr，不能撕裂 ndJSON 流。"""
    write_script(
        state,
        {
            "events": [["custom", {"type": "task_started", "task_id": "t1", "name": "搜索"}], ["end", {"usage": {}}]],
            "pollute_stdout": True,
        },
    )
    proc = spawn_worker(state, "df-b")
    try:
        assert proc.stdout is not None
        raw = proc.stdout.read().decode("utf-8", errors="replace")
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    finally:
        proc.wait(timeout=30)

    # 逐行校验：IPC 通道上不允许出现任何非消息行
    lines = [line for line in raw.splitlines() if line.strip()]
    parsed = [ipc.decode(line) for line in lines]
    assert all(p is not None for p in parsed), f"IPC 通道被污染：{lines}"
    assert [p["t"] for p in parsed] == [ipc.MSG_READY, ipc.MSG_EVENT, ipc.MSG_EVENT, ipc.MSG_DONE]

    # 垃圾必须原样出现在 stderr——证明它确实被写出去了，只是改道了
    assert "这行垃圾绝不能出现在 IPC 通道里" in stderr
    assert "裸 write 也不行" in stderr
    assert proc.returncode == 0


def test_worker_reports_backend_error_as_type_only(state: Path):
    secret = "sk-proj-Ab3xQ9zK7mNpR2vT5wY8cE1dF4gH6jL0oP"
    write_script(state, {"raise": f"provider 拒绝：{secret}"})
    proc = spawn_worker(state, "df-c")
    try:
        messages = read_messages(proc)
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    finally:
        proc.wait(timeout=30)

    errors = [m for m in messages if m["t"] == ipc.MSG_ERROR]
    assert errors == [{"t": ipc.MSG_ERROR, "cls": "RuntimeError"}]
    assert secret not in json.dumps(messages), "异常消息不得跨进程传输"
    assert secret not in stderr, "worker 的 stderr 也不得泄露秘密"
    assert "RuntimeError" in stderr, "必须保留可诊断的类型名"
    assert proc.returncode == 1


def test_worker_exits_cooperatively_on_sigterm(state: Path):
    """SIGTERM 到达时，worker 应在下一个事件边界收摊并正常退出。"""
    write_script(
        state,
        {"events": [["messages-tuple", {"type": "ai", "content": f"c{i}", "id": "m"}] for i in range(3000)]},
    )
    proc = spawn_worker(state, "df-d")
    try:
        assert proc.stdout is not None
        # 等它确实开始产出事件
        seen = 0
        while seen < 5:
            line = proc.stdout.readline()
            assert line, "worker 提前退出"
            if (ipc.decode(line) or {}).get("t") == ipc.MSG_EVENT:
                seen += 1

        proc.send_signal(signal.SIGTERM)
        returncode = proc.wait(timeout=20)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    assert returncode == 0, "协作退出应是正常退出码"


def test_worker_exits_on_stdin_eof_without_job(state: Path):
    """没有 job 就 EOF：worker 必须立刻退出，不得挂住。"""
    env = {
        **os.environ,
        "DEERFLOW_ACP_WORKER_BACKEND": "worker_fakes:build",
        "DEERFLOW_ACP_FAKE_STATE": str(state),
        "PYTHONPATH": os.pathsep.join([str(REPO_TESTS), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep),
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "deerflow_acp.worker"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=True,
    )
    assert proc.wait(timeout=30) == 0


def test_worker_close_error_does_not_leak_secret_to_stderr(state: Path):
    """``close()`` 阶段抛带秘密的异常时，worker 的 stderr 只留类型名。"""
    secret = "sk-proj-Ab3xQ9zK7mNpR2vT5wY8cE1dF4gH6jL0oP"
    write_script(
        state,
        {
            "events": [["messages-tuple", {"type": "ai", "content": f"c{i}", "id": "m"}] for i in range(3000)],
            "raise_on_close": f"清理连接失败：{secret}",
        },
    )
    proc = spawn_worker(state, "df-e")
    try:
        assert proc.stdout is not None
        seen = 0
        while seen < 5:
            line = proc.stdout.readline()
            assert line, "worker 提前退出"
            if (ipc.decode(line) or {}).get("t") == ipc.MSG_EVENT:
                seen += 1
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=20)
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    assert secret not in stderr, "close() 异常把秘密写进了 worker 的 stderr"
    assert "sk-proj" not in stderr
    assert "关闭 DeerFlow 生成器时出错" in stderr, "该异常必须被记录，不能静默吞掉"
    assert "RuntimeError" in stderr, "必须保留异常类型"
    assert "Traceback" not in stderr, "不得输出 traceback"
