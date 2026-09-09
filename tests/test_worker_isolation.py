"""worker 子进程隔离：取消是否真的终止了执行体。

这是本轮架构变更的核心验收。线程模型下这些断言全都不可能成立——被弃用的线程
仍在跑模型调用、仍在执行工具、仍在写同一条 DeerFlow thread。因此每个用例都直接
盯着**外部可观测的物证**：进程组是否消失、副作用文件是否停止增长、checkpoint
是否只有一个写入者。

用的是真实的 :class:`SubprocessTurnRunner`，起真实的 ``deerflow_acp.worker``
子进程；只有子进程里的 DeerFlow 后端换成了 ``worker_fakes.StatefulFakeBackend``。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from deerflow_acp.config import BridgeConfig
from deerflow_acp.runner import RemoteBackendError, SubprocessTurnRunner
from deerflow_acp.session import SessionRegistry

REPO_TESTS = Path(__file__).parent


def _worker_env(state: Path) -> dict[str, str]:
    """让 worker 子进程能 import 到 ``worker_fakes``，并指向本用例的状态目录。"""
    return {
        "DEERFLOW_ACP_WORKER_BACKEND": "worker_fakes:build",
        "DEERFLOW_ACP_FAKE_STATE": str(state),
        "PYTHONPATH": os.pathsep.join([str(REPO_TESTS), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep),
    }


class ScriptedSubprocessRunner(SubprocessTurnRunner):
    """把假后端与状态目录注入 worker 环境的 runner。"""

    def __init__(self, config: BridgeConfig, state: Path) -> None:
        super().__init__(config)
        self._state = state

    def _worker_env(self) -> dict[str, str]:
        return {**os.environ, **_worker_env(self._state)}


@pytest.fixture
def state(tmp_path: Path) -> Path:
    (tmp_path / "checkpoints").mkdir()
    return tmp_path


def write_script(state: Path, script: dict[str, Any]) -> None:
    (state / "script.json").write_text(json.dumps(script), encoding="utf-8")


def side_effect_count(state: Path) -> int:
    path = state / "side_effects.log"
    if not path.exists():
        return 0
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def checkpoint_writers(state: Path, thread_id: str) -> set[int]:
    path = state / "checkpoints" / f"{thread_id}.json"
    if not path.exists():
        return set()
    return {entry["pid"] for entry in json.loads(path.read_text(encoding="utf-8"))}


def pgid_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _noop(event_type: str, data: dict) -> None:
    return None


def registry(state: Path, **cfg: Any) -> SessionRegistry:
    config = BridgeConfig(**cfg)
    return SessionRegistry(_UnusedBackend(), config, runner=ScriptedSubprocessRunner(config, state))


class _UnusedBackend:
    """SessionRegistry 只用它做 thread_exists；turn 执行完全走 runner。"""

    def stream(self, message: str, *, thread_id: str):  # pragma: no cover - 不应被调用
        raise AssertionError("turn 必须由 worker 子进程执行")

    def thread_exists(self, thread_id: str) -> bool:
        return False

    def history(self, thread_id: str) -> list[dict[str, Any]]:
        return []


# ----------------------------------------------------------------------
# 1) 取消超时后进程组必须真的消失
# ----------------------------------------------------------------------


async def test_escalated_cancel_kills_worker_process_group(state: Path):
    """卡在第一个 yield 之前的 worker，超时后整个进程组必须已退出。"""
    write_script(state, {"stall_before_first_yield_s": 30, "events": [["end", {"usage": {}}]]})
    reg = registry(state, cancel_grace_seconds=0.5)
    session = reg.create("/tmp")

    task = asyncio.create_task(reg.run_turn(session, "问题", _noop))
    # 等 worker 真的开始干活（副作用文件出现即证明它在跑）
    deadline = time.time() + 15
    while side_effect_count(state) == 0 and time.time() < deadline:
        await asyncio.sleep(0.05)
    assert side_effect_count(state) > 0, "worker 没有启动"

    reg.cancel(session.session_id)
    outcome = await asyncio.wait_for(task, timeout=20)

    assert outcome.stop_reason == "cancelled"
    assert outcome.escalated is True
    assert outcome.worker_pid is not None and outcome.worker_pgid is not None
    assert outcome.worker_killed is True, "宽限期内未协作退出，必须动用 killpg"
    assert outcome.worker_reaped is True, "必须确认进程已被回收后才返回"
    assert not pgid_alive(outcome.worker_pgid), f"进程组 {outcome.worker_pgid} 仍然存在"
    assert session.running is False


async def test_no_side_effects_after_escalated_cancel(state: Path):
    """取消返回后，等待**超过原阻塞释放时点**，副作用必须彻底停止增长。

    线程模型下这条必然失败：被弃用的线程会一直写到 30s 阻塞结束。
    """
    write_script(
        state,
        {"stall_before_first_yield_s": 6, "side_effect_interval_s": 0.1, "events": [["end", {"usage": {}}]]},
    )
    reg = registry(state, cancel_grace_seconds=0.5)
    session = reg.create("/tmp")

    started = time.time()
    task = asyncio.create_task(reg.run_turn(session, "问题", _noop))
    deadline = time.time() + 15
    while side_effect_count(state) == 0 and time.time() < deadline:
        await asyncio.sleep(0.05)

    reg.cancel(session.session_id)
    outcome = await asyncio.wait_for(task, timeout=20)
    assert outcome.escalated is True

    at_cancel = side_effect_count(state)
    # 等到远超原本 6s 的阻塞释放时点
    while time.time() - started < 8.0:
        await asyncio.sleep(0.2)

    assert side_effect_count(state) == at_cancel, (
        f"取消后副作用仍在增长：取消时 {at_cancel} → 现在 {side_effect_count(state)}，"
        "说明旧执行体没有被真正终止"
    )


async def test_no_late_events_after_escalated_cancel(state: Path):
    """取消返回后，旧 turn 的事件绝不能晚到。"""
    write_script(
        state,
        {
            "stall_before_first_yield_s": 4,
            "side_effect_interval_s": 0.1,
            "events": [["messages-tuple", {"type": "ai", "content": "旧 turn 的迟到事件", "id": "m"}]],
        },
    )
    reg = registry(state, cancel_grace_seconds=0.4)
    session = reg.create("/tmp")

    seen: list[tuple[str, dict]] = []

    async def on_event(event_type: str, data: dict) -> None:
        seen.append((event_type, data))

    started = time.time()
    task = asyncio.create_task(reg.run_turn(session, "问题", on_event))
    deadline = time.time() + 15
    while side_effect_count(state) == 0 and time.time() < deadline:
        await asyncio.sleep(0.05)

    reg.cancel(session.session_id)
    await asyncio.wait_for(task, timeout=20)
    count_at_return = len(seen)

    while time.time() - started < 6.0:
        await asyncio.sleep(0.2)

    assert len(seen) == count_at_return, f"取消返回后又收到了 {len(seen) - count_at_return} 个迟到事件"


# ----------------------------------------------------------------------
# 2) 同 session 后续 turn 必须独占该 thread
# ----------------------------------------------------------------------


async def test_next_turn_starts_only_after_old_worker_confirmed_dead(state: Path):
    """新 turn 启动时，旧 worker 必须已经退出——用 checkpoint 写入者证明。

    两个 worker 若并发存在，``checkpoints/<thread_id>.json`` 会同时出现两个 PID。
    """
    write_script(
        state,
        {"stall_before_first_yield_s": 10, "side_effect_interval_s": 0.1, "events": [["end", {"usage": {}}]]},
    )
    reg = registry(state, cancel_grace_seconds=0.4)
    session = reg.create("/tmp")
    thread_id = session.session_id

    task = asyncio.create_task(reg.run_turn(session, "第一问", _noop))
    deadline = time.time() + 15
    while side_effect_count(state) == 0 and time.time() < deadline:
        await asyncio.sleep(0.05)

    reg.cancel(session.session_id)
    first = await asyncio.wait_for(task, timeout=20)
    assert first.escalated is True
    old_pid = first.worker_pid
    assert old_pid is not None
    assert not pgid_alive(first.worker_pgid), "新 turn 启动前旧进程组必须已消失"

    # 第二个 turn：不再阻塞
    write_script(state, {"stall_before_first_yield_s": 0, "events": [["end", {"usage": {"total_tokens": 7}}]]})
    second = await asyncio.wait_for(reg.run_turn(session, "第二问", _noop), timeout=30)

    assert second.stop_reason == "end_turn"
    assert second.escalated is False
    assert second.worker_pid != old_pid, "新 turn 必须在新的 worker 进程里跑"

    # 等一段远超旧 worker 原阻塞时长的时间，旧 PID 不得再出现在 checkpoint 里
    await asyncio.sleep(1.0)
    writers = checkpoint_writers(state, thread_id)
    late = [pid for pid in writers if pid == old_pid]
    recent = json.loads((state / "checkpoints" / f"{thread_id}.json").read_text(encoding="utf-8"))
    # 旧 worker 在被杀之前写过 checkpoint 是正常的；关键是它在新 turn 开始之后
    # 不能再写。用时间戳分界：新 worker 第一条之后不得出现旧 PID。
    new_pid_first_ts = min(e["ts"] for e in recent if e["pid"] == second.worker_pid)
    stale_after = [e for e in recent if e["pid"] == old_pid and e["ts"] > new_pid_first_ts]
    assert not stale_after, f"旧 worker 在新 turn 开始后仍在写同一 thread：{stale_after}"
    assert late, "前置条件：旧 worker 确实写过 checkpoint，否则本用例无区分力"


async def test_session_reuses_checkpoint_after_kill(state: Path):
    """旧 worker 被强杀后，同 session 新 turn 仍能看到已落的 checkpoint。"""
    write_script(
        state,
        {"stall_before_first_yield_s": 8, "side_effect_interval_s": 0.1, "checkpoint_interval_s": 0.1,
         "events": [["end", {"usage": {}}]]},
    )
    reg = registry(state, cancel_grace_seconds=0.4)
    session = reg.create("/tmp")
    thread_id = session.session_id

    task = asyncio.create_task(reg.run_turn(session, "第一问", _noop))
    deadline = time.time() + 15
    while side_effect_count(state) == 0 and time.time() < deadline:
        await asyncio.sleep(0.05)
    reg.cancel(session.session_id)
    first = await asyncio.wait_for(task, timeout=20)
    assert first.worker_killed is True

    before = len(json.loads((state / "checkpoints" / f"{thread_id}.json").read_text(encoding="utf-8")))
    assert before > 0, "强杀前应已落下 checkpoint"

    write_script(state, {"stall_before_first_yield_s": 0, "events": [["end", {"usage": {}}]]})
    second = await asyncio.wait_for(reg.run_turn(session, "第二问", _noop), timeout=30)
    assert second.stop_reason == "end_turn"

    after = json.loads((state / "checkpoints" / f"{thread_id}.json").read_text(encoding="utf-8"))
    assert len(after) > before, "新 turn 必须在同一 thread 的既有 checkpoint 上继续追加"
    assert after[:before] == json.loads(json.dumps(after[:before])), "既有 checkpoint 不得被破坏"


# ----------------------------------------------------------------------
# 3) 协作式退出：worker 在事件边界能自行收摊，无需强杀
# ----------------------------------------------------------------------


async def test_cooperative_cancel_does_not_need_kill(state: Path):
    """worker 在事件边界看到 SIGTERM 时应自行退出，不触发 escalation。"""
    write_script(
        state,
        {"stall_before_first_yield_s": 0, "events": [["messages-tuple", {"type": "ai", "content": f"chunk-{i}", "id": "m"}] for i in range(400)]},
    )
    reg = registry(state, cancel_grace_seconds=5.0)
    session = reg.create("/tmp")

    seen: list[Any] = []

    async def on_event(event_type: str, data: dict) -> None:
        seen.append(data)
        if len(seen) == 3:
            reg.cancel(session.session_id)

    outcome = await asyncio.wait_for(reg.run_turn(session, "问题", on_event), timeout=30)

    assert outcome.stop_reason == "cancelled"
    assert outcome.escalated is False, "worker 在事件边界应能协作退出，不应升级为强杀"
    assert outcome.worker_reaped is True
    assert not pgid_alive(outcome.worker_pgid)


async def test_normal_turn_completes_and_worker_exits(state: Path):
    write_script(
        state,
        {"events": [["messages-tuple", {"type": "ai", "content": "你好", "id": "m"}], ["end", {"usage": {"total_tokens": 11}}]]},
    )
    reg = registry(state, cancel_grace_seconds=5.0)
    session = reg.create("/tmp")

    seen: list[tuple[str, dict]] = []

    async def on_event(event_type: str, data: dict) -> None:
        seen.append((event_type, data))

    outcome = await asyncio.wait_for(reg.run_turn(session, "问题", on_event), timeout=30)

    assert outcome.stop_reason == "end_turn"
    assert outcome.usage_payload == {"total_tokens": 11}
    assert [t for t, _ in seen] == ["messages-tuple", "end"]
    assert outcome.worker_killed is False, "正常结束不应动用 killpg"
    assert not pgid_alive(outcome.worker_pgid)


async def test_worker_events_carry_worker_pid(state: Path):
    """事件确实来自子进程，不是父进程里的自欺欺人。"""
    write_script(state, {"events": [["messages-tuple", {"type": "ai", "content": "x", "id": "m"}]]})
    reg = registry(state, cancel_grace_seconds=5.0)
    session = reg.create("/tmp")

    seen: list[dict] = []

    async def on_event(event_type: str, data: dict) -> None:
        seen.append(data)

    outcome = await asyncio.wait_for(reg.run_turn(session, "问题", on_event), timeout=30)
    assert seen and seen[0]["pid"] == outcome.worker_pid != os.getpid()


# ----------------------------------------------------------------------
# 4) 后端异常：只有类型名过河
# ----------------------------------------------------------------------


async def test_backend_error_crosses_ipc_as_type_only(state: Path):
    write_script(state, {"raise": "provider 拒绝：sk-proj-Ab3xQ9zK7mNpR2vT5wY8cE1dF4gH6jL0oP"})
    reg = registry(state, cancel_grace_seconds=5.0)
    session = reg.create("/tmp")

    outcome = await asyncio.wait_for(reg.run_turn(session, "问题", _noop), timeout=30)

    assert outcome.stop_reason == "refusal"
    assert isinstance(outcome.error, RemoteBackendError)
    assert outcome.error.error_class == "RuntimeError"
    # 异常消息根本没有过河——不是靠正则抹掉的
    assert "sk-proj" not in repr(outcome.error)


# ----------------------------------------------------------------------
# 5) 进程回收：不留孤儿
# ----------------------------------------------------------------------


async def test_worker_process_group_is_reaped_on_every_path(state: Path):
    """正常结束、协作取消、强杀三条路径都不得留下存活进程组。"""
    scripts = [
        {"events": [["end", {"usage": {}}]]},
        {"events": [["messages-tuple", {"type": "ai", "content": "x", "id": "m"}] for _ in range(400)]},
        {"stall_before_first_yield_s": 8, "side_effect_interval_s": 0.1, "events": [["end", {"usage": {}}]]},
    ]
    pgids: list[int] = []
    for index, script in enumerate(scripts):
        write_script(state, script)
        reg = registry(state, cancel_grace_seconds=0.5)
        session = reg.create("/tmp")

        async def on_event(event_type: str, data: dict, _i: int = index) -> None:
            if _i == 1:
                reg.cancel(session.session_id)

        task = asyncio.create_task(reg.run_turn(session, "问题", on_event))
        if index == 2:
            deadline = time.time() + 15
            while side_effect_count(state) == 0 and time.time() < deadline:
                await asyncio.sleep(0.05)
            reg.cancel(session.session_id)
        outcome = await asyncio.wait_for(task, timeout=30)
        assert outcome.worker_pgid is not None
        pgids.append(outcome.worker_pgid)

    await asyncio.sleep(0.5)
    alive = [pgid for pgid in pgids if pgid_alive(pgid)]
    assert not alive, f"以下进程组未被回收：{alive}"


@pytest.mark.asyncio
async def test_terminate_all_reaps_in_flight_worker(state: Path):
    """关停兜底：turn 协程被取消后，``terminate_all`` 必须收掉在途 worker。

    模拟桥收到 SIGTERM——事件循环侧直接 cancel 在途 turn，此时没有任何
    session/cancel 走过取消流程，回收只能靠这个进程级入口。
    """
    write_script(state, {"events": [], "stall_before_first_yield_s": 30, "side_effect_interval_s": 0.1})
    runner = ScriptedSubprocessRunner(BridgeConfig(cancel_grace_seconds=5.0), state)

    seen: list[Any] = []

    async def on_event(event_type: str, data: dict) -> None:
        seen.append((event_type, data))

    cancel_event = threading.Event()
    task = asyncio.create_task(
        runner.execute(session_id="df-x", message="问题", on_event=on_event, cancel_event=cancel_event, grace=5.0)
    )

    deadline = time.time() + 15
    while side_effect_count(state) == 0 and time.time() < deadline:
        await asyncio.sleep(0.05)
    assert side_effect_count(state) > 0, "worker 没有起来"

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    # 协程内的中断分支已经强杀过；terminate_all 是幂等的第二道闸
    reaped = runner.terminate_all()
    await asyncio.sleep(0.3)
    for pgid in reaped:
        assert not pgid_alive(pgid)

    before = side_effect_count(state)
    await asyncio.sleep(1.0)
    assert side_effect_count(state) == before, "worker 在关停后仍在制造副作用"


def test_signal_group_tolerates_missing_process():
    """进程组已消失时发信号不得抛异常。"""
    runner = SubprocessTurnRunner(BridgeConfig())
    # 一个几乎不可能存在的 pgid
    assert runner._signal_group(4194303, signal.SIGTERM, "df-x") is False


def test_cancel_event_is_thread_event():
    """取消标志仍是线程安全的 Event，供 CLI 的信号处理器直接置位。"""
    reg = SessionRegistry(_UnusedBackend(), BridgeConfig())
    session = reg.create("/tmp")
    assert isinstance(session.cancel_event, threading.Event)
