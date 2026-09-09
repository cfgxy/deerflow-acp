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
import copy
import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from deerflow_acp.config import BridgeConfig
from deerflow_acp.runner import RemoteBackendError, SubprocessTurnRunner
from deerflow_acp.session import SessionQuarantinedError, SessionRegistry, TurnAlreadyRunningError

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
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        # worker 正好在 write_text 中途被强杀，文件被截断。这是「强制终止真的
        # 发生了」的副产品，不是被测行为的缺陷；按「没有可辨认的写入者」处理。
        return set()
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        return set()
    return {entry["pid"] for entry in entries}


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

    cp_path = state / "checkpoints" / f"{thread_id}.json"
    existing = json.loads(cp_path.read_text(encoding="utf-8"))
    assert existing, "强杀前应已落下 checkpoint"
    # 快照真实内容，而不是把切片和它自己的 JSON round-trip 比——后者对任何输入都成立，
    # 无法证明「既有条目未被改写」。
    snapshot = copy.deepcopy(existing)

    write_script(state, {"stall_before_first_yield_s": 0, "events": [["end", {"usage": {}}]]})
    second = await asyncio.wait_for(reg.run_turn(session, "第二问", _noop), timeout=30)
    assert second.stop_reason == "end_turn"

    after = json.loads(cp_path.read_text(encoding="utf-8"))
    assert len(after) > len(snapshot), "新 turn 必须在同一 thread 的既有 checkpoint 上继续追加"
    assert after[: len(snapshot)] == snapshot, "既有 checkpoint 前缀被改写了"
    assert all(entry["pid"] == second.worker_pid for entry in after[len(snapshot) :]), (
        "新追加的 checkpoint 必须全部来自新 worker"
    )


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
    """关停兜底：``terminate_all`` 必须能独立收掉在途 worker。

    模拟桥收到 SIGTERM——事件循环侧直接 cancel 在途 turn，此时没有任何
    session/cancel 走过取消流程，回收只能靠这个进程级入口。

    **本用例刻意屏蔽协程内的中断强杀分支**（`_signal_group` 打桩成 no-op）：
    否则 worker 早就被那一路收掉了，`terminate_all` 即使完全失效也照样通过，
    用例就失去了对兜底闸的区分力。
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

    real_signal_group = SubprocessTurnRunner._signal_group
    with mock.patch.object(SubprocessTurnRunner, "_signal_group", staticmethod(lambda *a, **k: True)):
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        # 前置条件：中断分支被屏蔽后 worker 确实还活着，兜底闸才有可测对象
        pgid = next(iter(runner._live_pgids), None)
        assert pgid is not None, "在途 pgid 必须仍被 runner 持有，否则兜底闸无从生效"
        assert real_signal_group(pgid, 0, "df-x") is True, "worker 应仍存活，否则本用例无区分力"

    reaped = runner.terminate_all()
    assert pgid in reaped, "terminate_all 必须处理仍在途的进程组"
    await asyncio.sleep(0.3)
    assert not pgid_alive(pgid)

    before = side_effect_count(state)
    await asyncio.sleep(1.0)
    assert side_effect_count(state) == before, "worker 在关停后仍在制造副作用"


# ----------------------------------------------------------------------
# 6) 阻断项返工：任何退出路径都必须确认**整个进程组**终结后才释放 session
# ----------------------------------------------------------------------


async def _start_stalled_turn(reg: SessionRegistry, session: Any, state: Path) -> asyncio.Task:
    """起一个卡在第一个 yield 之前的 turn，并等到它真的在跑。"""
    task = asyncio.create_task(reg.run_turn(session, "问题", _noop))
    deadline = time.time() + 15
    while side_effect_count(state) == 0 and time.time() < deadline:
        await asyncio.sleep(0.05)
    assert side_effect_count(state) > 0, "worker 没有起来"
    return task


async def test_close_during_turn_does_not_allow_resume_before_worker_dies(state: Path):
    """`session/close` 在 turn 运行中不得立即摘除注册项。

    反例形态：close 一进来就 pop，客户端随即 `session/resume` 拉回同一个 thread_id
    并发新 prompt——旧 worker 还在宽限期里，两个 worker 会并发写同一条 checkpoint。
    这里断言 resume 在旧执行体终结前拿不到一个可用的新会话。
    """
    write_script(
        state,
        {"stall_before_first_yield_s": 20, "side_effect_interval_s": 0.1, "events": [["end", {"usage": {}}]]},
    )
    reg = registry(state, cancel_grace_seconds=3.0)
    session = reg.create("/tmp")
    thread_id = session.session_id

    task = await _start_stalled_turn(reg, session, state)

    # close 到达：只置取消标志 + 记下关闭请求，注册项必须还在
    reg.close(thread_id)
    assert reg._sessions.get(thread_id) is session, "turn 在跑时 close 不得摘除注册项"
    assert session.running is True

    # 客户端立刻 resume 回来：拿到的必须是同一个仍在运行的 session，
    # 因此新 prompt 只能得到 TurnAlreadyRunningError，不可能起第二个 worker。
    resumed = reg.resume(thread_id, "/tmp")
    assert resumed is session
    with pytest.raises(TurnAlreadyRunningError):
        await reg.run_turn(resumed, "抢跑的第二问", _noop)

    outcome = await asyncio.wait_for(task, timeout=30)
    assert outcome.worker_pgid is not None
    assert not pgid_alive(outcome.worker_pgid)
    # 旧执行体确认终结后，延后的 close 才真正生效
    assert thread_id not in reg._sessions, "在途 turn 结束后 close 请求必须生效"
    # 全程只有一个 worker 写过这条 thread
    assert checkpoint_writers(state, thread_id) <= {outcome.worker_pid}


async def test_session_not_released_until_tool_child_in_group_exits(state: Path):
    """worker 主进程先退出、组内工具子进程仍存活时，session 不得被释放。

    反例形态：把「主 PID 退出」当成「进程组退出」。这里的工具子进程留在同一
    PGID 里但不是桥的子进程，``proc.wait()`` 根本看不到它；只有 ``killpg(pgid, 0)``
    才问得出真相。它持续写 ``tool_child.log``，因此提前释放会留下物证。
    """
    write_script(
        state,
        {
            "events": [["end", {"usage": {}}]],
            "tool_child_lifetime_s": 3.0,
            "side_effect_interval_s": 0.1,
        },
    )
    reg = registry(state, cancel_grace_seconds=5.0)
    session = reg.create("/tmp")

    outcome = await asyncio.wait_for(reg.run_turn(session, "问题", _noop), timeout=40)

    assert outcome.stop_reason == "end_turn"
    child_pid = int((state / "tool_child.pid").read_text(encoding="utf-8"))
    # 前置条件：工具子进程确实被派生出来过，否则本用例无区分力
    assert child_pid > 0
    # run_turn 返回时整个进程组必须已排空——包含那个孙进程
    assert outcome.worker_pgid is not None
    assert not pgid_alive(outcome.worker_pgid), "run_turn 返回时进程组内仍有存活进程"
    assert not _pid_alive(child_pid), f"工具子进程 {child_pid} 在 session 释放后仍存活"

    # 且它不再写日志
    before = _tool_child_lines(state)
    await asyncio.sleep(1.0)
    assert _tool_child_lines(state) == before, "组内工具子进程在 session 释放后仍在产生副作用"


async def test_session_quarantined_when_group_cannot_be_confirmed_dead(state: Path):
    """回收无法确认时必须隔离 session，禁止后续 prompt 与 resume。

    反例形态：`worker_reaped=False` 却照样释放 `_running`，下一个 turn 就与可能
    仍在写 checkpoint 的旧执行体并发。这里把「确认排空」打桩成永远失败，断言
    桥选择隔离而不是放行。
    """
    write_script(
        state,
        {"stall_before_first_yield_s": 20, "side_effect_interval_s": 0.1, "events": [["end", {"usage": {}}]]},
    )
    reg = registry(state, cancel_grace_seconds=0.4)
    session = reg.create("/tmp")
    thread_id = session.session_id

    task = await _start_stalled_turn(reg, session, state)

    async def never_drained(self, proc, pgid, timeout=None):
        return False

    with mock.patch.object(SubprocessTurnRunner, "_await_group_gone", never_drained):
        reg.cancel(thread_id)
        outcome = await asyncio.wait_for(task, timeout=30)

    assert outcome.worker_reaped is False, "前置条件：本轮必须是「未确认回收」"
    assert session.quarantined is True
    assert thread_id not in reg._sessions, "被隔离的会话必须从注册表摘除"

    # 三条入口全部拒绝：run_turn、get、resume
    with pytest.raises(SessionQuarantinedError):
        await reg.run_turn(session, "第二问", _noop)
    with pytest.raises(SessionQuarantinedError):
        reg.get(thread_id)
    with pytest.raises(SessionQuarantinedError):
        reg.resume(thread_id, "/tmp")

    # 隔离期间 pgid 仍归 runner 持有，关停兜底还能收它
    assert outcome.worker_pgid in reg._runner._live_pgids
    reg.terminate_all_workers()
    await asyncio.sleep(0.3)
    assert not pgid_alive(outcome.worker_pgid)


async def test_event_dispatch_failure_confirms_group_and_keeps_tracking(state: Path):
    """`on_event` 抛异常时，必须确认进程组终结，且不丢失回收跟踪。

    反例形态：只同步 killpg 就从 `_live_pgids` 删掉且不确认退出——session 被释放、
    关停兜底也失去这个 PGID，组内进程无人负责。
    """
    write_script(
        state,
        {
            "events": [["messages-tuple", {"type": "ai", "content": "x", "id": "m"}] for _ in range(50)],
            "side_effect_interval_s": 0.1,
        },
    )
    reg = registry(state, cancel_grace_seconds=5.0)
    session = reg.create("/tmp")
    thread_id = session.session_id

    class Boom(RuntimeError):
        pass

    async def exploding_on_event(event_type: str, data: dict) -> None:
        raise Boom("下发失败")

    # 记下本轮的 pgid：异常路径没有 TurnOutcome，只能从信号动作里抓。
    seen_pgids: list[int] = []
    real_signal_group = SubprocessTurnRunner._signal_group

    def spy(pgid: int, sig: int, session_id: str) -> bool:
        if sig != 0:
            seen_pgids.append(pgid)
        return real_signal_group(pgid, sig, session_id)

    with mock.patch.object(SubprocessTurnRunner, "_signal_group", staticmethod(spy)):
        with pytest.raises(Boom):
            await reg.run_turn(session, "问题", exploding_on_event)

    assert seen_pgids, "中断分支必须对进程组发过信号"
    pgid = seen_pgids[0]

    assert session.running is False
    # 关键区分点：只发 SIGKILL 不等于组已排空。不 ``proc.wait()`` 的反例实现会把
    # worker 主进程留成僵尸，而僵尸仍属于该进程组，``killpg(pgid, 0)`` 照样成功——
    # 也就是说 session 已被释放，组却还没消失。正确实现必须先回收主进程、再轮询到
    # 整组消失，才允许返回。
    assert not pgid_alive(pgid), "run_turn 抛出时进程组必须已确认排空（含回收主进程）"
    # 事件下发失败不是「未确认回收」——事件循环还活着，必须当场确认排空，
    # 因此 session 不该被隔离，而是干净结束。
    assert session.quarantined is False, "能确认排空时不应误隔离"
    assert not reg._runner._live_pgids, "确认排空后应从在途集合摘除"
    pgids = reg.terminate_all_workers()
    assert pgids == [], "已确认排空的进程组不应再残留在兜底集合里"
    # 无残余进程继续写这条 thread
    writers = checkpoint_writers(state, thread_id)
    assert len(writers) <= 1


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _tool_child_lines(state: Path) -> int:
    path = state / "tool_child.log"
    if not path.exists():
        return 0
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


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
