"""ACP 会话注册与 turn 生命周期控制。

映射关系：一个 ACP ``sessionId`` 一对一绑定一个 DeerFlow ``thread_id``，
两者取同一个字符串值——DeerFlow 的 checkpointer 以 thread_id 为主键，
共用同一标识才能让「跨进程恢复」退化成「用同一个 thread_id 继续 stream」，
桥自身不需要维护任何持久化映射表。

**turn 的执行被放在独立进程组的 worker 子进程里**（见 :mod:`deerflow_acp.runner`）。
这不是为了并行，而是为了取消能真正生效：DeerFlow 的模型调用与工具执行都发生在
``next(generator)`` 内部，线程模型下无法强制打断——取消超时后那个线程会继续跑完
模型调用、继续执行工具、继续往同一个 ``thread_id`` 写 checkpoint，而桥此时已经
释放了 ``_running``，下一个 turn 就会与它并发写同一条 DeerFlow thread。

因此取消的处理顺序是：置标志 → 向 worker 进程组发 ``SIGTERM`` 请求协作退出 →
超过 ``cancel_grace_seconds`` 仍未退出则 ``killpg(SIGKILL)`` →
**确认进程组已被回收之后**才释放 ``_running``。这条「确认退出」是同 session
串行语义的支点：``_running`` 一旦释放，新 turn 立刻可以在同一 thread 上启动。
"""

from __future__ import annotations

import asyncio
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from .backend import BackendUnavailableError, DeerFlowBackend
from .config import BridgeConfig
from .logging_setup import get_logger
from .runner import InProcessTurnRunner, RunResult, SubprocessTurnRunner, TurnRunner

logger = get_logger("session")

# 与 DeerFlow ``deerflow.utils.thread_id.validate_thread_id`` 完全一致的约束：
# 1-64 位 ASCII 字母、数字、连字符或下划线。ACP sessionId 直接作为 thread_id
# 使用，因此必须在桥这一层就拒掉不合法的值，而不是让 DeerFlow 抛 ValueError。
_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class SessionIdError(ValueError):
    """sessionId 不满足 DeerFlow thread_id 约束。"""


class UnknownSessionError(LookupError):
    """请求的 sessionId 在本进程与 DeerFlow checkpoint 中都不存在。"""


class TurnAlreadyRunningError(RuntimeError):
    """同一会话已有 turn 在执行。"""


def new_session_id() -> str:
    """生成满足 DeerFlow thread_id 约束的会话 ID。"""
    return f"df-{uuid.uuid4().hex}"


def validate_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or _THREAD_ID_RE.fullmatch(session_id) is None:
        raise SessionIdError("sessionId 必须是 1-64 位 ASCII 字母、数字、连字符或下划线")
    return session_id


@dataclass
class TurnOutcome:
    """一次 turn 的执行结果。

    Attributes:
        stop_reason: ACP ``StopReason``。
        usage_payload: DeerFlow ``end`` 事件里的累计 token 用量，可能为 None。
        error: 后端异常（若有），由调用方转成 JSON-RPC error。
        escalated: 取消宽限期内后端未协作退出，桥动用了强制终止手段。
        worker_pid: 执行本轮的 worker 子进程 PID；进程内执行路径为 None。
        worker_pgid: worker 进程组 ID；进程内执行路径为 None。
        worker_killed: 是否真的执行过 ``killpg(SIGKILL)``。
        worker_reaped: 强制终止后是否已确认进程退出。False 表示旧执行体可能
            仍在写同一条 DeerFlow thread——这是必须如实上报的严重状态。
    """

    stop_reason: str
    usage_payload: dict[str, Any] | None = None
    error: BaseException | None = None
    escalated: bool = False
    worker_pid: int | None = None
    worker_pgid: int | None = None
    worker_killed: bool = False
    worker_reaped: bool = True


@dataclass
class Session:
    """单个 ACP 会话的运行时状态。"""

    session_id: str
    cwd: str
    #: 当前 turn 的取消标志。每个 turn 换一个新对象——虽然 worker 已被强制终止，
    #: 换新对象仍是最省心的做法：任何持有旧对象的残留引用都看不到新 turn 的状态。
    cancel_event: threading.Event = field(default_factory=threading.Event)
    #: 当前 turn 的完成信号（供 stdin 断连时等待收尾）
    turn_finished: asyncio.Event | None = None
    #: 最近一次 turn 的 worker PID，仅用于观测与日志
    last_worker_pid: int | None = None
    _running: bool = False

    @property
    def running(self) -> bool:
        return self._running


class SessionRegistry:
    """会话注册表：新建、查找、恢复与关闭。"""

    def __init__(
        self,
        backend: DeerFlowBackend,
        config: BridgeConfig,
        *,
        runner: TurnRunner | None = None,
    ) -> None:
        self._backend = backend
        self._config = config
        self._sessions: dict[str, Session] = {}
        # 默认按后端类型选执行器：真实的嵌入式后端由 worker 子进程加载（可强制终止），
        # 直接注入的后端对象跨不过进程边界，只能在进程内跑（无强制终止能力）。
        self._runner = runner or self._default_runner(backend, config)

    @staticmethod
    def _default_runner(backend: DeerFlowBackend, config: BridgeConfig) -> TurnRunner:
        from .backend import EmbeddedDeerFlowBackend

        if isinstance(backend, EmbeddedDeerFlowBackend):
            return SubprocessTurnRunner(config)
        return InProcessTurnRunner(backend)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def create(self, cwd: str) -> Session:
        session = Session(session_id=new_session_id(), cwd=cwd)
        self._sessions[session.session_id] = session
        logger.info("新建会话 %s", session.session_id)
        return session

    def resume(self, session_id: str, cwd: str) -> Session:
        """恢复会话。

        本进程已有则直接复用；否则以 DeerFlow checkpoint 是否存在为唯一依据。
        checkpoint 不存在时抛 :class:`UnknownSessionError`——绝不静默新建，
        否则客户端会以为历史上下文还在，实际却从空白开始。
        """
        validate_session_id(session_id)
        existing = self._sessions.get(session_id)
        if existing is not None:
            return existing

        if not self._backend.thread_exists(session_id):
            raise UnknownSessionError(session_id)

        session = Session(session_id=session_id, cwd=cwd)
        self._sessions[session_id] = session
        logger.info("从 DeerFlow checkpoint 恢复会话 %s", session_id)
        return session

    def get(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise UnknownSessionError(session_id)
        return session

    def close(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.cancel_event.set()
            logger.info("关闭会话 %s", session_id)

    def cancel(self, session_id: str) -> None:
        """请求取消。未知会话按 ACP 语义静默忽略（cancel 是通知，无响应）。"""
        session = self._sessions.get(session_id)
        if session is None:
            logger.warning("收到未知会话的取消通知：%s", session_id)
            return
        session.cancel_event.set()

    @property
    def active_sessions(self) -> list[Session]:
        return [s for s in self._sessions.values() if s.running]

    def terminate_all_workers(self) -> list[int]:
        """关停兜底：强制终止执行器仍持有的所有 worker 进程组。

        协作式取消 + 宽限期覆盖不了「桥即刻退出、事件循环被拆掉」这种形态；
        没有这一步，正在跑的 worker 会成为孤儿继续消耗模型配额并写半截状态。
        进程内执行器没有强制终止能力，返回空列表。
        """
        terminate = getattr(self._runner, "terminate_all", None)
        return list(terminate()) if callable(terminate) else []

    # ------------------------------------------------------------------
    # turn 执行
    # ------------------------------------------------------------------

    async def run_turn(
        self,
        session: Session,
        message: str,
        on_event: Any,
    ) -> TurnOutcome:
        """驱动一次 turn，把 DeerFlow 事件逐条交给 ``on_event`` 协程。

        ``_running`` 的释放时机是这里最关键的语义：只有执行器返回之后才释放，
        而子进程执行器在返回前已确认 worker 进程组被回收。因此下一个 turn 启动时，
        不可能存在另一个进程还在写同一条 DeerFlow thread。

        Args:
            session: 目标会话。
            message: 用户消息文本。
            on_event: ``async (event_type, data) -> None``，在事件循环中执行。
        """
        if session.running:
            raise TurnAlreadyRunningError(session.session_id)

        session._running = True
        session.cancel_event = threading.Event()
        session.turn_finished = asyncio.Event()

        cancel_event = session.cancel_event
        session_id = session.session_id
        grace = self._config.cancel_grace_seconds

        try:
            result: RunResult = await self._runner.execute(
                session_id=session_id,
                message=message,
                on_event=on_event,
                cancel_event=cancel_event,
                grace=grace,
            )
        finally:
            # 执行器返回即代表旧执行体已终结（子进程路径已确认回收），
            # 此时释放 running 才是安全的。
            session._running = False
            if session.turn_finished is not None:
                session.turn_finished.set()

        session.last_worker_pid = result.worker_pid

        if result.escalated:
            logger.warning(
                "会话 %s 的 turn 在 %.1fs 宽限期内未协作退出（worker pid=%s pgid=%s，已强制终止=%s，已确认回收=%s）",
                session_id,
                grace,
                result.worker_pid,
                result.worker_pgid,
                result.worker_killed,
                result.worker_reaped,
            )

        outcome = TurnOutcome(
            stop_reason="end_turn",
            usage_payload=result.usage_payload,
            escalated=result.escalated,
            worker_pid=result.worker_pid,
            worker_pgid=result.worker_pgid,
            worker_killed=result.worker_killed,
            worker_reaped=result.worker_reaped,
        )

        if result.cancelled or cancel_event.is_set():
            outcome.stop_reason = "cancelled"
            return outcome

        if result.error is not None:
            outcome.stop_reason = "refusal"
            outcome.error = result.error
            return outcome

        return outcome


__all__ = [
    "BackendUnavailableError",
    "Session",
    "SessionIdError",
    "SessionRegistry",
    "TurnAlreadyRunningError",
    "TurnOutcome",
    "UnknownSessionError",
    "new_session_id",
    "validate_session_id",
]
