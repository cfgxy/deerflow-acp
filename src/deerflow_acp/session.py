"""ACP 会话注册与 turn 生命周期控制。

映射关系：一个 ACP ``sessionId`` 一对一绑定一个 DeerFlow ``thread_id``，
两者取同一个字符串值——DeerFlow 的 checkpointer 以 thread_id 为主键，
共用同一标识才能让「跨进程恢复」退化成「用同一个 thread_id 继续 stream」，
桥自身不需要维护任何持久化映射表。

DeerFlow ``stream()`` 是同步生成器，桥在专用工作线程中驱动它，
通过 ``asyncio.Queue`` 把事件送回事件循环。取消采用协作式：
置标志 → 工作线程在下一个 yield 边界调用 ``generator.close()``；
超过宽限期仍未退出则放弃等待并如实标记为 escalated（见 ``TurnOutcome``）。

**宽限期由事件循环侧计时，而不是等工作线程报到。** 后端完全可能卡在第一个
（或下一个）yield **之前**——模型调用、工具执行都在 ``next(generator)`` 内部，
此时工作线程既观察不到取消标志，也发不出完成信号。若事件循环无条件等待队列，
``session/prompt`` 就会永久挂起，宽限期形同虚设。因此取消一旦触发，
事件循环自己按 ``cancel_grace_seconds`` 倒计时；超时即把该 turn 判为 escalated
并**弃用**那个工作线程：弃用后它的事件与完成信号一律丢弃，且它持有的是本轮
专属的取消标志与队列，不会干扰同一 session 的后续 turn。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import re
import threading
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .backend import BackendUnavailableError, DeerFlowBackend
from .config import BridgeConfig
from .logging_setup import get_logger

logger = get_logger("session")

# 与 DeerFlow ``deerflow.utils.thread_id.validate_thread_id`` 完全一致的约束：
# 1-64 位 ASCII 字母、数字、连字符或下划线。ACP sessionId 直接作为 thread_id
# 使用，因此必须在桥这一层就拒掉不合法的值，而不是让 DeerFlow 抛 ValueError。
_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_SENTINEL_DONE = object()

#: 工作线程在阻塞等待队列时的取消轮询间隔。只影响「队列满 + 已取消」这一条
#: 慢路径；正常路径仍由 future 完成唤醒，不受这个间隔拖慢。
_CANCEL_POLL_SECONDS = 0.05


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
        escalated: 取消宽限期内后端未协作退出，桥已放弃等待。
    """

    stop_reason: str
    usage_payload: dict[str, Any] | None = None
    error: BaseException | None = None
    escalated: bool = False


@dataclass
class Session:
    """单个 ACP 会话的运行时状态。"""

    session_id: str
    cwd: str
    #: 当前 turn 的取消标志；置位后工作线程会在下一个 yield 边界关闭生成器。
    #: 每个 turn 换一个新对象——被弃用的工作线程仍持有旧对象，
    #: 因此它永远看不到新 turn 的状态，新 turn 也不会被它的取消标志误伤。
    cancel_event: threading.Event = field(default_factory=threading.Event)
    #: 当前 turn 的完成信号（供 stdin 断连时等待收尾）
    turn_finished: asyncio.Event | None = None
    _running: bool = False

    @property
    def running(self) -> bool:
        return self._running


class SessionRegistry:
    """会话注册表：新建、查找、恢复与关闭。"""

    def __init__(self, backend: DeerFlowBackend, config: BridgeConfig) -> None:
        self._backend = backend
        self._config = config
        self._sessions: dict[str, Session] = {}
        #: 宽限期内未协作退出、已被弃用的工作线程（仅用于观测与收割）
        self._abandoned: set[threading.Thread] = set()

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

        Args:
            session: 目标会话。
            message: 用户消息文本。
            on_event: ``async (event_type, data) -> None``，在事件循环中执行。
        """
        if session.running:
            raise TurnAlreadyRunningError(session.session_id)

        session._running = True
        # 换新对象而不是 clear()：上一轮若被弃用，那个工作线程还握着旧 Event，
        # 复用同一对象会让它的取消状态渗进本轮。
        session.cancel_event = threading.Event()
        session.turn_finished = asyncio.Event()

        loop = asyncio.get_running_loop()
        # 本轮专属队列：被弃用的工作线程只会往它自己那份队列里写，
        # 写进去也没人读，不会污染后续 turn。
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=256)
        cancel_event = session.cancel_event
        backend = self._backend
        session_id = session.session_id

        # 工作线程独占生成器：生成器的 next()/close() 必须在同一线程调用，
        # 跨线程 close() 在 CPython 中会破坏生成器帧状态。
        def put_with_backpressure(event: tuple[str, dict[str, Any]]) -> bool:
            """把事件放进队列，等待期间保持对取消标志的响应。

            返回 False 表示本轮已被取消/弃用，工作线程应立即收摊。
            直接 ``fut.result()`` 无限等是不行的：队列满时（事件循环侧已经不再
            读取，比如本轮已被弃用）工作线程会永远卡在这里，连生成器都关不掉。
            """
            fut = asyncio.run_coroutine_threadsafe(queue.put(event), loop)
            while True:
                if cancel_event.is_set():
                    fut.cancel()
                    return False
                try:
                    fut.result(timeout=_CANCEL_POLL_SECONDS)
                    return True
                except concurrent.futures.TimeoutError:
                    continue

        def worker() -> None:
            generator: Iterator[tuple[str, dict[str, Any]]] | None = None
            error: BaseException | None = None
            cancelled = False
            try:
                generator = backend.stream(message, thread_id=session_id)
                for event in generator:
                    if cancel_event.is_set():
                        cancelled = True
                        break
                    if not put_with_backpressure(event):
                        cancelled = True
                        break
                    if cancel_event.is_set():
                        cancelled = True
                        break
            except BaseException as exc:  # noqa: BLE001 —— 后端可抛任意异常，必须完整回传
                error = exc
            finally:
                if generator is not None and cancelled:
                    close = getattr(generator, "close", None)
                    if callable(close):
                        try:
                            close()
                        except BaseException:  # noqa: BLE001
                            logger.warning("关闭 DeerFlow 生成器时出错", exc_info=True)
                loop.call_soon_threadsafe(queue.put_nowait, (_SENTINEL_DONE, error, cancelled))

        thread = threading.Thread(target=worker, name=f"deerflow-turn-{session_id}", daemon=True)
        thread.start()

        usage_payload: dict[str, Any] | None = None
        error: BaseException | None = None
        cancelled = False
        escalated = False
        grace = self._config.cancel_grace_seconds
        #: 取消触发后的绝对截止时刻；None 表示尚未取消
        deadline: float | None = None

        # 持久 getter：每轮循环复用同一个取数任务，超时只是「这轮没等到」，
        # 不取消它。若改用 ``asyncio.wait_for(queue.get(), ...)``，每次超时都会
        # 取消一个已经排进 ``_getters`` 的等待者，在取消竞态下容易丢事件。
        getter: asyncio.Task[Any] | None = None
        try:
            while True:
                if deadline is None and cancel_event.is_set():
                    # 取消一进来就由事件循环侧计时，不依赖工作线程报到——
                    # 它可能正卡在 next(generator) 里，永远到不了下一个 yield。
                    deadline = loop.time() + grace

                if getter is None:
                    getter = asyncio.ensure_future(queue.get())

                if deadline is None:
                    # 未取消时也要定期醒来，否则取消通知到达时我们正睡在
                    # queue.get() 上，宽限期根本不会开始计时。
                    timeout = _CANCEL_POLL_SECONDS
                else:
                    timeout = deadline - loop.time()
                    if timeout <= 0:
                        escalated = True
                        cancelled = True
                        break

                await asyncio.wait({getter}, timeout=timeout)
                if not getter.done():
                    if deadline is not None and loop.time() >= deadline:
                        escalated = True
                        cancelled = True
                        break
                    continue

                item = getter.result()
                getter = None

                if isinstance(item, tuple) and len(item) == 3 and item[0] is _SENTINEL_DONE:
                    _, error, cancelled = item
                    break
                event_type, data = item
                if event_type == "end":
                    usage_payload = data.get("usage") if isinstance(data, dict) else None
                await on_event(event_type, data)
        finally:
            if getter is not None and not getter.done():
                getter.cancel()
            session._running = False
            if session.turn_finished is not None:
                session.turn_finished.set()

        if escalated:
            # 弃用这个工作线程：它握着本轮专属的 cancel_event 与 queue，
            # 醒来后会自行收摊；即便一直卡着，也碰不到 session 的新状态。
            # 它是 daemon 线程，不会阻止进程退出。
            self._abandoned.add(thread)
            self._reap_abandoned()
            logger.warning(
                "会话 %s 的 DeerFlow turn 在 %.1fs 宽限期内未协作退出，已弃用工作线程 %s",
                session_id,
                grace,
                thread.name,
            )
            return TurnOutcome(stop_reason="cancelled", usage_payload=usage_payload, escalated=True)

        if cancelled or cancel_event.is_set():
            # 工作线程已经发出完成信号，join 只是确认 close() 已跑完。
            await asyncio.to_thread(thread.join, grace)
            escalated = thread.is_alive()
            if escalated:
                self._abandoned.add(thread)
                logger.warning(
                    "会话 %s 的工作线程 %s 在 close() 后仍未退出，已弃用",
                    session_id,
                    thread.name,
                )
            return TurnOutcome(stop_reason="cancelled", usage_payload=usage_payload, escalated=escalated)

        if error is not None:
            return TurnOutcome(stop_reason="refusal", usage_payload=usage_payload, error=error)

        return TurnOutcome(stop_reason="end_turn", usage_payload=usage_payload)

    def _reap_abandoned(self) -> None:
        """清掉已经自行退出的弃用线程，避免集合无界增长。"""
        self._abandoned = {t for t in self._abandoned if t.is_alive()}


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
