"""turn 执行器：把「谁来驱动 DeerFlow 生成器」与「会话状态管理」解耦。

两种实现：

* :class:`SubprocessTurnRunner` —— **生产路径**。每次 turn 起一个独立进程组的
  worker 子进程，取消宽限期超时后 ``killpg`` 强制终止，**确认退出后才返回**。
  这是「旧 turn 绝不干扰同 session 后续 turn」的唯一可靠保证：DeerFlow 的模型
  调用与工具执行卡在 ``next(generator)`` 里时，线程无法被打断，进程可以。
* :class:`InProcessTurnRunner` —— 仅用于把 :class:`DeerFlowBackend` 对象直接
  注入进来的单元测试。Python 对象无法跨进程传递，这条路径为它们保留。
  **它没有强制终止能力**：宽限期超时后只能弃用工作线程，被弃用的线程仍可能
  继续写同一条 DeerFlow thread。因此生产路径绝不使用它。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import dataclasses
import json
import os
import signal
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from . import ipc
from .backend import BackendUnavailableError, DeerFlowBackend
from .config import BridgeConfig
from .logging_setup import get_logger
from .sanitize import describe_exception

logger = get_logger("runner")

#: 轮询取消标志的间隔。取消标志由事件循环侧置位，执行器必须定期醒来观察它，
#: 否则取消到达时我们正睡在「等下一条事件」上，宽限期根本不会开始计时。
_CANCEL_POLL_SECONDS = 0.05

#: ``killpg(SIGKILL)`` 之后等待进程被回收的时限。内核层面这是即时的，
#: 给出上限只是为了绝不无限等待。
_REAP_TIMEOUT_SECONDS = 10.0

#: 确认「整个进程组」消失时的轮询间隔。worker 主进程退出 ≠ 进程组空了：
#: DeerFlow 的工具可能派生出仍在同一 PGID 里跑的子进程，它们不是我们的子进程，
#: 无法 ``wait()``，只能用 ``killpg(pgid, 0)`` 探活。
_GROUP_POLL_SECONDS = 0.02

#: 等待进程组彻底排空的时限。超时即判定「未确认回收」，由会话层隔离该 session。
_GROUP_DRAIN_TIMEOUT_SECONDS = 5.0

_SENTINEL_DONE = object()


class RemoteBackendError(RuntimeError):
    """worker 子进程里后端抛出的异常在父进程侧的代理。

    只携带**类型名**：异常消息、args 与 traceback 一律不跨进程传输
    （见 :mod:`deerflow_acp.ipc`），因此这里也无从泄露。
    """

    def __init__(self, error_class: str) -> None:
        super().__init__(error_class)
        self.error_class = error_class


class WorkerStartError(RuntimeError):
    """worker 子进程无法启动或在 ready 之前就退出。"""


@dataclass
class RunResult:
    """一次 turn 的执行结果（不含会话层语义）。

    Attributes:
        cancelled: 本轮被取消。
        escalated: 宽限期内未协作退出，桥动用了强制手段。
        error: 后端异常；worker 路径下是 :class:`RemoteBackendError`。
        usage_payload: ``end`` 事件里的用量。
        worker_pid: worker 子进程 PID；进程内路径为 None。
        worker_pgid: worker 进程组 ID；进程内路径为 None。
        worker_killed: 是否真的执行过 ``killpg(SIGKILL)``。
        worker_reaped: 强制终止后是否已确认进程退出。
    """

    cancelled: bool = False
    escalated: bool = False
    error: BaseException | None = None
    usage_payload: dict[str, Any] | None = None
    worker_pid: int | None = None
    worker_pgid: int | None = None
    worker_killed: bool = False
    worker_reaped: bool = True


class TurnRunner(Protocol):
    async def execute(
        self,
        *,
        session_id: str,
        message: str,
        on_event: Any,
        cancel_event: threading.Event,
        grace: float,
    ) -> RunResult: ...


# ----------------------------------------------------------------------
# 生产路径：worker 子进程
# ----------------------------------------------------------------------


class SubprocessTurnRunner:
    """在独立进程组的 worker 子进程中执行 turn。"""

    def __init__(self, config: BridgeConfig) -> None:
        self._config = config
        #: 在途 worker 的进程组，供关停时兜底回收。协程内的 finally 覆盖不了
        #: 「事件循环整体被拆掉」这种关停形态，必须有一个进程级的收尾入口。
        self._live_pgids: set[int] = set()
        #: 未能确认终结的进程组，按 session_id 记录。会话层据此隔离该 session：
        #: 只要旧进程组里还可能有活着的进程，就绝不允许同一 thread_id 上再开 turn。
        #: 这些 pgid **不从** ``_live_pgids`` 移除——关停兜底还要继续尝试收它们。
        self._unreaped: dict[str, int] = {}

    def take_unreaped(self, session_id: str) -> int | None:
        """取出并清除某 session 的「未确认终结」记录。

        会话层在 turn 收尾时调用。返回非 None 即表示该 session 必须被隔离。
        """
        return self._unreaped.pop(session_id, None)

    def terminate_all(self) -> list[int]:
        """强制终止所有在途 worker 进程组，返回被处理的 pgid。

        由关停路径在放弃等待之后调用：桥即将退出，此时任何仍在跑的 worker 都会
        变成孤儿，继续烧模型配额并往 checkpoint 里写半截状态。
        """
        pgids = sorted(self._live_pgids)
        for pgid in pgids:
            if self._signal_group(pgid, signal.SIGKILL, "关停"):
                logger.warning("关停时强制终止残留 worker 进程组 %s", pgid)
        self._live_pgids.clear()
        return pgids

    def _job_payload(self, session_id: str, message: str) -> bytes:
        # 只传后端构造所需的配置，且这些字段本身不含凭据——DeerFlow 的 key
        # 仍由它自己的本地注入机制（.env + config.yaml 占位符）从环境读取，
        # 桥既不读也不转发它们。
        config_fields = {
            f.name: getattr(self._config, f.name)
            for f in dataclasses.fields(self._config)
            if f.name in {"deerflow_config_path", "model_name", "thinking_enabled", "client_extra"}
        }
        payload = {"session_id": session_id, "message": message, "thread_id": session_id, "config": config_fields}
        return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")

    async def _spawn(self, session_id: str, message: str) -> asyncio.subprocess.Process:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "deerflow_acp.worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            # stderr 继承桥进程：worker 的日志与桥的日志同流，都只走 stderr。
            stderr=None,
            # 独立进程组——这是 killpg 能连带回收 DeerFlow 派生的工具子进程的前提。
            start_new_session=True,
            env=self._worker_env(),
        )
        assert proc.stdin is not None
        proc.stdin.write(self._job_payload(session_id, message))
        with contextlib.suppress(Exception):
            await proc.stdin.drain()
        proc.stdin.close()
        return proc

    def _worker_env(self) -> dict[str, str]:
        """worker 的环境变量。

        必须沿用父进程环境：DeerFlow 的凭据正是通过环境注入的，剥掉就跑不起来。
        桥自己不向其中**添加**任何东西，也不读取其中任何凭据。
        """
        return dict(os.environ)

    async def execute(
        self,
        *,
        session_id: str,
        message: str,
        on_event: Any,
        cancel_event: threading.Event,
        grace: float,
    ) -> RunResult:
        try:
            proc = await self._spawn(session_id, message)
        except Exception as exc:  # noqa: BLE001
            raise WorkerStartError(describe_exception(exc)) from exc

        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            pgid = proc.pid

        result = RunResult(worker_pid=proc.pid, worker_pgid=pgid)
        self._live_pgids.add(pgid)
        assert proc.stdout is not None
        loop = asyncio.get_running_loop()

        deadline: float | None = None
        terminated = False
        reader: asyncio.Task[bytes] | None = None

        try:
            while True:
                if deadline is None and cancel_event.is_set():
                    # 取消一进来就先请求协作退出（SIGTERM），同时由事件循环侧计时。
                    # 不依赖 worker 报到——它可能正卡在模型调用里。
                    deadline = loop.time() + grace
                    if not terminated:
                        terminated = True
                        self._signal_group(pgid, signal.SIGTERM, session_id)

                if reader is None:
                    reader = asyncio.ensure_future(proc.stdout.readline())

                if deadline is None:
                    timeout = _CANCEL_POLL_SECONDS
                else:
                    timeout = deadline - loop.time()
                    if timeout <= 0:
                        result.escalated = True
                        result.cancelled = True
                        break

                await asyncio.wait({reader}, timeout=timeout)
                if not reader.done():
                    if deadline is not None and loop.time() >= deadline:
                        result.escalated = True
                        result.cancelled = True
                        break
                    continue

                line = reader.result()
                reader = None
                if not line:
                    # worker 的 IPC 通道关闭：进程已结束或即将结束。
                    break

                message_obj = ipc.decode(line)
                if message_obj is None:
                    continue

                kind = message_obj.get("t")
                if kind == ipc.MSG_EVENT:
                    event_type = message_obj.get("type")
                    data = message_obj.get("data")
                    if not isinstance(data, dict) or not isinstance(event_type, str):
                        continue
                    if event_type == "end":
                        usage = data.get("usage")
                        result.usage_payload = usage if isinstance(usage, dict) else None
                    await on_event(event_type, data)
                elif kind == ipc.MSG_DONE:
                    result.cancelled = bool(message_obj.get("cancelled"))
                    break
                elif kind == ipc.MSG_ERROR:
                    result.error = self._to_error(str(message_obj.get("cls") or "UnknownError"))
                    break
        except BaseException as exc:
            # 本协程被取消（桥收到 SIGINT/SIGTERM 后 cancel 在途 turn），或事件下发
            # （``on_event`` → ``session/update``）抛异常时，绝不能带着活着的 worker
            # 离开——那就是孤儿进程。这里必须先用**同步**的 killpg：`await` 在取消
            # 传播期间可能再次被打断，同步系统调用不会。
            self._signal_group(pgid, signal.SIGKILL, session_id)

            confirmed = False
            if not isinstance(exc, asyncio.CancelledError):
                # 非取消路径（典型是 on_event 抛错）：事件循环仍在正常运转，
                # 因此**必须**在这里确认整个进程组已排空再把异常抛上去。
                # 否则会话被释放、工具子进程却还在跑，与主线验收直接冲突。
                with contextlib.suppress(Exception):
                    confirmed = await self._await_group_gone(proc, pgid)

            if confirmed:
                self._live_pgids.discard(pgid)
                logger.warning("会话 %s：turn 中断，worker 进程组 %s 已强制终止并确认排空", session_id, pgid)
            else:
                # 无法确认（协程正在被取消，或排空超时）：**保留** pgid，让关停兜底
                # 继续尝试回收；同时登记 unreaped，由会话层隔离该 session。
                self._unreaped[session_id] = pgid
                logger.error(
                    "会话 %s：turn 中断后无法确认 worker 进程组 %s 已排空，该会话将被隔离",
                    session_id,
                    pgid,
                )
            raise
        finally:
            if reader is not None and not reader.done():
                reader.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await reader

        killed, reaped = await self._shutdown(proc, pgid, session_id, escalated=result.escalated)
        result.worker_killed = killed
        result.worker_reaped = reaped
        if reaped:
            self._live_pgids.discard(pgid)
        else:
            # 同上：不摘 pgid，关停兜底还要再收一次；会话层据此隔离 session。
            self._unreaped[session_id] = pgid
        if cancel_event.is_set():
            result.cancelled = True
        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _to_error(error_class: str) -> BaseException:
        # 后端不可用要走专用错误码，必须在这里还原成同一个类型。
        if error_class == "BackendUnavailableError":
            return BackendUnavailableError("DeerFlow 后端不可用")
        return RemoteBackendError(error_class)

    @staticmethod
    def _signal_group(pgid: int, sig: int, session_id: str) -> bool:
        """向整个进程组发信号；进程组已消失时返回 False。"""
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return False
        except PermissionError:
            logger.warning("会话 %s：无权向进程组 %s 发信号", session_id, pgid)
            return False
        return True

    @staticmethod
    def _group_alive(pgid: int) -> bool:
        """进程组里是否还有存活进程。

        ``killpg(pgid, 0)`` 是唯一可用的探针：组内的工具子进程不是桥的子进程，
        ``waitpid`` 对它们无效，只有信号 0 能问出「这个组还在不在」。
        ``PermissionError`` 说明组内还有进程（只是我们无权发信号），按存活处理——
        宁可判定「未确认」也不能误报「已排空」。
        """
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    async def _await_group_gone(
        self,
        proc: asyncio.subprocess.Process,
        pgid: int,
        timeout: float = _GROUP_DRAIN_TIMEOUT_SECONDS,
    ) -> bool:
        """等到**整个进程组**排空，返回是否确认排空。

        两步都必需：
        1. ``proc.wait()`` 回收 worker 主进程——它是我们的子进程，不 wait 会留僵尸，
           而僵尸会让 ``killpg(pgid, 0)`` 永远报「组还活着」。
        2. 轮询 ``killpg(pgid, 0)``，直到组内再无进程。**worker 主进程退出不等于
           进程组空了**：DeerFlow 的工具可以派生仍在同一 PGID 里的子进程，它们
           继续执行副作用、继续写同一条 thread。
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        if proc.returncode is None:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=max(0.0, deadline - loop.time()))

        while loop.time() < deadline:
            if not self._group_alive(pgid):
                return True
            await asyncio.sleep(_GROUP_POLL_SECONDS)

        return not self._group_alive(pgid)

    async def _shutdown(
        self,
        proc: asyncio.subprocess.Process,
        pgid: int,
        session_id: str,
        *,
        escalated: bool,
    ) -> tuple[bool, bool]:
        """确保 worker **进程组**彻底排空，返回 ``(是否强杀, 是否确认回收)``。

        **这里必须等到整个进程组真的空了才返回 True**：调用方随后就会释放
        ``_running``，允许下一个 turn 在同一 DeerFlow ``thread_id`` 上启动。只要
        组内还有任何进程活着，它就可能继续执行工具副作用、继续写同一条 thread
        的 checkpoint。因此判据是「组已排空」，不是「主 PID 已退出」——后者在
        工具派生子进程的情况下会漏掉一整棵进程树。
        """
        if not escalated:
            # 正常/协作结束：worker 已写完 IPC 通道，给它一点时间自行退出。
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=_CANCEL_POLL_SECONDS * 20)
            if proc.returncode is not None and not self._group_alive(pgid):
                # 主进程已退出且组已排空，这才是真正的干净结束。
                return False, True
            if proc.returncode is not None:
                logger.warning(
                    "会话 %s：worker %s 已退出但进程组 %s 内仍有进程存活，强制回收整组",
                    session_id,
                    proc.pid,
                    pgid,
                )

        self._signal_group(pgid, signal.SIGKILL, session_id)
        drained = await self._await_group_gone(proc, pgid, timeout=_REAP_TIMEOUT_SECONDS)
        if not drained:
            logger.error(
                "会话 %s：worker %s 的进程组 %s 在 SIGKILL 后 %.1fs 内仍未排空",
                session_id,
                proc.pid,
                pgid,
                _REAP_TIMEOUT_SECONDS,
            )
            return True, False

        logger.warning(
            "会话 %s：worker %s（进程组 %s）未在宽限期内协作退出，已强制终止整个进程组并确认排空",
            session_id,
            proc.pid,
            pgid,
        )
        return True, True


# ----------------------------------------------------------------------
# 测试路径：进程内线程
# ----------------------------------------------------------------------


class InProcessTurnRunner:
    """在工作线程里驱动一个直接注入的 :class:`DeerFlowBackend` 对象。

    只服务于把后端对象注入进来的单元测试——Python 对象跨不过进程边界。
    **没有强制终止能力**：宽限期超时后只能弃用线程，被弃用的线程仍可能继续
    写同一条 DeerFlow thread。生产路径一律走 :class:`SubprocessTurnRunner`。
    """

    def __init__(self, backend: DeerFlowBackend) -> None:
        self._backend = backend
        self._abandoned: set[threading.Thread] = set()

    async def execute(
        self,
        *,
        session_id: str,
        message: str,
        on_event: Any,
        cancel_event: threading.Event,
        grace: float,
    ) -> RunResult:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=256)
        backend = self._backend

        def put_with_backpressure(event: tuple[str, dict[str, Any]]) -> bool:
            """把事件放进队列，等待期间保持对取消标志的响应。

            返回 False 表示本轮已被弃用，工作线程应立即收摊。直接 ``fut.result()``
            无限等是不行的：队列满时（事件循环侧已不再读取）线程会永远卡住。
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
            except BaseException as exc:  # noqa: BLE001
                error = exc
            finally:
                if generator is not None and cancelled:
                    close = getattr(generator, "close", None)
                    if callable(close):
                        try:
                            close()
                        except BaseException as exc:  # noqa: BLE001
                            # 绝不能用 exc_info=：close() 的异常同样可能夹带凭据，
                            # 而 stderr 是客户端可见的输出面。只留类型名。
                            logger.warning("关闭 DeerFlow 生成器时出错：%s", describe_exception(exc))
                loop.call_soon_threadsafe(queue.put_nowait, (_SENTINEL_DONE, error, cancelled))

        thread = threading.Thread(target=worker, name=f"deerflow-turn-{session_id}", daemon=True)
        thread.start()

        result = RunResult()
        deadline: float | None = None
        getter: asyncio.Task[Any] | None = None
        try:
            while True:
                if deadline is None and cancel_event.is_set():
                    deadline = loop.time() + grace

                if getter is None:
                    getter = asyncio.ensure_future(queue.get())

                if deadline is None:
                    timeout = _CANCEL_POLL_SECONDS
                else:
                    timeout = deadline - loop.time()
                    if timeout <= 0:
                        result.escalated = True
                        result.cancelled = True
                        break

                await asyncio.wait({getter}, timeout=timeout)
                if not getter.done():
                    if deadline is not None and loop.time() >= deadline:
                        result.escalated = True
                        result.cancelled = True
                        break
                    continue

                item = getter.result()
                getter = None

                if isinstance(item, tuple) and len(item) == 3 and item[0] is _SENTINEL_DONE:
                    _, result.error, result.cancelled = item
                    break
                event_type, data = item
                if event_type == "end":
                    result.usage_payload = data.get("usage") if isinstance(data, dict) else None
                await on_event(event_type, data)
        finally:
            if getter is not None and not getter.done():
                getter.cancel()

        if result.escalated:
            self._abandoned = {t for t in self._abandoned if t.is_alive()}
            self._abandoned.add(thread)
            logger.warning("会话 %s 的工作线程 %s 未在宽限期内退出，已弃用（无法强制终止）", session_id, thread.name)
            return result

        if result.cancelled or cancel_event.is_set():
            result.cancelled = True
            await asyncio.to_thread(thread.join, grace)
            if thread.is_alive():
                result.escalated = True
                self._abandoned.add(thread)

        return result


__all__ = [
    "InProcessTurnRunner",
    "RemoteBackendError",
    "RunResult",
    "SubprocessTurnRunner",
    "TurnRunner",
    "WorkerStartError",
]
