"""DeerFlow worker 子进程：一次 turn 的可终止执行载体。

为什么必须是子进程：DeerFlow 的模型调用与工具执行都发生在 ``next(generator)``
内部，线程模型下没有任何办法强制打断它。取消超时后，那个线程会继续跑完模型
调用、继续执行工具、继续往同一个 ``thread_id`` 写 checkpoint——而桥此时已经
释放了 ``_running``，下一个 turn 会和它并发写同一条 DeerFlow thread。
把执行放进独立进程组后，``killpg`` 能真正终止它，包括它派生的工具子进程。

**这仍然是嵌入式 API 调用**：worker 里跑的是 ``DeerFlowClient``，不是 DeerFlow
CLI 的文本包装；子进程只是隔离载体，事件语义与进程内调用完全一致。

生命周期：

1. 父进程以 ``setsid`` 起本模块，写一行 job JSON 到 stdin
2. worker 抢占 fd 1 作为 IPC 通道（DeerFlow 往 stdout 打的东西全落 stderr）
3. worker 迭代 ``backend.stream()``，逐条把事件写成 ndJSON
4. 收到 SIGTERM → 置协作退出标志，在下一个事件边界 ``close()`` 生成器
5. 父进程宽限期内未见退出 → ``killpg(SIGKILL)``（见 :mod:`deerflow_acp.session`）
"""

from __future__ import annotations

import json
import os
import signal
import sys
from collections.abc import Iterator
from typing import IO, Any

from . import ipc
from .config import BridgeConfig
from .logging_setup import configure_logging, get_logger, redirect_root_logging_to_stderr
from .sanitize import describe_exception

logger = get_logger("worker")

#: 允许测试把后端换成脚本化替身。值形如 ``module:callable``，
#: 该 callable 接受 BridgeConfig 返回 DeerFlowBackend。生产路径不设此变量。
BACKEND_FACTORY_ENV = "DEERFLOW_ACP_WORKER_BACKEND"

#: 收到 SIGTERM 后置位；只在信号处理器里做这一件事（信号处理器里不能做别的）
_stopping = False


def _on_terminate(signum: int, frame: Any) -> None:
    global _stopping
    _stopping = True


def _isolate_ipc_fd() -> IO[bytes]:
    """把 fd 1 抢占为 IPC 专用通道，其余 stdout 写入全部改道 stderr。

    与 :func:`deerflow_acp.cli.isolate_stdout` 同一手法，理由也相同：
    DeerFlow / LangGraph / C 扩展里的 ``print`` 一旦落到 fd 1 就会撕裂消息流。
    """
    ipc_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(ipc_fd, "wb", buffering=0)


def _load_backend(config: BridgeConfig) -> Any:
    spec = os.environ.get(BACKEND_FACTORY_ENV)
    if not spec:
        from .backend import EmbeddedDeerFlowBackend

        return EmbeddedDeerFlowBackend(config)

    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(f"{BACKEND_FACTORY_ENV} 必须形如 module:callable")
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, attr)(config)


def _read_job(stream: IO[str]) -> dict[str, Any]:
    line = stream.readline()
    if not line:
        raise SystemExit(0)
    return json.loads(line)


def run(job: dict[str, Any], out: IO[bytes]) -> int:
    """执行一次 turn，把事件写进 IPC 通道。返回进程退出码。"""
    config = BridgeConfig(**job.get("config", {}))
    message = job["message"]
    thread_id = job["thread_id"]

    try:
        backend = _load_backend(config)
    except BaseException as exc:  # noqa: BLE001 —— 后端构造可抛任意异常
        # 只回传类型名：构造期异常最常见的内容就是配置与凭据。
        out.write(ipc.encode(ipc.error_message(exc)))
        logger.error("worker 后端构造失败：%s", describe_exception(exc))
        return 1

    out.write(ipc.encode({"t": ipc.MSG_READY, "pid": os.getpid()}))

    generator: Iterator[tuple[str, dict[str, Any]]] | None = None
    try:
        generator = backend.stream(message, thread_id=thread_id)
        for event_type, data in generator:
            if _stopping:
                break
            out.write(ipc.encode(ipc.event_message(event_type, dict(data or {}))))
            if _stopping:
                break
    except BaseException as exc:  # noqa: BLE001 —— 后端可抛任意异常，必须如实分类
        out.write(ipc.encode(ipc.error_message(exc)))
        logger.error("worker turn 失败：%s", describe_exception(exc))
        return 1
    finally:
        if generator is not None and _stopping:
            close = getattr(generator, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as exc:  # noqa: BLE001
                    # 绝不能用 exc_info=：close() 抛出的异常同样可能夹带凭据，
                    # 而 stderr 是客户端可见的输出面。只留类型名。
                    logger.warning("关闭 DeerFlow 生成器时出错：%s", describe_exception(exc))

    out.write(ipc.encode({"t": ipc.MSG_DONE, "cancelled": _stopping}))
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_logging(None)
    redirect_root_logging_to_stderr(None)

    signal.signal(signal.SIGTERM, _on_terminate)
    signal.signal(signal.SIGINT, _on_terminate)

    out = _isolate_ipc_fd()
    job = _read_job(sys.stdin)
    try:
        return run(job, out)
    finally:
        out.flush()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
