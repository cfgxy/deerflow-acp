"""DeerFlow 后端适配层。

桥不复制 DeerFlow 的编排能力，只通过 ``DeerFlowClient`` 这一个入口消费它：

* ``stream()``  —— 同步生成器，桥在工作线程中驱动，``close()`` 是协作式取消的抓手
* ``get_thread()`` —— 判定某个 thread 是否真实存在 checkpoint（会话恢复的唯一依据）

``DeerFlowBackend`` 是一个 Protocol，真实实现 ``EmbeddedDeerFlowBackend`` 在
首次真正需要跑 turn 时才 import 并构造 ``DeerFlowClient``——``initialize`` 阶段
绝不触发重型后端加载。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from .config import BridgeConfig
from .logging_setup import get_logger
from .sanitize import describe_exception

logger = get_logger("backend")


class BackendUnavailableError(RuntimeError):
    """DeerFlow 运行时不可用（未安装、配置缺失或初始化失败）。"""


@runtime_checkable
class DeerFlowBackend(Protocol):
    """桥所依赖的 DeerFlow 能力的最小面。"""

    def stream(self, message: str, *, thread_id: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """产出 ``(event_type, data)`` 二元组；必须是可 ``close()`` 的生成器。"""
        ...

    def thread_exists(self, thread_id: str) -> bool:
        """thread 是否已有 checkpoint。未知 thread 必须返回 False。"""
        ...

    def history(self, thread_id: str) -> list[dict[str, Any]]:
        """返回最新 checkpoint 中已序列化的消息列表；无历史时返回空列表。"""
        ...


class EmbeddedDeerFlowBackend:
    """基于 ``deerflow.DeerFlowClient`` 的嵌入式后端（进程内调用）。"""

    def __init__(self, config: BridgeConfig) -> None:
        self._config = config
        self._client: Any | None = None

    # ------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            # 延迟 import：避免 initialize 阶段拉起重型依赖。
            # 注意入口在 ``deerflow.client``，``deerflow`` 顶层不导出该符号。
            from deerflow.client import DeerFlowClient
        except Exception as exc:  # noqa: BLE001 —— 上游可能抛任意导入期异常
            raise BackendUnavailableError(f"无法导入 DeerFlow 运行时：{describe_exception(exc)}") from exc

        kwargs: dict[str, Any] = dict(self._config.client_extra)
        if self._config.deerflow_config_path:
            kwargs["config_path"] = self._config.deerflow_config_path
        if self._config.model_name:
            kwargs["model_name"] = self._config.model_name
        kwargs.setdefault("thinking_enabled", self._config.thinking_enabled)

        try:
            self._client = DeerFlowClient(**kwargs)
        except Exception as exc:  # noqa: BLE001
            # 只暴露异常类型，不回显消息与配置内容——两者都可能带秘密。
            # 完整异常仍挂在 __cause__ 上，本地排障时可取。
            raise BackendUnavailableError(f"DeerFlowClient 初始化失败：{describe_exception(exc)}") from exc
        logger.info("DeerFlowClient 已初始化")
        return self._client

    # ------------------------------------------------------------------

    def stream(self, message: str, *, thread_id: str) -> Iterator[tuple[str, dict[str, Any]]]:
        client = self._ensure_client()
        for event in client.stream(message, thread_id=thread_id):
            yield event.type, dict(event.data or {})

    def _checkpoints(self, thread_id: str) -> list[dict[str, Any]]:
        client = self._ensure_client()
        try:
            thread = client.get_thread(thread_id)
        except Exception as exc:  # noqa: BLE001
            # checkpointer 后端异常与「线程不存在」必须分账：这里如实上抛，
            # 由调用方转成 internal error，而不是伪装成「未知会话」。
            raise BackendUnavailableError(f"读取 DeerFlow thread 失败：{describe_exception(exc)}") from exc
        checkpoints = thread.get("checkpoints") if isinstance(thread, dict) else None
        return [cp for cp in checkpoints if isinstance(cp, dict)] if isinstance(checkpoints, list) else []

    def thread_exists(self, thread_id: str) -> bool:
        return bool(self._checkpoints(thread_id))

    def history(self, thread_id: str) -> list[dict[str, Any]]:
        checkpoints = self._checkpoints(thread_id)
        # get_thread 按 ts 升序排列，最后一个 checkpoint 携带最完整的消息列表。
        for checkpoint in reversed(checkpoints):
            messages = (checkpoint.get("values") or {}).get("messages")
            if isinstance(messages, list) and messages:
                return [m for m in messages if isinstance(m, dict)]
        return []
