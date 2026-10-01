"""桥接器运行配置。

配置只从环境变量与命令行读取，桥自身不存储任何凭据：
DeerFlow 所需的模型 / 搜索 API key 仍由 DeerFlow 既有的本地秘密注入机制
（gitignored ``.env`` + ``config.yaml`` 中的 ``$VAR`` 占位符）提供，桥只是
把当前进程环境原样交给 ``DeerFlowClient``。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# 取消请求发出后，等待 DeerFlow 生成器协作式退出的时限（秒）。
# 超时后升级为强制手段（见 session.py 的 escalation 路径）。
DEFAULT_CANCEL_GRACE_SECONDS = 5.0
# stdin 断连后等待在途 turn 收尾的时限（秒），超时直接退出进程。
DEFAULT_SHUTDOWN_GRACE_SECONDS = 5.0
# 每个 turn 的 LangGraph super-step 预算。嵌入路径 DeerFlowClient 默认 100，
# 中间件多的图一轮工具循环就要消耗多个 super-step，自主任务很容易触顶
# GraphRecursionError；这里给足余量，仍可用环境变量按需收紧。
DEFAULT_RECURSION_LIMIT = 500


def _env_get(name: str, legacy_name: str) -> str | None:
    """新名优先；旧 ``DEERFLOW_ACP_`` 前缀名已废弃，仅作兼容回退。"""
    return os.environ.get(name) or os.environ.get(legacy_name)


def _env_float(name: str, legacy_name: str, default: float) -> float:
    raw = _env_get(name, legacy_name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_int(name: str, legacy_name: str, default: int) -> int:
    raw = _env_get(name, legacy_name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_flag(name: str, legacy_name: str, default: bool) -> bool:
    raw = _env_get(name, legacy_name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class BridgeConfig:
    """桥接器配置。

    Attributes:
        deerflow_config_path: DeerFlow ``config.yaml`` 路径；None 表示由
            DeerFlow 自行按其默认查找顺序解析。
        model_name: 覆盖 DeerFlow 默认模型；None 表示沿用 DeerFlow 配置。
        thinking_enabled: 是否请求模型输出推理内容（映射到 thought chunk）。
        cancel_grace_seconds: 协作式取消的等待时限。
        shutdown_grace_seconds: stdin 断连后的收尾时限。
        emit_usage_update: 是否额外下发 ``usage_update`` 通知。
            DeerFlow 只提供 input/output/total token 增量，ACP 的
            ``usage_update`` 要求 ``size``（上下文窗口）与 ``used``（当前占用）
            两个语义不同的字段，无法无损映射，默认关闭，避免伪造数据。
        context_window_tokens: 显式提供上下文窗口大小后才允许开启
            ``usage_update``；用于用户明知语义近似仍希望看到进度条的场景。
        client_extra: 透传给 ``DeerFlowClient`` 的额外构造参数。
        recursion_limit: 每个 turn 的 LangGraph ``recursion_limit``。
    """

    deerflow_config_path: str | None = None
    model_name: str | None = None
    thinking_enabled: bool = True
    cancel_grace_seconds: float = DEFAULT_CANCEL_GRACE_SECONDS
    shutdown_grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS
    emit_usage_update: bool = False
    context_window_tokens: int | None = None
    client_extra: dict[str, Any] = field(default_factory=dict)
    recursion_limit: int = DEFAULT_RECURSION_LIMIT

    @classmethod
    def from_env(cls) -> BridgeConfig:
        context_window = _env_get("DEER_FLOW_ACP_CONTEXT_WINDOW_TOKENS", "DEERFLOW_ACP_CONTEXT_WINDOW_TOKENS")
        try:
            context_window_tokens = int(context_window) if context_window else None
        except ValueError:
            context_window_tokens = None
        if context_window_tokens is not None and context_window_tokens <= 0:
            context_window_tokens = None

        emit_requested = _env_flag("DEER_FLOW_ACP_EMIT_USAGE_UPDATE", "DEERFLOW_ACP_EMIT_USAGE_UPDATE", False)
        return cls(
            deerflow_config_path=_env_get("DEER_FLOW_CONFIG_PATH", "DEERFLOW_ACP_CONFIG_PATH") or None,
            model_name=_env_get("DEER_FLOW_ACP_MODEL", "DEERFLOW_ACP_MODEL") or None,
            thinking_enabled=_env_flag("DEER_FLOW_ACP_THINKING", "DEERFLOW_ACP_THINKING", True),
            cancel_grace_seconds=_env_float(
                "DEER_FLOW_ACP_CANCEL_GRACE_SECONDS",
                "DEERFLOW_ACP_CANCEL_GRACE_SECONDS",
                DEFAULT_CANCEL_GRACE_SECONDS,
            ),
            shutdown_grace_seconds=_env_float(
                "DEER_FLOW_ACP_SHUTDOWN_GRACE_SECONDS",
                "DEERFLOW_ACP_SHUTDOWN_GRACE_SECONDS",
                DEFAULT_SHUTDOWN_GRACE_SECONDS,
            ),
            # 只有同时给出窗口大小，usage_update 才是有意义的；否则保持关闭。
            emit_usage_update=emit_requested and context_window_tokens is not None,
            context_window_tokens=context_window_tokens,
            recursion_limit=_env_int("DEER_FLOW_ACP_RECURSION_LIMIT", "DEERFLOW_ACP_RECURSION_LIMIT", DEFAULT_RECURSION_LIMIT),
        )
