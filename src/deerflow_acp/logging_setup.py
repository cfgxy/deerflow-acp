"""日志初始化：stdout 是 JSON-RPC 专用通道，所有日志只允许写 stderr。"""

from __future__ import annotations

import logging
import os
import sys

_LOGGER_NAME = "deerflow_acp"
_DEFAULT_LEVEL = "INFO"


def _resolve_level(explicit: str | None = None) -> int:
    raw = explicit or os.environ.get("DEERFLOW_ACP_LOG_LEVEL") or _DEFAULT_LEVEL
    level = logging.getLevelName(raw.strip().upper())
    return level if isinstance(level, int) else logging.INFO


def configure_logging(level: str | None = None) -> logging.Logger:
    """配置桥接器日志器，只挂 stderr handler，并阻断向 root 传播。

    向 root 传播会让第三方（DeerFlow / LangGraph）预置的 stdout handler
    有机会污染 JSON-RPC 通道，因此这里显式关闭 propagate。
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(_resolve_level(level))
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    return logger


def redirect_root_logging_to_stderr(level: str | None = None) -> None:
    """把 root logger 的输出统一钉到 stderr。

    DeerFlow 及其依赖会在 import 期调用 ``logging.basicConfig()``，默认
    handler 写 stderr，但个别库会显式装 stdout handler。这里在桥启动时
    重建 root handler，确保没有任何日志能落到 stdout。
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(_resolve_level(level))


def get_logger(suffix: str | None = None) -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME if not suffix else f"{_LOGGER_NAME}.{suffix}")
