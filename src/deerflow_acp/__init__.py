"""deerflow-acp：DeerFlow 与 ACP 客户端之间的独立桥接器。

本包只负责 ACP 方法实现、事件归一化、会话恢复、取消与进程生命周期，
不复制 DeerFlow 的研究编排能力。
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
