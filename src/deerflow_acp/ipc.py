"""桥主进程与 DeerFlow worker 子进程之间的 ndJSON 消息协议。

只有四种消息，父子两侧共用这一份定义，避免两边各写一套后悄悄漂移。

**安全约束（结构性，不依赖正则兜底）**：错误消息跨进程时只传异常**类型名**，
不传 ``str(exc)``、``args``、``__cause__`` 与 traceback。DeerFlow、provider SDK
与工具都可能把凭据回显进异常消息，让消息根本不进入 IPC payload，比事后脱敏
可靠——脱敏是启发式的，类型名是封闭集合。

传输载体是 worker 子进程的 stdout；worker 启动时会把 fd 1 抢占为 IPC 专用
（见 :mod:`deerflow_acp.worker`），因此 DeerFlow 往 stdout 打的任何东西都落在
stderr，物理上不可能混进消息流。
"""

from __future__ import annotations

import json
from typing import Any

#: worker 已构造好后端，即将开始迭代
MSG_READY = "ready"
#: 一条 DeerFlow 事件
MSG_EVENT = "ev"
#: 流正常耗尽
MSG_DONE = "done"
#: 后端抛异常；只携带类型名
MSG_ERROR = "err"


def encode(message: dict[str, Any]) -> bytes:
    """把一条消息编成一行 ndJSON。

    ``default=str`` 是必要的兜底：DeerFlow 事件 data 里可能出现 datetime、
    Decimal 或自定义对象，不能因为某一个字段不可序列化就让整条流断掉。
    """
    return (json.dumps(message, ensure_ascii=False, default=str) + "\n").encode("utf-8")


def decode(line: bytes | str) -> dict[str, Any] | None:
    """解一行 ndJSON；空行与畸形行返回 None（调用方跳过即可）。"""
    text = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def event_message(event_type: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"t": MSG_EVENT, "type": event_type, "data": data}


def error_message(exc: BaseException) -> dict[str, Any]:
    """把异常压成只有类型名的消息——消息体一律不过河。"""
    return {"t": MSG_ERROR, "cls": type(exc).__name__}


__all__ = [
    "MSG_DONE",
    "MSG_ERROR",
    "MSG_EVENT",
    "MSG_READY",
    "decode",
    "encode",
    "error_message",
    "event_message",
]
