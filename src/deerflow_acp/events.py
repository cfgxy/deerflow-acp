"""DeerFlow 流事件 → ACP ``session/update`` 的归一化。

本模块是纯逻辑层：输入是 DeerFlow ``StreamEvent`` 的 ``(type, data)`` 结构
（普通 dict 即可，不依赖 DeerFlow 运行时），输出是 ACP SDK 的 SessionUpdate
对象列表，便于在没有真实后端的情况下做单元测试。

映射关系见 ``docs/compatibility.md``；无法无损映射的能力一律显式降级，
不伪造数据。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from acp.helpers import (
    start_tool_call,
    text_block,
    tool_content,
    update_agent_message_text,
    update_agent_thought_text,
    update_tool_call,
)
from acp.schema import AgentMessageChunk, AgentThoughtChunk, ToolCallStart, ToolCallProgress, ToolKind

from .logging_setup import get_logger
from .sanitize import redact_text

logger = get_logger("events")

SessionUpdate = AgentMessageChunk | AgentThoughtChunk | ToolCallStart | ToolCallProgress

# DeerFlow 内建工具名 → ACP ToolKind。未命中的工具统一落到 "other"，
# 不猜测语义。
_TOOL_KIND_BY_NAME: dict[str, ToolKind] = {
    "read_file": "read",
    "read": "read",
    "view": "read",
    "write_file": "edit",
    "write": "edit",
    "edit_file": "edit",
    "edit": "edit",
    "str_replace": "edit",
    "delete_file": "delete",
    "move_file": "move",
    "glob": "search",
    "grep": "search",
    "search": "search",
    "web_search": "fetch",
    "crawl": "fetch",
    "crawl_tool": "fetch",
    "fetch": "fetch",
    "bash": "execute",
    "shell": "execute",
    "python_repl": "execute",
    "task": "think",
    "think": "think",
    "ask_clarification": "other",
}

# DeerFlow custom 事件中代表「子任务失败」的类型。
_TASK_FAILURE_TYPES = frozenset({"task_failed", "task_timed_out"})
_TASK_PROGRESS_TYPES = frozenset({"task_started", "task_running"})


def tool_kind_for(name: str | None) -> ToolKind:
    if not name:
        return "other"
    return _TOOL_KIND_BY_NAME.get(name.strip().lower(), "other")


@dataclass
class TurnUsage:
    """一次 turn 的累计 token 用量（来自 DeerFlow ``end`` 事件）。"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_payload(cls, payload: Any) -> TurnUsage | None:
        if not isinstance(payload, dict):
            return None
        try:
            return cls(
                input_tokens=max(0, int(payload.get("input_tokens") or 0)),
                output_tokens=max(0, int(payload.get("output_tokens") or 0)),
                total_tokens=max(0, int(payload.get("total_tokens") or 0)),
            )
        except (TypeError, ValueError):
            logger.warning("无法解析 DeerFlow usage 负载，已忽略")
            return None


@dataclass
class EventNormalizer:
    """把一次 turn 内的 DeerFlow 事件流归一化为 ACP session updates。

    归一化器是有状态的（按 turn 创建一次）：需要记住已声明的 tool call、
    已下发的 reasoning 前缀，才能把 DeerFlow 的增量/累计混合语义收敛成
    ACP 要求的单调事件流。
    """

    #: 已经发过 ``tool_call``（start）的 tool_call_id → 标题
    _announced_tools: dict[str, str] = field(default_factory=dict)
    #: message_id → 已下发的 reasoning 文本，用于把累计值转成增量
    _reasoning_sent: dict[str, str] = field(default_factory=dict)
    #: 本 turn 的最终 usage
    usage: TurnUsage | None = None
    #: 已产生过任何 agent 可见输出（用于判定空 turn）
    produced_output: bool = False

    def normalize(self, event_type: str, data: dict[str, Any]) -> list[SessionUpdate]:
        """归一化单个 DeerFlow 事件，返回 0..n 条 ACP session update。"""
        if event_type == "messages-tuple":
            updates = self._normalize_message(data)
        elif event_type == "custom":
            updates = self._normalize_custom(data)
        elif event_type == "end":
            self.usage = TurnUsage.from_payload(data.get("usage"))
            updates = []
        elif event_type == "values":
            # values 是状态快照，其中的消息已由 messages-tuple 逐条下发过
            # （DeerFlow 内部用 seen_ids/streamed_ids 去重），重复下发会
            # 导致客户端出现重复文本，因此整体丢弃。
            updates = []
        else:
            logger.debug("忽略未知 DeerFlow 事件类型：%s", event_type)
            updates = []

        if updates:
            self.produced_output = True
        return updates

    # ------------------------------------------------------------------
    # messages-tuple
    # ------------------------------------------------------------------

    def _normalize_message(self, data: dict[str, Any]) -> list[SessionUpdate]:
        kind = data.get("type")
        if kind == "ai":
            return self._normalize_ai_message(data)
        if kind == "tool":
            return self._normalize_tool_result(data)
        logger.debug("忽略未知 messages-tuple 子类型：%s", kind)
        return []

    def _normalize_ai_message(self, data: dict[str, Any]) -> list[SessionUpdate]:
        updates: list[SessionUpdate] = []
        message_id = data.get("id") or ""

        # thought 先于正文：推理内容在时间上总是发生在最终答案之前。
        reasoning_delta = self._reasoning_delta(message_id, data.get("additional_kwargs"))
        if reasoning_delta:
            updates.append(update_agent_thought_text(reasoning_delta))

        content = data.get("content")
        if isinstance(content, str) and content:
            updates.append(update_agent_message_text(content))

        for call in data.get("tool_calls") or []:
            update = self._tool_call_start(call)
            if update is not None:
                updates.append(update)

        return updates

    def _reasoning_delta(self, message_id: str, additional_kwargs: Any) -> str:
        """从 ``additional_kwargs`` 中取出本次新增的推理文本。

        不同 provider 的行为不一致：有的每个 chunk 给增量，有的给累计值。
        这里以「已下发前缀」为准做统一：新值是旧值的前缀扩展就取后缀，
        否则视为独立片段整体下发，两种情况都不会丢内容。
        """
        if not isinstance(additional_kwargs, dict):
            return ""
        raw = additional_kwargs.get("reasoning_content")
        if not isinstance(raw, str) or not raw:
            return ""

        sent = self._reasoning_sent.get(message_id, "")
        if raw == sent:
            return ""
        if sent and raw.startswith(sent):
            delta = raw[len(sent) :]
            self._reasoning_sent[message_id] = raw
            return delta
        self._reasoning_sent[message_id] = sent + raw
        return raw

    def _tool_call_start(self, call: Any) -> ToolCallStart | None:
        if not isinstance(call, dict):
            return None
        tool_call_id = call.get("id")
        if not tool_call_id:
            # 没有 id 就无法在后续 tool_call_update 中定位，宁可丢弃也不编造。
            logger.warning("DeerFlow tool_call 缺少 id，已跳过：%s", call.get("name"))
            return None
        name = call.get("name") or "tool"
        if tool_call_id in self._announced_tools:
            return None
        self._announced_tools[tool_call_id] = name
        return start_tool_call(
            tool_call_id,
            name,
            kind=tool_kind_for(name),
            status="in_progress",
            raw_input=call.get("args") if isinstance(call.get("args"), dict) else None,
        )

    def _normalize_tool_result(self, data: dict[str, Any]) -> list[SessionUpdate]:
        tool_call_id = data.get("tool_call_id")
        if not tool_call_id:
            logger.warning("DeerFlow tool 结果缺少 tool_call_id，已跳过")
            return []

        name = data.get("name") or self._announced_tools.get(tool_call_id) or "tool"
        updates: list[SessionUpdate] = []
        if tool_call_id not in self._announced_tools:
            # 恢复的会话里可能只看到结果而没看到声明，补一条 start，
            # 避免客户端收到孤儿 tool_call_update。
            self._announced_tools[tool_call_id] = name
            updates.append(start_tool_call(tool_call_id, name, kind=tool_kind_for(name), status="in_progress"))

        text = data.get("content")
        content = [tool_content(text_block(text))] if isinstance(text, str) and text else None
        raw_output = data.get("artifact")
        updates.append(
            update_tool_call(
                tool_call_id,
                status="completed",
                content=content,
                raw_output=raw_output if isinstance(raw_output, dict) else None,
            )
        )
        return updates

    # ------------------------------------------------------------------
    # custom
    # ------------------------------------------------------------------

    def _normalize_custom(self, data: dict[str, Any]) -> list[SessionUpdate]:
        event_kind = data.get("type")

        if event_kind in _TASK_PROGRESS_TYPES:
            return self._task_progress(data)
        if event_kind in _TASK_FAILURE_TYPES:
            return self._task_terminal(data, status="failed")
        if event_kind == "task_cancelled":
            return self._task_terminal(data, status="failed")
        if event_kind == "task_completed":
            # 子任务的正式结果会以 ToolMessage 形式到达，这里不重复终结，
            # 只在缺少 tool 结果时由 tool_call_update 兜底。
            return []
        if event_kind == "llm_retry":
            return [update_agent_thought_text(_format_retry_notice(data))]
        if event_kind == "safety_termination":
            return [update_agent_thought_text(_format_safety_notice(data))]

        logger.debug("忽略未映射的 DeerFlow custom 事件：%s", event_kind)
        return []

    def _task_progress(self, data: dict[str, Any]) -> list[SessionUpdate]:
        task_id = data.get("task_id")
        if not task_id:
            return []
        title = data.get("description") or self._announced_tools.get(task_id) or "task"
        if task_id not in self._announced_tools:
            self._announced_tools[task_id] = title
            return [start_tool_call(task_id, title, kind="think", status="in_progress")]
        return [update_tool_call(task_id, status="in_progress")]

    def _task_terminal(self, data: dict[str, Any], *, status: str) -> list[SessionUpdate]:
        task_id = data.get("task_id")
        if not task_id:
            return []
        error = data.get("error")
        # 后端把上游报错原文塞进 error 是常态，先脱敏再交给客户端。
        content = [tool_content(text_block(redact_text(error)))] if error else None
        updates: list[SessionUpdate] = []
        if task_id not in self._announced_tools:
            self._announced_tools[task_id] = "task"
            updates.append(start_tool_call(task_id, "task", kind="think", status="in_progress"))
        updates.append(update_tool_call(task_id, status=status, content=content))
        return updates


def _format_retry_notice(data: dict[str, Any]) -> str:
    attempt = data.get("attempt")
    max_attempts = data.get("max_attempts")
    reason = redact_text(data.get("reason")) or "未提供原因"
    wait_ms = data.get("wait_ms")
    parts = ["[DeerFlow] 模型调用重试"]
    if attempt is not None and max_attempts is not None:
        parts.append(f"（第 {attempt}/{max_attempts} 次）")
    parts.append(f"：{reason}")
    if wait_ms is not None:
        parts.append(f"，等待 {wait_ms}ms")
    return "".join(parts)


def _format_safety_notice(data: dict[str, Any]) -> str:
    reason = redact_text(data.get("reason") or data.get("finish_reason")) or "未提供原因"
    suppressed = data.get("suppressed_tool_calls") or data.get("suppressed_names")
    text = f"[DeerFlow] 安全策略提前终止本轮工具调用：{reason}"
    if isinstance(suppressed, Iterable) and not isinstance(suppressed, (str, bytes)):
        names = [str(item) for item in suppressed]
        if names:
            text += f"（已抑制：{', '.join(names)}）"
    return text
