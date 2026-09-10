"""事件归一化单元测试。"""

from __future__ import annotations

import pytest

from deerflow_acp.events import EventNormalizer, TurnUsage, tool_kind_for


def kinds(updates):
    return [u.session_update for u in updates]


def test_ai_text_becomes_agent_message_chunk():
    n = EventNormalizer()
    updates = n.normalize("messages-tuple", {"type": "ai", "content": "你好", "id": "m1"})
    assert kinds(updates) == ["agent_message_chunk"]
    assert updates[0].content.text == "你好"


def test_empty_ai_content_produces_nothing():
    n = EventNormalizer()
    assert n.normalize("messages-tuple", {"type": "ai", "content": "", "id": "m1"}) == []


def test_reasoning_content_becomes_thought_before_text():
    n = EventNormalizer()
    updates = n.normalize(
        "messages-tuple",
        {"type": "ai", "content": "答案", "id": "m1", "additional_kwargs": {"reasoning_content": "先想一下"}},
    )
    assert kinds(updates) == ["agent_thought_chunk", "agent_message_chunk"]
    assert updates[0].content.text == "先想一下"


def test_cumulative_reasoning_is_deduped_into_deltas():
    """provider 给累计值时，只下发新增后缀，不重复整段。"""
    n = EventNormalizer()
    first = n.normalize("messages-tuple", {"type": "ai", "content": "", "id": "m1", "additional_kwargs": {"reasoning_content": "abc"}})
    second = n.normalize("messages-tuple", {"type": "ai", "content": "", "id": "m1", "additional_kwargs": {"reasoning_content": "abcdef"}})
    assert first[0].content.text == "abc"
    assert second[0].content.text == "def"


def test_identical_reasoning_repeat_is_dropped():
    n = EventNormalizer()
    n.normalize("messages-tuple", {"type": "ai", "content": "", "id": "m1", "additional_kwargs": {"reasoning_content": "abc"}})
    assert n.normalize("messages-tuple", {"type": "ai", "content": "", "id": "m1", "additional_kwargs": {"reasoning_content": "abc"}}) == []


def test_incremental_reasoning_from_different_ids_is_independent():
    n = EventNormalizer()
    a = n.normalize("messages-tuple", {"type": "ai", "content": "", "id": "m1", "additional_kwargs": {"reasoning_content": "aa"}})
    b = n.normalize("messages-tuple", {"type": "ai", "content": "", "id": "m2", "additional_kwargs": {"reasoning_content": "aa"}})
    assert a[0].content.text == "aa"
    assert b[0].content.text == "aa"


def test_tool_call_then_result_produces_start_and_completed():
    n = EventNormalizer()
    start = n.normalize(
        "messages-tuple",
        {"type": "ai", "content": "", "id": "m1", "tool_calls": [{"name": "web_search", "args": {"q": "x"}, "id": "t1"}]},
    )
    assert kinds(start) == ["tool_call"]
    assert start[0].tool_call_id == "t1"
    assert start[0].kind == "fetch"
    assert start[0].status == "in_progress"
    assert start[0].raw_input == {"q": "x"}

    done = n.normalize("messages-tuple", {"type": "tool", "content": "结果", "name": "web_search", "tool_call_id": "t1", "id": "m2"})
    assert kinds(done) == ["tool_call_update"]
    assert done[0].status == "completed"
    assert done[0].content[0].content.text == "结果"


def test_duplicate_tool_call_declaration_is_emitted_once():
    n = EventNormalizer()
    call = {"type": "ai", "content": "", "id": "m1", "tool_calls": [{"name": "bash", "args": {}, "id": "t1"}]}
    assert kinds(n.normalize("messages-tuple", call)) == ["tool_call"]
    assert n.normalize("messages-tuple", call) == []


def test_orphan_tool_result_synthesizes_start():
    """恢复的会话里可能只看到结果，桥必须补 start，避免孤儿 update。"""
    n = EventNormalizer()
    updates = n.normalize("messages-tuple", {"type": "tool", "content": "r", "name": "grep", "tool_call_id": "t9", "id": "m1"})
    assert kinds(updates) == ["tool_call", "tool_call_update"]
    assert updates[0].kind == "search"


def test_tool_call_without_id_is_dropped_not_faked():
    n = EventNormalizer()
    assert n.normalize("messages-tuple", {"type": "ai", "content": "", "id": "m1", "tool_calls": [{"name": "bash", "args": {}}]}) == []


def test_values_snapshot_is_dropped_to_avoid_duplication():
    n = EventNormalizer()
    assert n.normalize("values", {"messages": [{"type": "ai", "content": "重复", "id": "m1"}], "title": "t"}) == []


def test_end_event_captures_usage():
    n = EventNormalizer()
    assert n.normalize("end", {"usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}}) == []
    assert n.usage == TurnUsage(3, 4, 7)


def test_malformed_usage_is_ignored_not_faked():
    n = EventNormalizer()
    n.normalize("end", {"usage": {"input_tokens": "abc"}})
    assert n.usage is None


def test_custom_task_lifecycle_maps_to_tool_call():
    n = EventNormalizer()
    started = n.normalize("custom", {"type": "task_started", "task_id": "k1", "description": "查资料"})
    assert kinds(started) == ["tool_call"]
    assert started[0].title == "查资料"

    running = n.normalize("custom", {"type": "task_running", "task_id": "k1"})
    assert kinds(running) == ["tool_call_update"]
    assert running[0].status == "in_progress"

    failed = n.normalize("custom", {"type": "task_failed", "task_id": "k1", "error": "boom"})
    assert kinds(failed) == ["tool_call_update"]
    assert failed[0].status == "failed"
    assert failed[0].content[0].content.text == "boom"


def test_custom_task_timeout_is_failure():
    n = EventNormalizer()
    n.normalize("custom", {"type": "task_started", "task_id": "k1", "description": "d"})
    out = n.normalize("custom", {"type": "task_timed_out", "task_id": "k1", "error": "timeout"})
    assert out[0].status == "failed"


def test_custom_task_completed_defers_to_tool_message():
    n = EventNormalizer()
    n.normalize("custom", {"type": "task_started", "task_id": "k1", "description": "d"})
    assert n.normalize("custom", {"type": "task_completed", "task_id": "k1"}) == []


def test_llm_retry_becomes_thought():
    n = EventNormalizer()
    out = n.normalize("custom", {"type": "llm_retry", "attempt": 2, "max_attempts": 3, "reason": "429", "wait_ms": 500})
    assert kinds(out) == ["agent_thought_chunk"]
    assert "2/3" in out[0].content.text
    assert "429" in out[0].content.text


def test_safety_termination_becomes_thought():
    n = EventNormalizer()
    out = n.normalize("custom", {"type": "safety_termination", "reason": "policy", "suppressed_names": ["bash"]})
    assert kinds(out) == ["agent_thought_chunk"]
    assert "bash" in out[0].content.text


def test_unknown_custom_event_is_dropped():
    n = EventNormalizer()
    assert n.normalize("custom", {"type": "brand_new_thing"}) == []


def test_unknown_event_type_is_dropped():
    n = EventNormalizer()
    assert n.normalize("totally_unknown", {}) == []


@pytest.mark.parametrize(
    ("name", "expected"),
    [("read_file", "read"), ("Edit", "edit"), ("web_search", "fetch"), ("bash", "execute"), ("nope", "other"), (None, "other")],
)
def test_tool_kind_mapping(name, expected):
    assert tool_kind_for(name) == expected
