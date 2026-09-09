"""IPC 消息层：编解码鲁棒性与「异常消息不过河」的结构性保证。"""

from __future__ import annotations

import datetime as dt

from deerflow_acp import ipc


def test_roundtrip_event():
    line = ipc.encode(ipc.event_message("messages-tuple", {"type": "ai", "content": "你好"}))
    parsed = ipc.decode(line)
    assert parsed == {"t": "ev", "type": "messages-tuple", "data": {"type": "ai", "content": "你好"}}


def test_encode_ends_with_single_newline():
    """ndJSON 的分帧完全依赖行尾；多一个或少一个换行都会撕裂消息流。"""
    line = ipc.encode({"t": "done"})
    assert line.endswith(b"\n")
    assert line.count(b"\n") == 1


def test_encode_survives_unserializable_payload():
    """事件 data 里出现不可 JSON 化的对象时，不得让整条流断掉。"""
    line = ipc.encode(ipc.event_message("custom", {"ts": dt.datetime(2026, 9, 10, 12, 0, 0)}))
    parsed = ipc.decode(line)
    assert parsed is not None
    assert "2026-09-10" in parsed["data"]["ts"]


def test_encode_preserves_non_ascii():
    parsed = ipc.decode(ipc.encode({"t": "ev", "type": "x", "data": {"content": "紫罗兰七号"}}))
    assert parsed["data"]["content"] == "紫罗兰七号"


def test_decode_skips_blank_and_malformed():
    assert ipc.decode(b"") is None
    assert ipc.decode(b"   \n") is None
    assert ipc.decode("{ 这不是 JSON\n".encode("utf-8")) is None
    # 合法 JSON 但不是对象，同样必须拒绝——否则下游 .get() 会炸
    assert ipc.decode(b"[1,2,3]\n") is None
    assert ipc.decode(b'"str"\n') is None


def test_decode_tolerates_invalid_utf8():
    """worker 的 stderr 若意外混入，也不能让父进程崩掉。"""
    assert ipc.decode(b"\xff\xfe not json\n") is None


def test_error_message_carries_type_name_only():
    """核心安全断言：异常消息、args、cause 一律不进 payload。"""
    secret = "sk-proj-Ab3xQ9zK7mNpR2vT5wY8cE1dF4gH6jL0oP"
    try:
        try:
            raise ValueError(f"底层原因 {secret}")
        except ValueError as cause:
            raise RuntimeError(f"provider 拒绝：{secret}") from cause
    except RuntimeError as exc:
        message = ipc.error_message(exc)

    assert message == {"t": "err", "cls": "RuntimeError"}
    # 不是靠脱敏正则，而是这个字段压根不存在
    assert secret not in ipc.encode(message).decode("utf-8")


def test_error_message_for_backend_unavailable_keeps_class():
    """后端不可用要还原成专用错误码，类型名必须原样过河。"""
    from deerflow_acp.backend import BackendUnavailableError

    assert ipc.error_message(BackendUnavailableError("带着配置路径 /home/u/.env"))["cls"] == "BackendUnavailableError"
