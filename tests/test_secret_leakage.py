"""秘密不得离开桥进程——覆盖每一条对外输出链路。

这些用例里的「秘密」全部是本地构造的假串，不是任何真实凭据。

覆盖面（缺一条就等于留了一个出口）：

* JSON-RPC 响应体：``session/prompt``、``session/load`` 的 error data
* stderr 日志：``logger.error(..., exc_info=...)`` 会把整条 traceback 连同
  异常消息写出去，而 stderr 是客户端能看到的
* 客户端可见事件：custom task 的 ``error``、``llm_retry.reason``、
  ``safety_termination.reason`` 都会原样变成 tool_call / agent_message 文本
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from acp.exceptions import RequestError
from acp.helpers import text_block

from conftest import FakeBackend, RecordingConnection

from deerflow_acp.agent import DeerFlowAgent
from deerflow_acp.backend import BackendUnavailableError
from deerflow_acp.config import BridgeConfig
from deerflow_acp.events import EventNormalizer

# 构造的假秘密：形态逼真但完全无效
FAKE_KEY = "sk-proj-Ab3xQ9zK7mNpR2vT5wY8cE1dF4gH6jL0oP"
FAKE_DSN = "postgres://dfuser:Sup3rS3cretPw@127.0.0.1:5432/deerflow"
FAKE_HEADER = "Authorization: Bearer Ab3xQ9zK7mNpR2vT5wY8cE1d"


def _flatten(value: Any) -> str:
    """把任意结构压成一个字符串，用于「秘密是否出现在任何角落」的断言。"""
    return repr(value)


def make_agent(backend: FakeBackend, **cfg: Any) -> tuple[DeerFlowAgent, RecordingConnection]:
    conn = RecordingConnection()
    return DeerFlowAgent(conn, config=BridgeConfig(**cfg), backend=backend), conn


async def run_prompt(agent: DeerFlowAgent, backend: FakeBackend) -> Any:
    resp = await agent.new_session("/tmp")
    return await agent.prompt([text_block("问题")], resp.session_id)


# ----------------------------------------------------------------------
# 1) JSON-RPC 响应体
# ----------------------------------------------------------------------


async def test_backend_unavailable_detail_does_not_leak_secret():
    """专用错误链路：BackendUnavailableError 的消息常常带着上游原文。"""
    backend = FakeBackend(raise_on_stream=BackendUnavailableError(f"连接失败 {FAKE_DSN}"))
    agent, _conn = make_agent(backend)

    with pytest.raises(RequestError) as excinfo:
        await run_prompt(agent, backend)

    # 断言整个上线的 error 对象，而不只是 data——message 同样会发给客户端
    blob = _flatten(excinfo.value.to_error_obj())
    assert "Sup3rS3cretPw" not in blob
    assert "后端" in blob


async def test_generic_backend_error_response_does_not_leak_secret():
    backend = FakeBackend(raise_on_stream=RuntimeError(f"provider 拒绝：{FAKE_KEY}"))
    agent, _conn = make_agent(backend)

    with pytest.raises(RequestError) as excinfo:
        await run_prompt(agent, backend)

    blob = _flatten(excinfo.value.to_error_obj())
    assert FAKE_KEY not in blob
    # 分类必须保留，否则错误不可诊断
    assert "RuntimeError" in blob


async def test_session_load_error_detail_does_not_leak_secret():
    backend = FakeBackend(thread_lookup_error=BackendUnavailableError(f"checkpointer 打不开 {FAKE_DSN}"))
    agent, _conn = make_agent(backend)

    with pytest.raises(RequestError) as excinfo:
        await agent.load_session("/tmp", "df-abc")

    assert "Sup3rS3cretPw" not in _flatten(excinfo.value.to_error_obj())


# ----------------------------------------------------------------------
# 2) stderr 日志
# ----------------------------------------------------------------------


async def test_turn_failure_log_does_not_leak_secret_to_stderr(caplog):
    """exc_info 会把异常消息连同 traceback 写进 stderr，必须先脱敏。"""
    backend = FakeBackend(raise_on_stream=RuntimeError(f"provider 拒绝：{FAKE_KEY} / {FAKE_HEADER}"))
    agent, _conn = make_agent(backend)

    with caplog.at_level(logging.DEBUG, logger="deerflow_acp"):
        with pytest.raises(RequestError):
            await run_prompt(agent, backend)

    emitted = "\n".join(record.getMessage() for record in caplog.records)
    formatter = logging.Formatter("%(message)s")
    emitted += "\n" + "\n".join(formatter.format(record) for record in caplog.records)

    assert FAKE_KEY not in emitted
    assert "Ab3xQ9zK7mNpR2vT5wY8cE1d" not in emitted
    # 仍要看得出「哪一类错误」
    assert "RuntimeError" in emitted


# ----------------------------------------------------------------------
# 3) 客户端可见事件
# ----------------------------------------------------------------------


def _texts(updates: list[Any]) -> str:
    return _flatten(updates)


def test_custom_task_error_is_redacted():
    normalizer = EventNormalizer()
    normalizer.normalize("custom", {"type": "task_started", "task_id": "t1", "name": "搜索"})
    updates = normalizer.normalize(
        "custom",
        {"type": "task_failed", "task_id": "t1", "error": f"调用失败：{FAKE_KEY}"},
    )
    assert FAKE_KEY not in _texts(updates)
    assert updates, "脱敏不得把事件整个吞掉"


def test_llm_retry_reason_is_redacted():
    normalizer = EventNormalizer()
    updates = normalizer.normalize(
        "custom",
        {
            "type": "llm_retry",
            "attempt": 2,
            "max_attempts": 3,
            "reason": f"401 unauthorized，{FAKE_HEADER}",
            "wait_ms": 500,
        },
    )
    blob = _texts(updates)
    assert "Ab3xQ9zK7mNpR2vT5wY8cE1d" not in blob
    # 重试次数是有用的诊断信息，不能一起抹掉
    assert "2" in blob and "3" in blob


def test_safety_termination_reason_is_redacted():
    normalizer = EventNormalizer()
    updates = normalizer.normalize(
        "custom",
        {"type": "safety_termination", "reason": f"策略命中：{FAKE_DSN}"},
    )
    blob = _texts(updates)
    assert "Sup3rS3cretPw" not in blob
    assert "127.0.0.1:5432" in blob, "主机名是诊断信息，不应被一并抹掉"


def test_normal_event_text_is_untouched():
    """脱敏层不能改写正常的模型输出，否则会污染回答内容。"""
    normalizer = EventNormalizer()
    updates = normalizer.normalize(
        "custom",
        {"type": "llm_retry", "attempt": 1, "max_attempts": 3, "reason": "连接 127.0.0.1:8080 超时"},
    )
    assert "127.0.0.1:8080 超时" in _texts(updates)


