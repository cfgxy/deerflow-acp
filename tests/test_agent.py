"""ACP Agent 方法层测试。"""

from __future__ import annotations

import threading

import pytest
from acp import PROTOCOL_VERSION
from acp.exceptions import RequestError
from acp.helpers import text_block
from acp.schema import ImageContentBlock

from conftest import FakeBackend, RecordingConnection

from deerflow_acp.agent import (
    ERROR_BACKEND_UNAVAILABLE,
    ERROR_TURN_IN_PROGRESS,
    ERROR_UNKNOWN_SESSION,
    DeerFlowAgent,
)
from deerflow_acp.backend import BackendUnavailableError
from deerflow_acp.config import BridgeConfig


def make_agent(backend: FakeBackend, connection: RecordingConnection | None = None, **cfg):
    conn = connection or RecordingConnection()
    agent = DeerFlowAgent(conn, config=BridgeConfig(**cfg), backend=backend)
    return agent, conn


# ----------------------------------------------------------------------
# initialize
# ----------------------------------------------------------------------


async def test_initialize_declares_capabilities_without_touching_backend():
    class ExplodingBackend(FakeBackend):
        def thread_exists(self, thread_id):  # pragma: no cover - 不应被调用
            raise AssertionError("initialize 阶段不得访问 DeerFlow 后端")

        def stream(self, message, *, thread_id):  # pragma: no cover
            raise AssertionError("initialize 阶段不得访问 DeerFlow 后端")

    agent, _ = make_agent(ExplodingBackend())
    resp = await agent.initialize(PROTOCOL_VERSION)

    assert resp.protocol_version == PROTOCOL_VERSION
    assert resp.agent_info.name == "deerflow-acp"
    assert resp.agent_capabilities.load_session is True
    # 显式降级：不接受客户端 MCP，不支持非文本内容块
    assert resp.agent_capabilities.mcp_capabilities.http is False
    assert resp.agent_capabilities.mcp_capabilities.sse is False
    assert resp.agent_capabilities.prompt_capabilities.image is False
    assert resp.agent_capabilities.prompt_capabilities.audio is False
    assert resp.agent_capabilities.prompt_capabilities.embedded_context is False
    # 不声明任何认证方法
    assert not resp.auth_methods


async def test_initialize_negotiates_down_to_client_version():
    agent, _ = make_agent(FakeBackend())
    resp = await agent.initialize(0)
    assert resp.protocol_version == 0


async def test_initialize_caps_at_agent_supported_version():
    agent, _ = make_agent(FakeBackend())
    resp = await agent.initialize(999)
    assert resp.protocol_version == PROTOCOL_VERSION


async def test_authenticate_is_explicitly_unsupported():
    agent, _ = make_agent(FakeBackend())
    with pytest.raises(RequestError) as exc:
        await agent.authenticate("anything")
    assert exc.value.code == -32601


# ----------------------------------------------------------------------
# session 生命周期
# ----------------------------------------------------------------------


async def test_new_session_returns_valid_session_id():
    from deerflow_acp.session import validate_session_id

    agent, _ = make_agent(FakeBackend())
    resp = await agent.new_session("/tmp")
    assert validate_session_id(resp.session_id)


async def test_new_session_rejects_client_supplied_mcp_servers():
    agent, _ = make_agent(FakeBackend())
    with pytest.raises(RequestError) as exc:
        await agent.new_session("/tmp", mcp_servers=[{"name": "x"}])
    assert exc.value.code == -32602


async def test_load_session_unknown_id_errors_and_does_not_create():
    agent, _ = make_agent(FakeBackend(threads={}))
    with pytest.raises(RequestError) as exc:
        await agent.load_session("/tmp", "df-missing")
    assert exc.value.code == ERROR_UNKNOWN_SESSION

    # 未知 ID 被拒后，prompt 也必须继续拒绝，不能悄悄有了会话
    with pytest.raises(RequestError) as prompt_exc:
        await agent.prompt([text_block("hi")], "df-missing")
    assert prompt_exc.value.code == ERROR_UNKNOWN_SESSION


async def test_resume_session_unknown_id_errors():
    agent, _ = make_agent(FakeBackend(threads={}))
    with pytest.raises(RequestError) as exc:
        await agent.resume_session("/tmp", "df-missing")
    assert exc.value.code == ERROR_UNKNOWN_SESSION


async def test_resume_session_malformed_id_is_invalid_params():
    agent, _ = make_agent(FakeBackend(threads={}))
    with pytest.raises(RequestError) as exc:
        await agent.resume_session("/tmp", "非法 id/带斜杠")
    assert exc.value.code == -32602


async def test_resume_session_backend_failure_is_distinct_error_code():
    agent, _ = make_agent(FakeBackend(thread_lookup_error=BackendUnavailableError("db down")))
    with pytest.raises(RequestError) as exc:
        await agent.resume_session("/tmp", "df-known")
    assert exc.value.code == ERROR_BACKEND_UNAVAILABLE


async def test_load_session_replays_history():
    backend = FakeBackend(
        threads={
            "df-known": [
                {"type": "human", "content": "问题"},
                {"type": "ai", "content": "回答"},
                {"type": "tool", "content": "工具输出", "tool_call_id": "t1"},
            ]
        }
    )
    agent, conn = make_agent(backend)
    await agent.load_session("/tmp", "df-known")

    assert conn.update_kinds() == ["user_message_chunk", "agent_message_chunk"]
    assert conn.texts("user_message_chunk") == ["问题"]
    assert conn.texts("agent_message_chunk") == ["回答"]


async def test_resume_session_does_not_replay_history():
    backend = FakeBackend(threads={"df-known": [{"type": "ai", "content": "回答"}]})
    agent, conn = make_agent(backend)
    await agent.resume_session("/tmp", "df-known")
    assert conn.updates == []


async def test_resumed_session_reuses_same_thread_id_for_prompt():
    backend = FakeBackend(
        [("end", {"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}})],
        threads={"df-known": [{"type": "ai", "content": "旧回答"}]},
    )
    agent, _ = make_agent(backend)
    await agent.resume_session("/tmp", "df-known")
    await agent.prompt([text_block("继续")], "df-known")
    assert backend.stream_calls == [("继续", "df-known")]


async def test_close_session_removes_it():
    agent, _ = make_agent(FakeBackend())
    resp = await agent.new_session("/tmp")
    await agent.close_session(resp.session_id)
    with pytest.raises(RequestError) as exc:
        await agent.prompt([text_block("hi")], resp.session_id)
    assert exc.value.code == ERROR_UNKNOWN_SESSION


# ----------------------------------------------------------------------
# prompt
# ----------------------------------------------------------------------


async def test_prompt_streams_updates_and_returns_usage():
    backend = FakeBackend(
        [
            ("messages-tuple", {"type": "ai", "content": "思考中", "id": "m1", "additional_kwargs": {"reasoning_content": "推理"}}),
            ("messages-tuple", {"type": "ai", "content": "", "id": "m1", "tool_calls": [{"name": "web_search", "args": {}, "id": "t1"}]}),
            ("messages-tuple", {"type": "tool", "content": "命中", "name": "web_search", "tool_call_id": "t1", "id": "m2"}),
            ("end", {"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}),
        ]
    )
    agent, conn = make_agent(backend)
    session_id = (await agent.new_session("/tmp")).session_id
    resp = await agent.prompt([text_block("查一下")], session_id)

    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 10
    assert resp.usage.total_tokens == 15
    assert conn.update_kinds() == [
        "agent_thought_chunk",
        "agent_message_chunk",
        "tool_call",
        "tool_call_update",
    ]
    assert all(sid == session_id for sid, _ in conn.updates)


async def test_prompt_concatenates_multiple_text_blocks():
    backend = FakeBackend([("end", {"usage": {}})])
    agent, _ = make_agent(backend)
    session_id = (await agent.new_session("/tmp")).session_id
    await agent.prompt([text_block("第一段"), text_block("第二段")], session_id)
    assert backend.stream_calls[0][0] == "第一段\n\n第二段"


async def test_prompt_rejects_non_text_blocks_instead_of_dropping():
    agent, _ = make_agent(FakeBackend())
    session_id = (await agent.new_session("/tmp")).session_id
    image = ImageContentBlock(type="image", data="aGk=", mime_type="image/png")
    with pytest.raises(RequestError) as exc:
        await agent.prompt([text_block("看图"), image], session_id)
    assert exc.value.code == -32602
    assert exc.value.data["unsupportedBlockTypes"] == ["image"]


async def test_prompt_rejects_empty_prompt():
    agent, _ = make_agent(FakeBackend())
    session_id = (await agent.new_session("/tmp")).session_id
    with pytest.raises(RequestError) as exc:
        await agent.prompt([], session_id)
    assert exc.value.code == -32602


async def test_prompt_unknown_session_errors():
    agent, _ = make_agent(FakeBackend())
    with pytest.raises(RequestError) as exc:
        await agent.prompt([text_block("hi")], "df-nope")
    assert exc.value.code == ERROR_UNKNOWN_SESSION


async def test_prompt_backend_error_becomes_internal_error_without_leaking_detail():
    backend = FakeBackend(raise_on_stream=RuntimeError("sk-secret-token 泄露风险"))
    agent, _ = make_agent(backend)
    session_id = (await agent.new_session("/tmp")).session_id
    with pytest.raises(RequestError) as exc:
        await agent.prompt([text_block("hi")], session_id)
    assert exc.value.code == -32603
    # 只暴露异常类型，不回显异常消息（可能含路径或凭据）
    assert exc.value.data == {"sessionId": session_id, "errorType": "RuntimeError"}


async def test_prompt_backend_unavailable_has_dedicated_code():
    backend = FakeBackend(raise_on_stream=BackendUnavailableError("DeerFlow 未安装"))
    agent, _ = make_agent(backend)
    session_id = (await agent.new_session("/tmp")).session_id
    with pytest.raises(RequestError) as exc:
        await agent.prompt([text_block("hi")], session_id)
    assert exc.value.code == ERROR_BACKEND_UNAVAILABLE


async def test_concurrent_prompt_on_same_session_is_rejected():
    import asyncio

    backend = FakeBackend([("messages-tuple", {"type": "ai", "content": "x", "id": "m"})])
    backend.gate = threading.Event()
    agent, _ = make_agent(backend)
    session_id = (await agent.new_session("/tmp")).session_id

    first = asyncio.create_task(agent.prompt([text_block("a")], session_id))
    await asyncio.sleep(0.05)
    with pytest.raises(RequestError) as exc:
        await agent.prompt([text_block("b")], session_id)
    assert exc.value.code == ERROR_TURN_IN_PROGRESS

    backend.gate.set()
    await first


# ----------------------------------------------------------------------
# 取消
# ----------------------------------------------------------------------


async def test_cancel_yields_cancelled_stop_reason():
    import asyncio

    events = [("messages-tuple", {"type": "ai", "content": str(i), "id": "m"}) for i in range(200)]
    backend = FakeBackend(events)
    agent, conn = make_agent(backend, cancel_grace_seconds=2.0)
    session_id = (await agent.new_session("/tmp")).session_id

    async def cancel_soon():
        while len(conn.updates) < 3:
            await asyncio.sleep(0.005)
        await agent.cancel(session_id)

    canceller = asyncio.create_task(cancel_soon())
    resp = await agent.prompt([text_block("跑一会")], session_id)
    await canceller

    assert resp.stop_reason == "cancelled"
    assert backend.closed.wait(timeout=2)
    assert len(conn.updates) < len(events)


async def test_cancel_unknown_session_is_silent():
    agent, _ = make_agent(FakeBackend())
    await agent.cancel("df-nope")


async def test_session_can_be_reused_after_cancel():
    import asyncio

    events = [("messages-tuple", {"type": "ai", "content": str(i), "id": "m"}) for i in range(200)]
    backend = FakeBackend(events)
    agent, conn = make_agent(backend, cancel_grace_seconds=2.0)
    session_id = (await agent.new_session("/tmp")).session_id

    async def cancel_soon():
        while len(conn.updates) < 2:
            await asyncio.sleep(0.005)
        await agent.cancel(session_id)

    canceller = asyncio.create_task(cancel_soon())
    first = await agent.prompt([text_block("a")], session_id)
    await canceller
    assert first.stop_reason == "cancelled"

    backend.events = [("end", {"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}})]
    second = await agent.prompt([text_block("b")], session_id)
    assert second.stop_reason == "end_turn"


# ----------------------------------------------------------------------
# usage 降级
# ----------------------------------------------------------------------


async def test_usage_update_is_not_emitted_by_default():
    backend = FakeBackend([("end", {"usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}})])
    agent, conn = make_agent(backend)
    session_id = (await agent.new_session("/tmp")).session_id
    await agent.prompt([text_block("hi")], session_id)
    assert "usage_update" not in conn.update_kinds()


async def test_usage_update_emitted_when_context_window_configured():
    backend = FakeBackend([("end", {"usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}})])
    agent, conn = make_agent(backend, emit_usage_update=True, context_window_tokens=1000)
    session_id = (await agent.new_session("/tmp")).session_id
    await agent.prompt([text_block("hi")], session_id)

    usage_updates = [u for _, u in conn.updates if getattr(u, "session_update", None) == "usage_update"]
    assert len(usage_updates) == 1
    assert usage_updates[0].size == 1000
    assert usage_updates[0].used == 3


async def test_usage_update_is_clamped_to_window():
    backend = FakeBackend([("end", {"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 99999}})])
    agent, conn = make_agent(backend, emit_usage_update=True, context_window_tokens=100)
    session_id = (await agent.new_session("/tmp")).session_id
    await agent.prompt([text_block("hi")], session_id)
    usage_updates = [u for _, u in conn.updates if getattr(u, "session_update", None) == "usage_update"]
    assert usage_updates[0].used == 100


# ----------------------------------------------------------------------
# 显式降级的方法
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda a: a.set_session_mode("m", "s"),
        lambda a: a.set_config_option("k", "s", "v"),
        lambda a: a.fork_session(),
        lambda a: a.list_sessions(),
        lambda a: a.ext_method("custom/thing", {}),
    ],
)
async def test_unsupported_methods_return_method_not_found(call):
    agent, _ = make_agent(FakeBackend())
    with pytest.raises(RequestError) as exc:
        await call(agent)
    assert exc.value.code == -32601


# ----------------------------------------------------------------------
# 模型选择：session/set_model 与 new_session 模型列表
# ----------------------------------------------------------------------

FAKE_MODELS = [
    {"name": "basic", "model": "deepseek-chat", "display_name": "Basic", "description": "默认基础模型"},
    {"name": "advanced", "model": "deepseek-reasoner", "display_name": "Advanced", "description": "高级推理模型", "supports_thinking": True},
]


async def test_set_session_model_unknown_session_errors():
    agent, _ = make_agent(FakeBackend(models=FAKE_MODELS))
    with pytest.raises(RequestError) as exc:
        await agent.set_session_model("basic", "df-missing")
    assert exc.value.code == ERROR_UNKNOWN_SESSION


async def test_set_session_model_invalid_model_is_invalid_params_not_silent():
    """无效模型必须显式报错，不允许静默接受。"""
    agent, _ = make_agent(FakeBackend(models=FAKE_MODELS))
    session_id = (await agent.new_session("/tmp")).session_id
    with pytest.raises(RequestError) as exc:
        await agent.set_session_model("no-such-model", session_id)
    assert exc.value.code == -32602


async def test_set_session_model_accepts_listed_model():
    agent, _ = make_agent(FakeBackend(models=FAKE_MODELS))
    session_id = (await agent.new_session("/tmp")).session_id
    # ACP set_session_model 返回 None 或响应对象都合法，不应抛错
    await agent.set_session_model("advanced", session_id)


async def test_set_session_model_applies_to_next_turn():
    """set 后的下一轮 turn 必须使用新模型；set 之前的轮次不受影响。"""
    backend = FakeBackend(models=FAKE_MODELS)
    agent, _ = make_agent(backend)
    session_id = (await agent.new_session("/tmp")).session_id

    await agent.prompt([text_block("第一轮")], session_id)
    assert backend.model_calls == [None]

    await agent.set_session_model("advanced", session_id)
    await agent.prompt([text_block("第二轮")], session_id)
    assert backend.model_calls == [None, "advanced"]


async def test_model_override_is_per_session():
    """模型覆盖只作用于设置它的会话，不泄漏到其他会话。"""
    backend = FakeBackend(models=FAKE_MODELS)
    agent, _ = make_agent(backend)
    first = (await agent.new_session("/tmp")).session_id
    second = (await agent.new_session("/tmp")).session_id

    await agent.set_session_model("advanced", first)
    await agent.prompt([text_block("hi")], first)
    await agent.prompt([text_block("hi")], second)
    assert backend.model_calls == ["advanced", None]


async def test_new_session_response_carries_model_list():
    """new_session 响应携带的模型清单必须与后端 list_models() 一致。"""
    agent, _ = make_agent(FakeBackend(models=FAKE_MODELS))
    resp = await agent.new_session("/tmp")

    assert resp.models is not None
    assert [(m.model_id, m.name, m.description) for m in resp.models.available_models] == [
        ("basic", "Basic", "默认基础模型"),
        ("advanced", "Advanced", "高级推理模型"),
    ]
    # 未显式配置默认模型时，当前模型取 DeerFlow 默认语义（清单第一项）
    assert resp.models.current_model_id == "basic"


async def test_new_session_current_model_reflects_configured_default():
    """启动期配置了默认模型（DEER_FLOW_ACP_MODEL）时，current_model_id 跟随它。"""
    agent, _ = make_agent(FakeBackend(models=FAKE_MODELS), model_name="advanced")
    resp = await agent.new_session("/tmp")
    assert resp.models is not None
    assert resp.models.current_model_id == "advanced"


async def test_new_session_survives_model_list_failure():
    """模型清单获取失败不得阻断会话创建：models 缺省，会话照常可用。"""
    backend = FakeBackend(model_lookup_error=BackendUnavailableError("清单不可用"))
    agent, _ = make_agent(backend)
    resp = await agent.new_session("/tmp")
    assert resp.session_id
    assert resp.models is None


async def test_set_session_model_with_backend_list_failure_is_rejected():
    """无法取得模型清单时不能盲设模型——显式拒绝，不静默接受。"""
    backend = FakeBackend(model_lookup_error=BackendUnavailableError("清单不可用"))
    agent, _ = make_agent(backend)
    session_id = (await agent.new_session("/tmp")).session_id
    with pytest.raises(RequestError) as exc:
        await agent.set_session_model("whatever", session_id)
    assert exc.value.code == ERROR_BACKEND_UNAVAILABLE


async def test_resumed_session_keeps_model_override_in_same_process():
    """同进程内 resume 复用同一会话对象：模型覆盖保持。"""
    backend = FakeBackend(models=FAKE_MODELS, threads={"df-known": [{"type": "ai", "content": "旧"}]})
    agent, _ = make_agent(backend)
    await agent.resume_session("/tmp", "df-known")
    await agent.set_session_model("advanced", "df-known")

    await agent.resume_session("/tmp", "df-known")
    await agent.prompt([text_block("hi")], "df-known")
    assert backend.model_calls == ["advanced"]


async def test_resume_into_fresh_registry_falls_back_to_default_model():
    """跨进程恢复（父进程重启、注册表重建）后无覆盖记录：回到默认模型。

    这是显式选定的恢复语义：桥不持久化会话级模型状态，恢复后的会话
    使用进程级默认（与「未 set 时行为与现状一致」同源），文档明示。
    """
    backend = FakeBackend(models=FAKE_MODELS, threads={"df-known": [{"type": "ai", "content": "旧"}]})
    first_agent, _ = make_agent(backend)
    await first_agent.resume_session("/tmp", "df-known")
    await first_agent.set_session_model("advanced", "df-known")

    fresh_agent, _ = make_agent(backend)
    await fresh_agent.resume_session("/tmp", "df-known")
    await fresh_agent.prompt([text_block("hi")], "df-known")
    assert backend.model_calls[-1] is None


async def test_backend_without_model_capability_still_serves_sessions():
    """后端未提供模型清单能力时：new_session 不带 models，set_model 显式拒绝。"""

    class NoModelsBackend(FakeBackend):
        list_models = None  # type: ignore[assignment]

    backend = NoModelsBackend()
    agent, _ = make_agent(backend)
    resp = await agent.new_session("/tmp")
    assert resp.models is None

    with pytest.raises(RequestError):
        await agent.set_session_model("basic", resp.session_id)


async def test_ext_notification_is_ignored_silently():
    agent, _ = make_agent(FakeBackend())
    await agent.ext_notification("custom/thing", {})
