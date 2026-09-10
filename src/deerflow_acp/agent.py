"""ACP Agent 实现：把 ACP 方法调用翻译成 DeerFlow 调用。"""

from __future__ import annotations

import asyncio
from typing import Any

from acp import PROTOCOL_VERSION
from acp.exceptions import RequestError
from acp.helpers import (
    session_notification,
    update_agent_message_text,
    update_user_message_text,
)
from acp.schema import (
    AgentCapabilities,
    ClientCapabilities,
    Implementation,
    InitializeResponse,
    LoadSessionResponse,
    McpCapabilities,
    NewSessionResponse,
    PromptCapabilities,
    PromptResponse,
    ResumeSessionResponse,
    Usage,
    UsageUpdate,
)

from . import __version__
from .backend import BackendUnavailableError, DeerFlowBackend, EmbeddedDeerFlowBackend
from .config import BridgeConfig
from .events import EventNormalizer
from .logging_setup import get_logger
from .runner import RemoteBackendError, TurnRunner
from .sanitize import describe_exception, redact_text
from .session import (
    Session,
    SessionIdError,
    SessionQuarantinedError,
    SessionRegistry,
    TurnAlreadyRunningError,
    UnknownSessionError,
    validate_session_id,
)

logger = get_logger("agent")

AGENT_NAME = "deerflow-acp"
AGENT_TITLE = "DeerFlow ACP Bridge"

# 私有 JSON-RPC 错误码（-32000..-32099 为实现保留区间），
# 用于让客户端区分「会话不存在」与「后端故障」。
ERROR_UNKNOWN_SESSION = -32001
ERROR_BACKEND_UNAVAILABLE = -32010
ERROR_TURN_IN_PROGRESS = -32011
#: 会话被隔离：上一轮 worker 进程组未确认终结，不允许在同一 thread 上继续。
#: 与「后端故障」严格分开——这是桥主动拒绝并发写 checkpoint，后端本身可能好着。
ERROR_SESSION_QUARANTINED = -32012


def _text_from_prompt(blocks: list[Any]) -> str:
    """把 ACP prompt 内容块拼成 DeerFlow 需要的单条文本。

    DeerFlow ``stream()`` 只接受 ``str``，因此非文本块（图片、音频、
    嵌入资源）无法无损传递。这里只取文本块，其余显式拒绝，
    不做静默丢弃——能力协商中也已声明不支持这些块。
    """
    parts: list[str] = []
    unsupported: list[str] = []
    for block in blocks:
        block_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
        if block_type == "text":
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        else:
            unsupported.append(str(block_type or "unknown"))

    if unsupported:
        raise RequestError.invalid_params(
            {
                "reason": "deerflow-acp 只接受 text 内容块",
                "unsupportedBlockTypes": sorted(set(unsupported)),
            }
        )
    if not parts:
        raise RequestError.invalid_params({"reason": "prompt 中没有任何非空 text 内容块"})
    return "\n\n".join(parts)


def _error_type_name(exc: BaseException) -> str:
    """给客户端的错误分类标签。

    worker 子进程里的异常过河时只剩类型名（见 :mod:`deerflow_acp.ipc`），
    在父进程侧被包成 ``RemoteBackendError``。直接报这个包装类型没有诊断价值，
    因此还原成 worker 侧的真实类型名。
    """
    if isinstance(exc, RemoteBackendError):
        return exc.error_class
    return type(exc).__name__


def _quarantine_error(session_id: str) -> RequestError:
    """会话被隔离时给客户端的错误。

    不回显 pgid：进程号对客户端没有可操作价值，运维信息留在 stderr 日志里。
    hint 必须明确「换新会话」，否则客户端会一直重试同一个 sessionId。
    """
    return RequestError(
        ERROR_SESSION_QUARANTINED,
        "该会话已被隔离：上一轮执行体未能确认终结",
        {
            "sessionId": session_id,
            "hint": "为避免与残留执行体并发写同一条对话历史，请用 session/new 开启新会话",
        },
    )


class DeerFlowAgent:
    """ACP ``Agent`` 协议实现。"""

    def __init__(
        self,
        connection: Any,
        *,
        config: BridgeConfig | None = None,
        backend: DeerFlowBackend | None = None,
        runner: TurnRunner | None = None,
    ) -> None:
        self._conn = connection
        self._config = config or BridgeConfig.from_env()
        # initialize 阶段不构造 DeerFlowClient：EmbeddedDeerFlowBackend
        # 内部对 client 做懒加载，重型后端只在第一次真正需要时才拉起。
        self._backend = backend or EmbeddedDeerFlowBackend(self._config)
        # runner 只在契约测试里显式给出（脚本化后端 + 真 worker 子进程）；
        # 生产与单元测试都由 SessionRegistry 按后端类型自行选择。
        self._sessions = SessionRegistry(self._backend, self._config, runner=runner)

    @property
    def registry(self) -> SessionRegistry:
        return self._sessions

    # ------------------------------------------------------------------
    # initialize
    # ------------------------------------------------------------------

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        if client_info is not None:
            logger.info("客户端接入：%s %s", client_info.name, getattr(client_info, "version", ""))
        # 协议版本按 ACP 约定向下协商：返回双方都支持的较低版本。
        # 注意 0 是合法版本号，不能用真值判断。
        if isinstance(protocol_version, int):
            negotiated = min(max(protocol_version, 0), PROTOCOL_VERSION)
        else:
            negotiated = PROTOCOL_VERSION
        return InitializeResponse(
            protocol_version=negotiated,
            agent_info=Implementation(name=AGENT_NAME, version=__version__, title=AGENT_TITLE),
            agent_capabilities=AgentCapabilities(
                load_session=True,
                # DeerFlow 的 MCP 接入由 DeerFlow 自身配置管理，桥不接受
                # 客户端下发的 mcpServers，显式声明为不支持。
                mcp_capabilities=McpCapabilities(http=False, sse=False),
                prompt_capabilities=PromptCapabilities(
                    audio=False,
                    embedded_context=False,
                    image=False,
                ),
            ),
        )

    async def authenticate(self, method_id: str, **kwargs: Any) -> None:
        # 桥不实现任何认证方法：凭据完全由 DeerFlow 的本地秘密注入机制提供。
        raise RequestError.method_not_found("authenticate")

    # ------------------------------------------------------------------
    # session 生命周期
    # ------------------------------------------------------------------

    async def new_session(self, cwd: str, mcp_servers: list[Any] | None = None, **kwargs: Any) -> NewSessionResponse:
        self._reject_mcp_servers(mcp_servers)
        session = self._sessions.create(cwd)
        return NewSessionResponse(session_id=session.session_id)

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse:
        session = self._resume_session(session_id, cwd, mcp_servers)
        await self._replay_history(session)
        return LoadSessionResponse()

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        """``session/resume``：与 ``session/load`` 同一恢复语义。

        差别只在于 resume 不重放历史——Multica 客户端在续会话时已持有
        本地会话记录，重放会造成 UI 重复。
        """
        self._resume_session(session_id, cwd, mcp_servers)
        return ResumeSessionResponse()

    async def close_session(self, session_id: str, **kwargs: Any) -> None:
        self._sessions.close(session_id)

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        self._sessions.cancel(session_id)

    # ------------------------------------------------------------------
    # prompt
    # ------------------------------------------------------------------

    async def prompt(self, prompt: list[Any], session_id: str, **kwargs: Any) -> PromptResponse:
        try:
            session = self._sessions.get(session_id)
        except SessionQuarantinedError:
            raise _quarantine_error(session_id) from None
        except UnknownSessionError:
            raise RequestError(
                ERROR_UNKNOWN_SESSION,
                "未知会话",
                {"sessionId": session_id, "hint": "先调用 session/new，或用 session/load 恢复已有会话"},
            ) from None

        message = _text_from_prompt(prompt)
        normalizer = EventNormalizer()

        async def on_event(event_type: str, data: dict[str, Any]) -> None:
            for update in normalizer.normalize(event_type, data):
                await self._conn.session_update(session_id=session_id, update=update)

        try:
            outcome = await self._sessions.run_turn(session, message, on_event)
        except SessionQuarantinedError:
            # 两处都要拦：进入前（上一轮留下的隔离）与本轮收尾时新置的隔离。
            raise _quarantine_error(session_id) from None
        except TurnAlreadyRunningError:
            raise RequestError(
                ERROR_TURN_IN_PROGRESS,
                "该会话已有 turn 正在执行",
                {"sessionId": session_id},
            ) from None
        except BackendUnavailableError as exc:
            raise RequestError(
                ERROR_BACKEND_UNAVAILABLE, "DeerFlow 后端不可用", {"detail": redact_text(exc)}
            ) from None

        if outcome.error is not None:
            if isinstance(outcome.error, BackendUnavailableError):
                raise RequestError(
                    ERROR_BACKEND_UNAVAILABLE,
                    "DeerFlow 后端不可用",
                    {"detail": redact_text(outcome.error)},
                ) from None
            # 后端异常已经中止了本轮；把可诊断分类交给客户端，
            # 但不回显 traceback 与配置内容（可能含路径与秘密）。
            # stderr 同样是客户端可见的输出面：绝不能用 exc_info=——那会把
            # 未经脱敏的异常消息和整条 traceback 直接写出去。
            logger.error(
                "会话 %s 的 turn 执行失败：%s（%s）",
                session_id,
                describe_exception(outcome.error),
                redact_text(outcome.error),
            )
            raise RequestError(
                RequestError.internal_error().code,
                "DeerFlow turn 执行失败",
                {"sessionId": session_id, "errorType": _error_type_name(outcome.error)},
            ) from None

        if outcome.escalated:
            # 强制终止已在会话层执行并确认；这里只补一条面向运维的可观测记录。
            # worker_reaped=False 意味着旧执行体可能仍在写同一条 DeerFlow thread，
            # 属于必须显眼上报的状态，不能与普通升级混为一谈。
            if outcome.worker_reaped:
                logger.warning(
                    "会话 %s 取消超时，worker %s（进程组 %s）已被强制终止并确认回收",
                    session_id,
                    outcome.worker_pid,
                    outcome.worker_pgid,
                )
            else:
                logger.error(
                    "会话 %s 取消超时，worker %s（进程组 %s）未能确认回收",
                    session_id,
                    outcome.worker_pid,
                    outcome.worker_pgid,
                )

        usage = self._build_usage(outcome.usage_payload)
        if usage is not None and self._config.emit_usage_update:
            await self._emit_usage_update(session_id, usage)

        return PromptResponse(stop_reason=outcome.stop_reason, usage=usage)

    # ------------------------------------------------------------------
    # 未实现能力：显式降级，不伪造
    # ------------------------------------------------------------------

    async def set_session_mode(self, mode_id: str, session_id: str, **kwargs: Any) -> None:
        raise RequestError.method_not_found("session/set_mode")

    async def set_session_model(self, model_id: str, session_id: str, **kwargs: Any) -> None:
        raise RequestError.method_not_found("session/set_model")

    async def set_config_option(self, config_option_id: str, session_id: str, value: Any, **kwargs: Any) -> None:
        raise RequestError.method_not_found("session/set_config_option")

    async def fork_session(self, *args: Any, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("session/fork")

    async def list_sessions(self, *args: Any, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("session/list")

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        logger.debug("忽略未知扩展通知：%s", method)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _reject_mcp_servers(self, mcp_servers: list[Any] | None) -> None:
        if mcp_servers:
            raise RequestError.invalid_params(
                {
                    "reason": "deerflow-acp 不接受客户端下发的 mcpServers",
                    "hint": "MCP 接入请在 DeerFlow 自身配置中声明",
                }
            )

    def _resume_session(self, session_id: str, cwd: str, mcp_servers: list[Any] | None) -> Session:
        self._reject_mcp_servers(mcp_servers)
        try:
            validate_session_id(session_id)
        except SessionIdError as exc:
            raise RequestError.invalid_params({"reason": str(exc), "sessionId": session_id}) from None
        try:
            return self._sessions.resume(session_id, cwd)
        except SessionQuarantinedError:
            # 隔离必须在 resume 这一层就拦住：注册项已被摘掉，只靠 prompt 拦不住
            # 「resume 拉回同一个 thread_id 再发 prompt」这条绕行路径。
            raise _quarantine_error(session_id) from None
        except UnknownSessionError:
            raise RequestError(
                ERROR_UNKNOWN_SESSION,
                "未知会话：DeerFlow 中不存在该 thread 的 checkpoint",
                {"sessionId": session_id},
            ) from None
        except BackendUnavailableError as exc:
            raise RequestError(
                ERROR_BACKEND_UNAVAILABLE, "DeerFlow 后端不可用", {"detail": redact_text(exc)}
            ) from None

    async def _replay_history(self, session: Session) -> None:
        """``session/load`` 时把历史消息重放为 session updates。"""
        history_fn = getattr(self._backend, "history", None)
        if not callable(history_fn):
            return
        try:
            messages = await asyncio.to_thread(history_fn, session.session_id)
        except BackendUnavailableError as exc:
            raise RequestError(
                ERROR_BACKEND_UNAVAILABLE, "DeerFlow 后端不可用", {"detail": redact_text(exc)}
            ) from None

        for message in messages:
            kind = message.get("type")
            text = message.get("content")
            if not isinstance(text, str) or not text:
                continue
            if kind == "human":
                update = update_user_message_text(text)
            elif kind == "ai":
                update = update_agent_message_text(text)
            else:
                # tool / system 消息没有无损的历史重放形态（ACP 的
                # tool_call_update 要求先有 tool_call 声明），跳过而不伪造。
                continue
            await self._conn.session_update(session_id=session.session_id, update=update)

    def _build_usage(self, payload: dict[str, Any] | None) -> Usage | None:
        if not isinstance(payload, dict):
            return None
        try:
            return Usage(
                input_tokens=max(0, int(payload.get("input_tokens") or 0)),
                output_tokens=max(0, int(payload.get("output_tokens") or 0)),
                total_tokens=max(0, int(payload.get("total_tokens") or 0)),
            )
        except (TypeError, ValueError):
            logger.warning("无法解析 DeerFlow usage 负载，已跳过 usage 上报")
            return None

    async def _emit_usage_update(self, session_id: str, usage: Usage) -> None:
        """下发 ``usage_update``。

        ACP 的 ``usage_update`` 表达的是「上下文窗口占用」，而 DeerFlow 给的是
        本轮 token 增量，两者语义不同。因此只有在用户显式配置了窗口大小
        （``DEERFLOW_ACP_CONTEXT_WINDOW_TOKENS``）时才近似上报，默认关闭。
        """
        size = self._config.context_window_tokens
        if not size:
            return
        await self._conn.session_update(
            session_id=session_id,
            update=UsageUpdate(
                session_update="usage_update",
                size=size,
                used=min(size, usage.total_tokens),
            ),
        )


def build_agent(
    connection: Any,
    *,
    config: BridgeConfig | None = None,
    backend: DeerFlowBackend | None = None,
    runner: TurnRunner | None = None,
) -> DeerFlowAgent:
    return DeerFlowAgent(connection, config=config, backend=backend, runner=runner)


__all__ = [
    "AGENT_NAME",
    "ERROR_BACKEND_UNAVAILABLE",
    "ERROR_SESSION_QUARANTINED",
    "ERROR_TURN_IN_PROGRESS",
    "ERROR_UNKNOWN_SESSION",
    "DeerFlowAgent",
    "build_agent",
    "session_notification",
]
