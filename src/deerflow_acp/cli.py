"""``deerflow-acp`` 命令行入口。

只有一个真正的服务子命令 ``acp``：在 stdio 上跑 ACP agent server。
Multica 侧的 hermes backend 会无条件在 argv 末尾拼接 ``acp``，因此
这个子命令名不可更改。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import sys
from typing import IO, Any

from . import __version__
from .config import BridgeConfig
from .logging_setup import configure_logging, get_logger, redirect_root_logging_to_stderr

logger = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deerflow-acp",
        description="DeerFlow 的 ACP(Agent Client Protocol) 桥接器",
    )
    parser.add_argument("--version", action="version", version=f"deerflow-acp {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    acp_parser = subparsers.add_parser("acp", help="在 stdio 上运行 ACP agent server")
    acp_parser.add_argument(
        "--log-level",
        default=None,
        help="日志级别（默认取 DEERFLOW_ACP_LOG_LEVEL，再默认 INFO）；日志始终写 stderr",
    )
    acp_parser.add_argument("--config-path", default=None, help="DeerFlow config.yaml 路径；默认由 DeerFlow 自行解析")
    acp_parser.add_argument("--model", default=None, help="覆盖 DeerFlow 默认模型名")

    doctor_parser = subparsers.add_parser("doctor", help="检查 DeerFlow 运行时是否可用")
    doctor_parser.add_argument("--log-level", default=None)

    return parser


def isolate_stdout() -> IO[bytes]:
    """把真实 stdout 抢占为 JSON-RPC 专用通道。

    做法：先 ``dup`` 出 fd 1 的副本作为协议输出，再把 fd 1 本身重定向到
    fd 2。此后进程内任何写 fd 1 的代码（DeerFlow / LangGraph / C 扩展里的
    ``print`` 或裸 ``write``）都会落到 stderr，物理上不可能污染协议流。
    """
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(protocol_fd, "wb", buffering=0)


class ParseErrorReader(asyncio.StreamReader):
    """在 SDK 之前拦截畸形 JSON 行，回 JSON-RPC ``-32700``。

    ACP SDK 的接收循环对 ``json.loads`` 失败只记一条日志然后 ``continue``，
    客户端拿不到任何响应。协议边界是桥的职责，所以这里在 reader 层补齐：
    解析失败的行直接回 parse error 并丢弃，连接保持存活。
    """

    def __init__(self) -> None:
        super().__init__()
        self._protocol_writer: asyncio.StreamWriter | None = None

    def bind_writer(self, writer: asyncio.StreamWriter) -> None:
        self._protocol_writer = writer

    async def readline(self) -> bytes:
        while True:
            line = await super().readline()
            if not line:
                return line
            stripped = line.strip()
            if not stripped:
                continue
            try:
                json.loads(stripped)
            except ValueError:
                await self._reply_parse_error()
                continue
            return line

    async def _reply_parse_error(self) -> None:
        if self._protocol_writer is None:
            return
        payload = {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "Parse error"},
        }
        self._protocol_writer.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
        with contextlib.suppress(Exception):
            await self._protocol_writer.drain()


async def _stdio_streams(protocol_stdout: IO[bytes]) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """在 stdin 与被隔离出来的 stdout 之间建立 asyncio 流。"""
    loop = asyncio.get_running_loop()

    reader = ParseErrorReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)

    # FlowControlMixin 提供 StreamWriter.drain() 所需的 _drain_helper。
    transport, protocol = await loop.connect_write_pipe(
        lambda: asyncio.streams.FlowControlMixin(loop=loop),
        protocol_stdout,
    )
    writer = asyncio.StreamWriter(transport, protocol, None, loop)
    reader.bind_writer(writer)
    return reader, writer


async def serve(config: BridgeConfig, protocol_stdout: IO[bytes]) -> int:
    import acp

    from .agent import DeerFlowAgent

    agent_holder: dict[str, DeerFlowAgent] = {}

    def to_agent(connection: Any) -> DeerFlowAgent:
        agent = DeerFlowAgent(connection, config=config)
        agent_holder["agent"] = agent
        return agent

    reader, writer = await _stdio_streams(protocol_stdout)

    loop = asyncio.get_running_loop()
    shutdown = asyncio.Event()

    def request_shutdown() -> None:
        # SIGINT/SIGTERM：先向所有活跃会话发协作式取消，再让 listen() 收尾。
        agent = agent_holder.get("agent")
        if agent is not None:
            for session in agent.registry.active_sessions:
                session.cancel_event.set()
        shutdown.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, request_shutdown)

    # run_agent 在 stdin EOF 时正常返回——这就是「stdin 断连即退出」的实现。
    # use_unstable_protocol=True 是硬需求：SDK 把 session/resume、session/close
    # 标为 unstable，关闭时会直接回 -32601，而 Multica 客户端续会话正是走
    # session/resume。
    # 注意 run_agent 的参数命名：input_stream 是写端，output_stream 是读端。
    serve_task = asyncio.create_task(
        acp.run_agent(to_agent, input_stream=writer, output_stream=reader, use_unstable_protocol=True)
    )
    shutdown_task = asyncio.create_task(shutdown.wait())

    await asyncio.wait({serve_task, shutdown_task}, return_when=asyncio.FIRST_COMPLETED)

    if serve_task.done():
        shutdown_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await shutdown_task
        exc = serve_task.exception()
        if exc is not None:
            logger.error("ACP server 异常退出：%s", type(exc).__name__, exc_info=exc)
            return 1
        logger.info("stdin 已断开，ACP server 正常退出")
        return 0

    # 收到信号：给在途 turn 一个宽限期，超时直接退出，绝不无限等待。
    logger.info("收到终止信号，等待在途 turn 收尾（最多 %.1fs）", config.shutdown_grace_seconds)
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(serve_task), timeout=config.shutdown_grace_seconds)
    serve_task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await serve_task
    return 0


def _doctor(config: BridgeConfig) -> int:
    from .backend import BackendUnavailableError, EmbeddedDeerFlowBackend

    from . import __version__

    print(f"deerflow-acp {__version__}", file=sys.stderr)
    backend = EmbeddedDeerFlowBackend(config)
    try:
        # 用一个必然不存在的 thread id 触发完整的客户端初始化 + checkpointer 访问。
        backend.thread_exists("deerflow-acp-doctor-probe")
    except BackendUnavailableError as exc:
        print(f"DeerFlow 运行时不可用：{exc}", file=sys.stderr)
        return 1
    print("DeerFlow 运行时可用", file=sys.stderr)
    return 0


def _apply_overrides(config: BridgeConfig, args: argparse.Namespace) -> BridgeConfig:
    overrides: dict[str, Any] = {}
    if getattr(args, "config_path", None):
        overrides["deerflow_config_path"] = args.config_path
    if getattr(args, "model", None):
        overrides["model_name"] = args.model
    if not overrides:
        return config
    return BridgeConfig(**{**config.__dict__, **overrides})


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    redirect_root_logging_to_stderr(args.log_level)

    config = _apply_overrides(BridgeConfig.from_env(), args)

    if args.command == "doctor":
        return _doctor(config)

    protocol_stdout = isolate_stdout()
    try:
        return asyncio.run(serve(config, protocol_stdout))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
