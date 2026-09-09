# deerflow-acp

DeerFlow 与 ACP（Agent Client Protocol）客户端之间的**独立桥接器**。

桥只负责协议层的事：ACP 方法实现、事件归一化、会话恢复、取消与进程生命周期。
它**不复制 DeerFlow 的研究编排能力**——所有编排、工具调用、模型选择、
记忆与检索仍然发生在 DeerFlow 内部，桥通过嵌入式 `DeerFlowClient` 消费其事件流。

## 架构

```mermaid
flowchart LR
    C["ACP Client<br/>(Multica hermes backend)"] -- "JSON-RPC / stdio" --> S["deerflow-acp acp<br/>(ACP server)"]
    S --> R["SessionRegistry<br/>sessionId ↔ thread_id"]
    R --> W["worker thread<br/>驱动同步生成器"]
    W --> D["DeerFlowClient.stream()"]
    D --> N["EventNormalizer"]
    N -- "session/update" --> C
    D --> CP[("LangGraph<br/>checkpointer")]
```

关键设计：

| 关注点 | 做法 |
| --- | --- |
| 会话映射 | ACP `sessionId` **就是** DeerFlow `thread_id`（同一字符串），桥不维护额外映射表 |
| 会话恢复 | 以 `DeerFlowClient.get_thread()` 是否返回 checkpoint 为唯一依据；不存在则报错，绝不静默新建 |
| 取消 | 协作式优先：置标志 → 工作线程在下一个 yield 边界 `generator.close()`。宽限期由**事件循环侧**计时，因此后端即使卡在 yield **之前**（模型/工具调用还没返回），`session/prompt` 仍在宽限期内返回 `cancelled` 并标记 `escalated`；卡住的工作线程被弃用，不影响同一 session 的后续 turn |
| stdout 纪律 | 启动时 `dup(1)` 出协议专用 fd，再把 fd 1 重定向到 fd 2；任何 `print` 物理上无法污染协议流 |
| 后端加载 | `initialize` 只回能力，不构造 `DeerFlowClient`；重型后端在首个真实请求时才拉起 |
| 凭据 | 桥不存储、不打印、不上传任何秘密；完全沿用 DeerFlow 既有的本地 `.env` 注入机制 |
| 秘密脱敏 | 所有离开进程的文本（JSON-RPC 响应体、stderr 日志、客户端可见事件）统一过 `sanitize.redact_text()`；对外错误只保留异常**类型名**等可诊断分类，不回显异常消息、traceback 与配置内容 |

## 安装

DeerFlow harness（`deerflow-harness`）**未发布到 PyPI**，只能以本地路径安装，
因此桥不把它声明为强依赖，而是要求运行环境里已经装好：

```bash
cd /home/guxy/Codes/offcial/deerflow-acp
uv venv -p 3.12
uv pip install --python .venv/bin/python /path/to/deerflow/backend/packages/harness
uv pip install --python .venv/bin/python -e '.[dev]'
```

自检 DeerFlow 运行时是否可用（不启动 ACP server）：

```bash
.venv/bin/deerflow-acp doctor
```

## 运行

```bash
deerflow-acp acp            # 在 stdio 上跑 ACP server
```

Multica 的 hermes backend 会无条件在 argv 末尾拼接 `acp`，所以在 Multica 里
只需要把命令配成 `/abs/path/to/.venv/bin/deerflow-acp`（不带子命令）。

## 配置

全部通过环境变量，桥自身不读写任何配置文件：

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `DEERFLOW_ACP_LOG_LEVEL` | `INFO` | 日志级别；日志**始终**写 stderr |
| `DEERFLOW_ACP_CONFIG_PATH` | 空 | DeerFlow `config.yaml` 路径；空则由 DeerFlow 自行解析 |
| `DEERFLOW_ACP_MODEL` | 空 | 覆盖 DeerFlow 默认模型名 |
| `DEERFLOW_ACP_THINKING` | `true` | 是否请求模型输出推理内容（映射为 `agent_thought_chunk`） |
| `DEERFLOW_ACP_CANCEL_GRACE_SECONDS` | `5` | 取消后等待后端协作退出的宽限期 |
| `DEERFLOW_ACP_SHUTDOWN_GRACE_SECONDS` | `5` | 收到 SIGTERM/SIGINT 后等待在途 turn 收尾的时限 |
| `DEERFLOW_ACP_CONTEXT_WINDOW_TOKENS` | 空 | 上下文窗口大小；不设则**不下发** `usage_update` |
| `DEERFLOW_ACP_EMIT_USAGE_UPDATE` | `false` | 仅在同时设置了窗口大小时生效，见下方「usage 降级」 |

模板见 `config.example.env`。**该文件只放非秘密的行为开关**；模型 / 搜索
API key 一律走 DeerFlow 自己的 gitignored `.env`，桥不接触。

## 能力矩阵与降级

见 [`docs/compatibility.md`](docs/compatibility.md)。要点：

- **支持**：`initialize`、`session/new`、`session/load`、`session/resume`、
  `session/prompt`、`session/cancel`、`session/close`、流式
  `agent_message_chunk` / `agent_thought_chunk` / `tool_call` / `tool_call_update`。
- **显式降级**（返回 `-32601`，不伪造）：`authenticate`、`session/set_mode`、
  `session/set_model`、`session/set_config_option`、`session/fork`、`session/list`。
- **不上报**：`plan`（DeerFlow 无稳定的结构化计划事件源）、
  `session/request_permission`（DeerFlow 侧无权限询问回路）。
- **usage 降级**：ACP `usage_update` 的 `size`/`used` 表达上下文窗口占用，
  DeerFlow 只给 input/output/total token 增量，语义不同。默认**不发**
  `usage_update`，只在 `PromptResponse.usage` 里如实回传 token 数。

## 回退

桥不可用时的回退路径（按代价从低到高）：

1. `deerflow-acp doctor` 判定是 DeerFlow 运行时问题还是协议层问题。
2. 设 `DEERFLOW_ACP_LOG_LEVEL=DEBUG` 复跑，stderr 会打印被忽略的事件类型。
3. 绕开桥，直接用 DeerFlow 自身的 HTTP 网关（`http://127.0.0.1:2026`）验证
   后端是否正常——桥的故障与 DeerFlow 的故障由此分账。
4. 桥是独立进程，卸载它不影响 DeerFlow 与 Multica 任何既有功能。

## 来源标识

`deerflow_acp/__init__.py` 的 `__version__` 与本仓库 Git commit 一一对应；
构建产物的来源追溯见交付卡中记录的完整 40 位 Git SHA。

## 测试

```bash
.venv/bin/pytest -q                       # 单元 + 协议契约测试
DEERFLOW_ACP_E2E=1 .venv/bin/pytest -q tests/test_e2e_deerflow.py   # 需要本机 DeerFlow
```
