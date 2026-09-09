# 兼容矩阵与降级说明

本文件记录 deerflow-acp 在 ACP 协议面、Multica 客户端和 DeerFlow 后端三个方向上
的支持边界。**凡是本文件写「不支持」的能力，桥一律返回明确错误码，绝不静默降级
成看起来成功的空实现。**

## 版本锁定

| 组件 | 版本 | 锁定位置 | 说明 |
| --- | --- | --- | --- |
| `agent-client-protocol`（Python SDK） | `0.9.0` | `pyproject.toml` 精确固定 | ACP 语义在 0.x 内不保证稳定，必须精确锁定 |
| ACP `PROTOCOL_VERSION` | `1` | SDK 常量 | 桥向下协商，见下节 |
| Python | `>= 3.12` | `pyproject.toml` | 用到 PEP 604 语法与 `asyncio.to_thread` 语义 |
| DeerFlow | 本机部署基线 `6d6df4e8022f5c63e95749b37562fba6d9114d43` | 运行时本地路径依赖 | 未发布 PyPI，只能以本地路径安装 |

DeerFlow harness **不在** `dependencies` 中。它不在任何公开索引上，写进依赖会让
`pip install deerflow-acp` 直接失败。安装方式见 README。

## 协议版本协商

桥实现 ACP 约定的向下协商：返回 `min(客户端版本, 1)`，并对负数钳制到 0。

| 客户端 `protocolVersion` | 桥返回 | 说明 |
| --- | --- | --- |
| `0` | `0` | `0` 是合法版本号，不能按 falsy 处理 |
| `1` | `1` | 当前主用路径 |
| `2`（未来版本） | `1` | 截断到桥支持的上限 |
| 缺失 / 非整数 | `1` | 按当前版本处理 |

## ACP 方法支持面

| 方法 | 状态 | 行为 |
| --- | --- | --- |
| `initialize` | 支持 | 不构造 `DeerFlowClient`，不触发任何模型或凭据加载 |
| `session/new` | 支持 | 生成 `df-<uuid4hex>` 作为 sessionId，同时即为 DeerFlow `thread_id` |
| `session/load` | 支持 | 恢复会话并**重放**历史消息 |
| `session/resume` | 支持（unstable） | 恢复会话，**不重放**——Multica 客户端已持有本地记录，重放会造成 UI 重复 |
| `session/prompt` | 支持 | 仅接受 `text` 内容块 |
| `session/cancel` | 支持 | 通知型。先向 worker 进程组发 `SIGTERM` 请求协作退出，宽限期超时后 `killpg(SIGKILL)`；未知 sessionId 静默忽略（无响应通道） |
| `session/close` | 支持（unstable） | 释放注册表条目，DeerFlow checkpoint 保留 |
| `authenticate` | **不支持** | `-32601`。凭据由 DeerFlow 本地机制注入，桥不参与认证 |
| `session/set_mode` | **不支持** | `-32601` |
| `session/set_model` | **不支持** | `-32601`。模型经 `DEERFLOW_ACP_MODEL` 在启动时固定 |
| `session/set_config_option` | **不支持** | `-32601` |
| `session/fork` | **不支持** | `-32601`。DeerFlow checkpointer 无 fork 语义 |
| `session/list` | **不支持** | `-32601`。桥不持久化会话清单 |
| `_ext/*` | **不支持** | `-32601` |

`session/resume`、`session/close`、`session/fork`、`session/set_model` 在 SDK router
中标记为 `unstable=True`。桥必须以 `use_unstable_protocol=True` 启动，否则
Multica 的续会话请求会直接收到 `-32601`。这是硬需求，不是可选项。

## 错误码

| 码 | 名称 | 触发条件 |
| --- | --- | --- |
| `-32700` | Parse error | stdin 上出现非法 JSON 行。桥在 reader 层拦截（SDK 本身只记日志不回响应），回错误后连接保持存活 |
| `-32601` | Method not found | 上表标「不支持」的方法 |
| `-32602` | Invalid params | 空 prompt、非 `text` 内容块（附 `unsupportedBlockTypes`）、客户端下发 `mcpServers`、非法 sessionId 格式 |
| `-32001` | Unknown session | sessionId 在本进程与 DeerFlow checkpointer 中都不存在 |
| `-32010` | Backend unavailable | `DeerFlowClient` 构造失败或 checkpointer 读取异常 |
| `-32011` | Turn in progress | 同一 session 上已有 turn 在跑 |
| `-32603` | Internal error | turn 执行失败。**响应体只含 `sessionId` 与 `errorType`**，不含 message、traceback 或配置内容 |

「未知会话」与「后端故障」严格分账：checkpointer 自身报错不得伪装成
「这个 session 不存在」，否则客户端会误以为历史丢失而新开会话。

## 秘密脱敏

桥无法预知 DeerFlow、LangGraph 或某个 provider SDK 会把什么塞进异常消息——
把整个请求头（含 `Authorization`）或数据库连接串回显进报错是常见做法。因此
**所有离开进程的文本**都先过 `deerflow_acp.sanitize.redact_text()`：

| 输出面 | 处理 |
| --- | --- |
| JSON-RPC error `data` | `-32010` 的 `detail` 已脱敏；`-32603` 只给 `sessionId` + `errorType`（异常类型名） |
| stderr 日志 | 不使用 `exc_info=`（会连 traceback 一起写出）；只记类型名与脱敏后的消息 |
| 客户端可见事件 | custom task 的 `error`、`llm_retry.reason`、`safety_termination.reason` 均脱敏后再转成文本 |

覆盖形态：provider key 前缀串（`sk-` / `ghp_` / `gsk-` / `AKIA…`）、JWT、
`Authorization: Bearer`、`key=value` 形态的 token/secret/password/credential、
URL userinfo、≥32 位裸十六进制串、PEM 私钥块。

脱敏保留诊断价值：host:port、异常类型、重试次数、认证方案名不被抹掉——
否则等于用「不可诊断」换「不泄露」。

## 客户端能力

桥在 `initialize` 中声明：

| 能力 | 值 | 原因 |
| --- | --- | --- |
| `loadSession` | `true` | 支持 `session/load` |
| `promptCapabilities.image` | `false` | DeerFlow 输入接口只收文本 |
| `promptCapabilities.audio` | `false` | 同上 |
| `promptCapabilities.embeddedContext` | `false` | 同上 |
| `mcpCapabilities.http` | `false` | 桥不代理 MCP；DeerFlow 自身工具链独立配置 |
| `mcpCapabilities.sse` | `false` | 同上 |

客户端仍下发 `mcpServers` 非空列表时，桥返回 `-32602` 而不是忽略——静默忽略会
让客户端以为工具已挂载。

## 事件归一化映射

| DeerFlow 事件 | ACP `session/update` | 说明 |
| --- | --- | --- |
| `messages-tuple` / `ai` / `content` | `agent_message_chunk` | 逐块下发 |
| `messages-tuple` / `ai` / `reasoning_content` | `agent_thought_chunk` | 先于同消息的文本块；自动处理 provider 给累计值 vs 增量的差异 |
| `messages-tuple` / `ai` / `tool_calls` | `tool_call` | 无 `id` 的调用**丢弃并告警**，不编造 id |
| `messages-tuple` / `tool` | `tool_call_update`（`completed`） | 无对应 start 时补发 start |
| `custom` / `task_*` | `tool_call` / `tool_call_update` | `task_timed_out` → `failed` |
| `custom` / `llm_retry`、`safety_termination` | `agent_thought_chunk` | 让用户看到重试与安全终止 |
| `end` | 无 update | 仅捕获 usage |
| `values` | **整体丢弃** | 消息已由 `messages-tuple` 逐条下发，重发会造成客户端文本重复 |
| 其他未知类型 | 丢弃 | 记 debug 日志 |

工具 `kind` 按名称映射（`read_file`→`read`、`web_search`→`fetch`、`bash`→`execute`、
`task`→`think` 等），未命中落 `other`。

## Usage 上报

ACP `usage_update` 的 `size` / `used` 表示**上下文窗口占用**；DeerFlow 提供的是
**token 增量**。两者语义不同，无法无损映射。

| 场景 | 行为 |
| --- | --- |
| 默认 | 不发 `usage_update`；仅在 `PromptResponse.usage` 中如实回传 DeerFlow 给的 token 数 |
| 设置 `DEERFLOW_ACP_CONTEXT_WINDOW_TOKENS` 且 `EMIT_USAGE_UPDATE=true` | 以该值为 `size`，累计 token 为 `used`（钳制到 `size`），近似上报 |
| usage 载荷畸形 | 忽略，不上报，不伪造 0 |

## 停止原因

| `stopReason` | 触发条件 |
| --- | --- |
| `end_turn` | 正常结束 |
| `cancelled` | 收到 `session/cancel`。宽限期由事件循环侧计时，**不依赖 worker 报到**：后端卡在下一个 yield 之前（模型/工具调用未返回）时同样在 `CANCEL_GRACE_SECONDS` 内返回。worker 未在宽限期内协作退出时 `killpg(SIGKILL)` 终止整个进程组，**等到进程确认被回收后才返回**并标记 `escalated`（stderr 记警告，不伪装成干净收敛）。因此「宽限期已过」之后不可能再出现晚到的 `session/update`、checkpoint 写入或工具副作用 |
| `refusal` | 后端 `stream()` 抛异常 |

## 进程与流

| 项 | 保证 |
| --- | --- |
| 执行载体 | 每个 turn 一个 worker 子进程（`python -m deerflow_acp.worker`），`start_new_session=True` 置于独立进程组。worker 内用**嵌入式** `DeerFlowClient`，不是 CLI 文本包装 |
| 桥 ↔ worker | 单向 ndJSON over worker stdout，4 种消息：`ready` / `ev` / `done` / `err`。job 载荷经 worker stdin 下发，只含 `session_id` / `message` / `thread_id` 与非凭据配置字段 |
| stdout（桥） | 仅 JSON-RPC。fd 级隔离：`dup(1)` 出协议专用 fd 后 `dup2(2, 1)`，进程内任何写 fd 1 的代码（含 C 扩展裸 `write`）都落 stderr |
| stdout（worker） | 仅 ndJSON IPC。worker 内做同样的 `dup`/`dup2` 隔离，因此 DeerFlow 或任何 provider SDK 的 `print` 不会撕裂 IPC 通道，更不会接到 ACP 协议 fd 上 |
| stderr | 全部日志（桥与 worker 同流）。级别由 `DEERFLOW_ACP_LOG_LEVEL` 控制 |
| 跨进程异常 | 只传 `type(exc).__name__`。异常消息、`args`、`__cause__` 与 traceback 一律不过 IPC 边界，因此凭据即使被塞进异常消息也到不了桥进程或客户端 |
| stdin EOF | 退出码 0；在途 worker 进程组被强制回收后才退出 |
| `SIGINT` / `SIGTERM` | 先给活跃会话置取消标志，宽限 `SHUTDOWN_GRACE_SECONDS`；超时后取消在途 turn 协程并强制回收所有在途 worker 进程组 |
| 孤儿进程 | 双闸冗余：① turn 协程被中断时在 `except BaseException` 内**同步** `killpg`（`await` 在取消传播期间可能再被打断，同步系统调用不会）；② 关停路径再调 `terminate_all_workers()` 兜底。两闸互为冗余——缺任一仍不产生孤儿，同时缺失才会漏 |

## 已知限制

1. **历史重放不含工具调用**。`session/load` 只重放 human / ai 文本消息，tool 与
   system 消息跳过——ACP 没有无损表达历史工具调用的形态，跳过好过伪造。
2. **同一 session 不支持并发 turn**。第二个 `session/prompt` 返回 `-32011`。
3. **模型在进程生命周期内固定**。`session/set_model` 不支持，换模型需重启桥。
4. **`session/fork` 不支持**。DeerFlow checkpointer 无对应语义。
5. **凭据完全交给 DeerFlow**。桥不读、不存、不转发任何 API key。
6. **脱敏是启发式的**。基于形态匹配，不可能覆盖全部秘密形态；它是最后一道
   兜底，不替代「不要把秘密放进异常消息」这条上游纪律。
7. **DeerFlow 按 cwd 定位 `config.yaml`**。当前版本 `DeerFlowClient(config_path=...)`
   不改变查找根，桥进程必须在 DeerFlow 部署根目录下启动（E2E 测试以
   `DEERFLOW_ACP_E2E_CWD` 指定，默认 `/home/guxy/srv/deerflow`）。
8. **强杀点上的 checkpoint 粒度由 DeerFlow 决定**。`killpg` 是在任意指令边界
   打断进程，桥不参与 checkpoint 写入；恢复到的是 LangGraph 最后一次成功
   持久化的节点，被打断节点内的进展会丢失。桥保证的是「能从 checkpoint
   继续且不与旧 worker 并发」，不是「不丢任何 token」。
9. **每 turn 一次进程启动开销**。worker 需重新 import DeerFlow 与模型 SDK，
   首个事件前有固定延迟。这是换取可终止性的代价，属已知设计取舍。
10. **`InProcessTurnRunner` 没有强制终止能力**。它只服务于把后端对象直接注入
    的单元测试（Python 对象跨不过进程边界）；宽限期超时后只能弃用工作线程。
    **它不在生产路径上**——`deerflow-acp acp` 一律使用 `SubprocessTurnRunner`。

## 回退路径

按影响面从小到大：

| 级别 | 场景 | 动作 |
| --- | --- | --- |
| 1 | 单个 turn 失败 | 客户端收到 `-32603`（脱敏）或 `refusal`；session 仍可用，重发 prompt |
| 2 | 会话状态异常 | `session/close` 后 `session/new`；DeerFlow checkpoint 不受影响 |
| 3 | 桥进程异常 | 重启进程后 `session/resume` 原 sessionId；checkpoint 在 DeerFlow 侧持久化，跨进程可恢复 |
| 4 | 桥整体不可用 | Multica 侧改回直连 DeerFlow HTTP 接口（`http://127.0.0.1:2026`）；桥是旁路组件，不修改 DeerFlow 任何代码或数据 |

第 4 级是本设计的关键性质：桥对 DeerFlow **只读调用**，摘掉桥不需要任何数据迁移
或回滚脚本。

## 来源标识

- 包版本：`deerflow_acp.__version__`
- 分发产物：wheel / sdist 的 `METADATA` 中的 `Version`
- Git 溯源：交付卡记录的 40 位 commit SHA

运行中的桥可通过 `deerflow-acp doctor` 打印版本与后端连通性（输出走 stderr）。
