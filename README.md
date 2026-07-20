# Registry Agent v2.1

面向实验室内部的可扩展 LLM Agent 问答系统。核心公式：

> **Agent = Model + Harness**

模型负责理解与推理；本项目的 Harness 负责 ReAct 循环、工具编排、会话记忆、MCP 技能、安全护栏、可观测性和上下文控制。

v2.1 已在实验室环境完成内部 LLM 和外部 OpenAI-compatible API 的真实运行验证。真实知识库/RAG 仍在建设中，项目保留了稳定接口但不绑定具体实现。

> v1 的早期引擎与实验代码完整保留在 [`_archive/`](./_archive)，只作历史留档，不参与 v2.1 运行。

---

## v2.1 能力一览

| 能力 | 实现 | 说明 |
|---|---|---|
| ReAct 多步执行 | `core/loop.py` | Reason → Act → Observe；最多 8 步防止 Doom Loop；无依赖工具在同一轮异步并发 |
| 原生 tool calls | `tools/__init__.py` | 本地工具、自研工具、MCP 工具统一为 OpenAI function schema |
| 工具路由 | `tools/router.py` | 工具多时按问题和工具描述粗筛候选；未命中时安全回退为全量工具 |
| MCP 技能管理 | `tools/mcp_client.py` | 支持 `streamablehttp` / `sse` / `stdio`；可添加、自动重连、列出和卸载 |
| L1 + L2 记忆 | `core/memory.py` | SQLite 保存完整近期消息；长会话自动压缩早期完整回合为摘要 |
| SSE 流式回答 | `core/model.py` + `web_server.py` | 实时推送思考状态、工具过程和最终 token 文本 |
| 模型可插拔 | `core/model.py` | 默认实验室本地 vLLM/Qwen；可切任意 OpenAI-compatible 服务 |
| 安全护栏 | `guardrails/policy.py` | 下单、支付、删除、外发等副作用操作默认要求确认 |
| 可观测性 | `core/trace.py` | 每步 JSONL trace；成功简洁、失败详报（Silence is Success） |
| 知识库接口 | `knowledge/base.py` | RAG 占位接口，待实验室真实知识库接入后直接替换实现 |
| Web 技能面板 | `web_server.py` | 流式聊天页 + MCP 添加/查看/卸载页，不回显 token/header/env |

---

## 架构

```text
入口层      Gradio Web / FastAPI JSON API / CLI
                       │
内核        ReAct: Reason → Act(并发) → Observe → 收敛
                       │
Harness
  Tool      ToolRegistry：本地 / 同门自研 / MCP 三来源统一注册
  Context   L1 SQLite 最近消息 + L2 历史摘要
  Model     OpenAI-compatible 抽象层：本地默认、外部可覆盖
  Govern    危险操作二次确认
  Observe   JSONL trace + 沉默即成功
  RAG       KnowledgeBase 抽象接口（当前 NullKnowledgeBase 占位）
```

### 一次提问的流程

1. 用户提问进入 `Agent.run()` 或 `Agent.run_stream()`。
2. 工具路由先按 query 粗筛候选 schema；工具少或匹配不可靠时安全回退全量工具。
3. 模型决定直接回答，或返回原生 `tool_calls`。
4. 多个无依赖工具通过 `asyncio.gather` 并发执行。
5. 工具结果以 `tool` 角色消息回注给模型；任务未完成则继续下一轮 ReAct。
6. 最终答案持久化到会话记忆；流式模式还会通过 SSE 逐 token 推给前端。

---

## 目录结构

```text
core/
  loop.py          ReAct 内核、流式事件、L2 摘要触发
  model.py         OpenAI-compatible 模型调用与 stream=True
  memory.py        SQLite L1 消息 + L2 历史摘要
  trace.py         结构化 trace、工具结果摘要

tools/
  __init__.py      ToolRegistry、schema 反射、统一 async 调用
  router.py        工具路由（Context Engineering）
  mcp_client.py    MCP 连接、自动注册、技能管理
  weather.py       内置天气工具
  media.py         内置远程多媒体检索工具

knowledge/base.py  KnowledgeBase / NullKnowledgeBase（RAG 占位）
guardrails/        危险操作确认策略
config/            MCP 配置、运行时 SQLite（运行时文件被 gitignore）
tests/             v2.1 自动回归测试
_archive/          v1 留档（勿删）
web_server.py      FastAPI + Gradio 入口
main.py            CLI 入口
```

---

## 快速开始

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
cp .env.example .env
```

### 模型配置

`.env` 默认接实验室本地 vLLM；换成外部服务只需改三项：

```env
LLM_BASE_URL=http://127.0.0.1:8502/v1
LLM_API_KEY=EMPTY
LLM_MODEL=qwen
```

本地 Qwen 走原生 tool calls 时，vLLM 需开启对应 tool parser，例如：

```text
--enable-auto-tool-choice --tool-call-parser hermes
```

### 上下文控制配置

```env
# 超过多少条未压缩消息时生成 L2 历史摘要
MEMORY_SUMMARY_THRESHOLD=30
# 压缩后至少保留多少条最近完整消息
MEMORY_KEEP_RECENT=12
# 工具数超过该值时才启用关键词候选工具路由
TOOL_ROUTER_MAX_CANDIDATES=12
```

### 启动

```bash
python web_server.py
# Web: http://127.0.0.1:8500

# 或使用 CLI
python main.py
```

---

## API

### 常规对话

```bash
curl -X POST http://127.0.0.1:8500/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"查一下武汉天气","session_id":"demo"}'
```

### SSE 流式对话

```bash
curl -N -X POST http://127.0.0.1:8500/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"查一下武汉天气","session_id":"demo"}'
```

流事件类型：`status`、`tool_call`、`tool_result`、`token`、`final`、`error`。

### 查看工具和 MCP 技能

```bash
curl http://127.0.0.1:8500/api/tools
curl http://127.0.0.1:8500/api/mcp/servers
```

---

## 新增工具

### 同门自研本地工具

在 `tools/` 中写 Python 函数，用 `@register_tool` 注册；系统会从函数签名自动生成 OpenAI schema：

```python
from typing import Annotated
from . import register_tool

@register_tool
def echo(text: Annotated[str, "要返回的文本", True]) -> str:
    """原样返回输入文本。"""
    return text
```

然后在 `load_builtin_tools()` 中 import 该模块即可。

### MCP 外部技能

MCP 工具可在 Web 的「技能管理」页添加，也可调用 API。配置格式对齐官方 `mcpServers`：

| type | 场景 | 关键字段 |
|---|---|---|
| `streamablehttp` | 远程云端 MCP（当前公有服务主流） | `url` + `headers` |
| `sse` | 远程 SSE MCP | `url` + `headers` |
| `stdio` | 本地进程 MCP | `command` + `args` + `env` |

示例：

```bash
curl -X POST http://127.0.0.1:8500/api/mcp/learn \
  -H "Content-Type: application/json" \
  -d '{
        "name": "my-tool",
        "type": "streamablehttp",
        "url": "https://example.com/mcp",
        "headers": {"Authorization": "Bearer <token>"}
      }'
```

系统连接后会调用 MCP 标准 `list_tools()`，将远程工具注册进 ToolRegistry，并把配置持久化到 `config/mcp_servers.json`；下次启动自动重连。多个 MCP 的同名工具会自动命名为 `mcp__<server>__<tool>`，避免覆盖。

卸载：

```bash
curl -X DELETE http://127.0.0.1:8500/api/mcp/servers/my-tool
```

> `.env`、运行时 SQLite、trace 日志和实际 MCP token 都在 `.gitignore` 中；请勿把真实 token 写进 README 或提交到仓库。

---

## 测试

```bash
.venv/Scripts/python.exe -m unittest tests/test_v21_features.py -v
```

v2.1 自动回归测试覆盖：

- 工具路由的相关工具筛选和安全回退；
- L2 摘要、协议安全切分、SQLite 文件释放；
- 流式 ReAct 中 tool call 分片重组、工具执行和最终 token 输出；
- MCP 配置脱敏、按 server 卸载工具和持久化配置。

---

## 当前边界与后续

- [x] 内部 LLM、外部 OpenAI-compatible API 真实运行验证
- [x] ReAct、原生 tool calls、MCP 三传输、流式输出、工具路由、L2 摘要、技能管理面板
- [ ] **实验室真实知识库/RAG 接入**：代码只留 `KnowledgeBase` 接口和 `NullKnowledgeBase` 占位，等待文档处理链路完成后接入

当前优先级是等待并配合真实知识库接入，而不是继续增加框架功能。
