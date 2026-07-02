# Registry Agent v2

面向实验室内部的 LLM Agent 问答系统。v2 基于 2026 主流 Harness 认知重制：
从 v1 的「一轮式意图解析 + 并发执行」升级为 **ReAct 循环内核 + 可插拔 Harness 各层**，
核心公式 **Agent = Model + Harness**。

> v1 的早期代码（一轮式引擎、实验原型）完整保留在 [`_archive/`](./_archive)，仅作历史留档，不参与运行。

---

## 为什么重制（v1 → v2）

| 维度 | v1（2025 底） | v2（现在） |
|---|---|---|
| 执行模型 | 一轮抽全部工具 → 并发跑 → 汇总（不看结果就收工） | **ReAct 循环**：Reason→Act→Observe，看结果再决定下一步 |
| 并发 | 架构主干 | 降级为**循环内优化**（无依赖工具仍并发；简单问题 1 轮即出，延迟不变） |
| 工具调用 | 手写 JSON 解析（prompt 祈祷模型听话，易碎） | **原生 tool_calls**（框架保证结构；也是接 MCP 的前提） |
| 工具来源 | 本地函数 | 本地函数 + 同门自研 + **MCP server（喂 URL 就学会）** |
| 模型 | 绑死本地 qwen:8502 | **模型抽象层**：本地默认，可覆盖 base_url/key/model 切任意 OpenAI 兼容服务 |
| 记忆 | 每请求 new，历史当场丢 | **SQLite 会话持久化** |
| 知识库 | 无 | **RAG 接口占位**，接入即生效 |
| 可观测 / 护栏 | 无 | **每步 trace + 沉默即成功**；**危险操作二次确认** |

---

## 架构

```
入口层        Web(Gradio) · JSON API(FastAPI) · CLI
                         │
内核 (L)      ReAct 主循环  Reason → Act(并发) → Observe → Verify   [core/loop.py]
                         │
Harness 各层
  Tool (T)    ToolRegistry：本地 / 自研 / MCP 统一注册            [tools/]
  Context(C)  SQLite 会话记忆                                     [core/memory.py]
  RAG         知识库接口（占位，接入即生效）                       [knowledge/]
  Model (E)   模型抽象层（本地默认 · 可覆盖）                       [core/model.py]
  Govern(G)   护栏：危险操作二次确认                                [guardrails/]
  Observe(O)  结构化 trace + 沉默即成功                            [core/trace.py]
```

## 目录结构

```
core/        loop.py(ReAct内核) model.py(模型层) memory.py(会话记忆) trace.py(可观测)
tools/       __init__.py(ToolRegistry) mcp_client.py(MCP客户端) weather.py media.py
knowledge/   base.py(RAG接口+占位)
guardrails/  policy.py(护栏)
config/      mcp_servers.json(已学会的MCP) .env
_archive/    v1 前人代码留档（勿删）
web_server.py  main.py
```

## 快速开始

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
cp .env.example .env        # 默认接实验室本地 vLLM，可改 LLM_BASE_URL 等切模型
```

启动：

```bash
python web_server.py        # Web http://127.0.0.1:8500 ，API /api/chat
# 或
python main.py              # 命令行
```

## 让它「学会」一个 MCP 技能

系统可以吃 MCP，配置格式对齐 MCP 官方 `mcpServers`，支持三种传输：

| type | 场景 | 关键字段 |
|---|---|---|
| `streamablehttp` | 远程云端 MCP（当前公有服务最主流，如瑞幸点餐） | `url` + `headers` |
| `sse` | 远程 SSE MCP | `url` + `headers` |
| `stdio` | 本地起子进程 | `command` + `args` + `env` |

**远程（Streamable HTTP，对应瑞幸给的接入方式）：**

```bash
curl -X POST http://127.0.0.1:8500/api/mcp/learn \
  -H "Content-Type: application/json" \
  -d '{
        "name": "my-coffee",
        "type": "streamablehttp",
        "url": "https://gwmcp.lkcoffee.com/order/user/mcp",
        "headers": {"Authorization": "Bearer <登录后复制的Token>"}
      }'
```

**本地进程（stdio）：**

```bash
curl -X POST http://127.0.0.1:8500/api/mcp/learn \
  -H "Content-Type: application/json" \
  -d '{"name":"some-tool","type":"stdio","command":"npx","args":["some-mcp"]}'
```

系统会连上该 server、拉取它暴露的全部工具、注册进工具表，并写入 `config/mcp_servers.json`，
下次启动自动加载。之后模型即可通过 tool_calls 调用这些新技能。查看已装工具：`GET /api/tools`。

> 配置结构与官方一致：厂商给你的 `mcpServers.<名字>` 配置体，加个 `name` 字段平铺进来即可。

## 新增本地工具（同门自研）

在 `tools/` 下写个函数并用 `@register_tool` 装饰，`Annotated` 描述参数即可：

```python
from typing import Annotated
from . import register_tool

@register_tool
def echo(text: Annotated[str, "要返回的文本", True]) -> str:
    """原样返回输入文本。"""
    return text
```

然后在 `tools/load_builtin_tools()` 里 import 该模块即可自动注册。

## 配置项（.env）

| 变量 | 说明 | 默认 |
|---|---|---|
| `LLM_BASE_URL` | OpenAI 兼容服务地址（实验室本地 vLLM / Claude / GPT / 中转站） | `http://127.0.0.1:8502/v1` |
| `LLM_API_KEY` | API Key（本地 vLLM 通常填 EMPTY） | `EMPTY` |
| `LLM_MODEL` | 模型名 | `qwen` |
| `SENIVERSE_API_KEY` | 心知天气（weather 工具） | 无 |
| `REMOTE_MEDIA_SERVICE_URL` | 媒体检索服务（media 工具） | `http://127.0.0.1:8600/api/search` |

## 路线图

- [x] ReAct 内核 + 原生 tool_calls
- [x] 模型抽象层 / SQLite 记忆 / trace / 护栏
- [x] MCP 客户端（喂 URL 就学会）
- [ ] 接入实验室内部知识库（当前为占位实现）
- [ ] SSE 流式输出
- [ ] 工具路由（工具量大时按需筛选 schema）
- [ ] 会话历史摘要（L2 记忆）
