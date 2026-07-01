# _archive · 前人代码留档

本目录保存项目早期（2025 年底 v1 阶段）的代码，**仅作历史留档与对照，不参与 v2 运行**。
保留原因：这些是前期设计的真实记录，体现了当时"一轮式意图解析 + 并发执行"的技术取舍，
对回顾演进思路、撰写项目复盘（如秋招项目介绍）有参考价值。**请勿删除。**

## 文件说明

| 文件 | 来源 | 说明 |
|---|---|---|
| `agent_engine_legacy.py` | 原 `agent_engine.py` 的快照 | v1 核心引擎：两次 LLM 调用（①意图解析抽参数 JSON ②综合工具结果成文），工具调用走手写 JSON 解析而非原生 function calling。v2 已用 `core/loop.py` 的 ReAct 循环替代。 |
| `prompts.py` | 原项目根目录 | `build_system_prompt()` 用于动态拼接工具描述到系统提示词。注意：v1 中该函数实际**从未被 `agent_engine.py` 调用**（引擎内硬编码了 prompt），属未启用代码。v2 的提示词组织在 `core/loop.py` 内。 |
| `sample_agent_test.py` | 原项目根目录 | 更早期的 `FunctionAgent` 实验版，采用三次 LLM 调用（选函数 → 提参数 → 美化回复）。是比 v1 主引擎更原始的探索原型，非正式链路。 |

## v1 → v2 的核心变化

- **一轮式规划 → ReAct 循环**：v1 一次性抽取所有工具再并发；v2 以 Reason→Act→Observe 循环为骨架，并发降级为"循环内无依赖工具的并发优化"。
- **手写 JSON 解析 → 原生 tool_calls**：不再靠 prompt 祈祷模型吐格式，改用推理框架保证结构化的工具调用。
- **绑死本地模型 → 模型抽象层**：本地 Qwen 仍为默认，同时支持覆盖 base_url / api_key / model。
- **无记忆 → SQLite 会话持久化**；**无 MCP → 支持 MCP 工具接入**；新增知识库接口占位、护栏、可观测层。

详见项目根目录 `README.md`。
