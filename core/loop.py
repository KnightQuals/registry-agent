"""
core/loop.py — ReAct 主循环（Lifecycle 内核）

这是 v2 的心脏，替代 v1 的"两次 LLM 调用"（意图解析 + 综合）。

核心决策（对应本轮重制）：
- 以 ReAct 循环为骨架：Reason（模型决定下一步）→ Act（原生 tool_calls）
  → Observe（结果回注上下文）→ 直到模型不再要工具（得出最终答案）或触顶。
- 保留 v1 的"并发"优势，但降级为**循环内优化**：模型在一轮里要多个工具时，
  这些工具（无依赖）并发执行；简单单工具问题 1 轮即结束，延迟与一轮式相当。
- 全程 async：MCP 工具、并发、未来的流式输出都天然异步。
- 每步落 trace；工具结果走"沉默即成功"摘要；失败可触发模型自我修正。

依赖：core/model.ModelClient、tools.ToolRegistry、core/memory.SessionMemory、
      core/trace.Tracer、guardrails/policy（护栏）。
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from .model import ModelClient, ModelConfig
from .memory import SessionMemory
from .trace import Tracer, summarize_tool_result


DEFAULT_SYSTEM_PROMPT = """你是实验室内部智能问答助手。
你可以直接回答问题，也可以调用提供的工具来获取信息或执行操作。

工作方式：
- 需要外部信息或执行动作时，调用合适的工具；可以一次调用多个相互独立的工具。
- 拿到工具结果后，判断是否已足够回答；若不够，可以继续调用工具。
- 得到足够信息后，用自然、准确的中文回答用户，不要编造工具没有返回的内容。
- 工具失败时如实说明，不要假装成功。
"""

MAX_STEPS = 8  # ReAct 循环最大轮数，防止 Doom Loop


class Agent:
    """ReAct Agent 内核。"""

    def __init__(
        self,
        registry,                      # tools.ToolRegistry
        model: Optional[ModelClient] = None,
        memory: Optional[SessionMemory] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = MAX_STEPS,
        guardrail=None,                # guardrails.policy.Guardrail，可选
        knowledge=None,                # knowledge.base.KnowledgeBase，可选（当前占位）
    ):
        self.registry = registry
        self.model = model or ModelClient(ModelConfig.from_env())
        self.memory = memory or SessionMemory()
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.guardrail = guardrail
        self.knowledge = knowledge

    def _build_messages(self, session_id: str) -> list[dict]:
        history = self.memory.load(session_id)
        return [{"role": "system", "content": self.system_prompt}] + history

    async def _run_tool(self, tracer: Tracer, name: str, arguments: dict) -> tuple[bool, str]:
        """执行单个工具，返回 (ok, observation_text)。含护栏检查。"""
        # 护栏：危险操作二次确认（MCP 外部工具 / 下单类）
        if self.guardrail is not None:
            allowed, reason = self.guardrail.check(name, arguments)
            if not allowed:
                tracer.log("tool_result", name, {"blocked": reason}, ok=False)
                return False, f"操作被安全策略拦截：{reason}"
        try:
            result = await self.registry.call(name, arguments)
            ok = True
        except Exception as e:  # noqa: BLE001
            result, ok = str(e), False
        tracer.log("tool_result", name, {"ok": ok}, ok=ok)
        return ok, summarize_tool_result(name, ok, result)

    async def run(self, session_id: str, user_input: str, to_console: bool = True) -> str:
        """
        跑一轮完整的 ReAct 对话。返回最终自然语言答案。
        """
        tracer = Tracer(session_id, to_console=to_console)
        self.memory.append(session_id, "user", content=user_input)
        tracer.log("system", "user_input", {"text": user_input[:200]})

        tools_schema = self.registry.openai_schema()

        for step in range(1, self.max_steps + 1):
            messages = self._build_messages(session_id)
            tracer.log("reason", f"step_{step}", {"n_tools": len(tools_schema)})

            resp = await self.model.chat(messages, tools=tools_schema or None)
            msg = resp.choices[0].message

            tool_calls = getattr(msg, "tool_calls", None)

            # 无工具调用 → 模型给出了最终答案，循环结束
            if not tool_calls:
                answer = msg.content or ""
                self.memory.append(session_id, "assistant", content=answer)
                tracer.log("final", f"step_{step}", {"chars": len(answer)})
                return answer

            # 有工具调用：先把 assistant 的 tool_calls 消息入历史（协议要求）
            self.memory.append(
                session_id, "assistant", content=msg.content,
                tool_calls=[tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in tool_calls],
            )

            # 循环内并发执行所有（无依赖的）工具调用
            async def _exec(tc):
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                tracer.log("tool_call", name, {"args": args})
                ok, observation = await self._run_tool(tracer, name, args)
                return tc.id, name, observation

            results = await asyncio.gather(*[_exec(tc) for tc in tool_calls])

            # 把每个工具结果作为 tool 角色消息回注（Observe）
            for tc_id, name, observation in results:
                self.memory.append(
                    session_id, "tool", content=observation, name=name,
                )
                # tool 消息需要 tool_call_id 关联；记在 content 里也可，这里补一条元信息
                self._patch_last_tool_call_id(session_id, tc_id)

        # 触顶仍未收敛：给出兜底回答
        fallback = "抱歉，本次任务较复杂，我在允许的步数内没能完全完成。请拆分问题或补充信息后再试。"
        self.memory.append(session_id, "assistant", content=fallback)
        tracer.log("error", "max_steps_reached", {"max": self.max_steps}, ok=False)
        return fallback

    def _patch_last_tool_call_id(self, session_id: str, tc_id: str) -> None:
        """给刚写入的 tool 消息补 tool_call_id（OpenAI 协议需要）。"""
        with self.memory._conn() as conn:  # noqa: SLF001（内部协作，可接受）
            row = conn.execute(
                "SELECT id, tool_calls FROM messages WHERE session_id=? AND role='tool' "
                "ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE messages SET tool_calls=? WHERE id=?",
                    (json.dumps({"tool_call_id": tc_id}), row["id"]),
                )
