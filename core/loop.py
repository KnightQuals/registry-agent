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
import os
from typing import Any, AsyncIterator, Optional

from .model import ModelClient, ModelConfig
from .memory import SessionMemory
from .trace import Tracer, summarize_tool_result
from tools.router import ToolRouter


DEFAULT_SYSTEM_PROMPT = """你是实验室内部智能问答助手。
你可以直接回答问题，也可以调用提供的工具来获取信息或执行操作。

工作方式：
- 需要外部信息或执行动作时，调用合适的工具；可以一次调用多个相互独立的工具。
- 拿到工具结果后，判断是否已足够回答；若不够，可以继续调用工具。
- 得到足够信息后，用自然、准确的中文回答用户，不要编造工具没有返回的内容。
- 工具失败时如实说明，不要假装成功。
"""

MAX_STEPS = 8  # ReAct 循环最大轮数，防止 Doom Loop
MEMORY_SUMMARY_THRESHOLD = int(os.getenv("MEMORY_SUMMARY_THRESHOLD", "30"))
MEMORY_KEEP_RECENT = int(os.getenv("MEMORY_KEEP_RECENT", "12"))


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
        router: Optional[ToolRouter] = None,  # 工具路由：工具变多时按 query 粗筛 schema
    ):
        self.registry = registry
        self.model = model or ModelClient(ModelConfig.from_env())
        self.memory = memory or SessionMemory()
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.guardrail = guardrail
        self.knowledge = knowledge
        self.router = router or ToolRouter()

    def _build_messages(self, session_id: str) -> list[dict]:
        # memory.load 会自动把 L2 历史摘要放在最近 L1 消息之前。
        history = self.memory.load(session_id, include_summary=True)
        return [{"role": "system", "content": self.system_prompt}] + history

    async def _summarize_history(self, prompt: str) -> str:
        """给 SessionMemory 注入的摘要器：用同一模型生成 L2 工作记忆，不提供工具。"""
        response = await self.model.chat(
            [
                {"role": "system", "content": "你是严谨的对话记忆压缩器，只输出摘要本身。"},
                {"role": "user", "content": prompt},
            ],
            tools=None,
            temperature=0.1,
        )
        return response.choices[0].message.content or ""

    async def _compact_memory_if_needed(self, session_id: str, tracer: Tracer) -> None:
        """上下文超阈值时尝试生成 L2 摘要；失败不影响本轮正常对话。"""
        try:
            compacted = await self.memory.summarize_if_needed(
                session_id,
                self._summarize_history,
                max_messages=MEMORY_SUMMARY_THRESHOLD,
                keep_recent=MEMORY_KEEP_RECENT,
            )
            if compacted:
                tracer.log("memory", "l2_summary", {"threshold": MEMORY_SUMMARY_THRESHOLD})
        except Exception as e:  # noqa: BLE001
            tracer.log("memory", "l2_summary", {"error": str(e)}, ok=False)

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
        # 新一轮开始前，必要时压缩已结束的早期历史，控制后续上下文大小。
        await self._compact_memory_if_needed(session_id, tracer)

        for step in range(1, self.max_steps + 1):
            messages = self._build_messages(session_id)

            # Context Engineering：工具少时全给；工具多时只把与原问题相关的候选 schema 给模型。
            # route 内置安全回退，不会因为关键词漏匹配而把全部能力藏掉。
            route = self.router.select(user_input, self.registry)
            tools_schema = self.registry.openai_schema(route.names)
            tracer.log(
                "reason",
                f"step_{step}",
                {
                    "n_tools": len(tools_schema),
                    "tool_names": route.names,
                    "router": route.reason,
                    "router_fallback": route.used_fallback,
                },
            )

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

    async def run_stream(
        self,
        session_id: str,
        user_input: str,
        to_console: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        SSE/网页流式入口。

        yield 的统一事件格式：
        - status      : ReAct 正在思考 / 调工具 / 观察结果
        - token       : 最终回答的一个增量文本片段（真实模型流，不是本地假切字）
        - tool_call   : 某个工具准备执行
        - tool_result : 某个工具执行完成（仅摘要，完整细节留 trace）
        - final/error : 本轮终态

        工具回合与最终文本回合共用一个 OpenAI streaming API：若 chunk 中出现 tool_calls，
        本方法会把分段的工具名/参数拼回完整调用；若没有 tool_calls，则 content token 直接向前端推送。
        """
        tracer = Tracer(session_id, to_console=to_console)
        self.memory.append(session_id, "user", content=user_input)
        tracer.log("system", "user_input", {"text": user_input[:200], "stream": True})
        await self._compact_memory_if_needed(session_id, tracer)

        for step in range(1, self.max_steps + 1):
            messages = self._build_messages(session_id)
            route = self.router.select(user_input, self.registry)
            tools_schema = self.registry.openai_schema(route.names)
            trace_detail = {
                "n_tools": len(tools_schema),
                "tool_names": route.names,
                "router": route.reason,
                "router_fallback": route.used_fallback,
                "stream": True,
            }
            tracer.log("reason", f"step_{step}", trace_detail)
            yield {
                "event": "status",
                "data": {"step": step, "message": "正在分析下一步…", "tools": route.names},
            }

            # index -> OpenAI 标准 tool_call 字典；流式参数会分多个 chunk 到达，需累加。
            tool_buffers: dict[int, dict] = {}
            answer_parts: list[str] = []
            saw_tool_call = False

            try:
                async for chunk in self.model.stream_chat(messages, tools=tools_schema or None):
                    choices = getattr(chunk, "choices", None) or []
                    if not choices:
                        continue
                    delta = getattr(choices[0], "delta", None)
                    if delta is None:
                        continue

                    for raw_call in getattr(delta, "tool_calls", None) or []:
                        saw_tool_call = True
                        index = getattr(raw_call, "index", 0) or 0
                        entry = tool_buffers.setdefault(
                            index,
                            {
                                "id": None,
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if getattr(raw_call, "id", None):
                            entry["id"] = raw_call.id
                        function = getattr(raw_call, "function", None)
                        if function is not None:
                            if getattr(function, "name", None):
                                entry["function"]["name"] += function.name
                            if getattr(function, "arguments", None):
                                entry["function"]["arguments"] += function.arguments

                    content = getattr(delta, "content", None)
                    if content:
                        answer_parts.append(content)
                        # OpenAI 协议里工具调用和最终文本通常互斥；最终文本可直接真实流式转发。
                        if not saw_tool_call:
                            yield {"event": "token", "data": {"text": content}}
            except Exception as e:  # noqa: BLE001
                tracer.log("error", "stream_model", {"error": str(e)}, ok=False)
                yield {"event": "error", "data": {"message": f"模型流式调用失败：{e}"}}
                return

            if not tool_buffers:
                answer = "".join(answer_parts)
                self.memory.append(session_id, "assistant", content=answer)
                tracer.log("final", f"step_{step}", {"chars": len(answer), "stream": True})
                yield {"event": "final", "data": {"answer": answer, "step": step}}
                return

            # 工具回合：把模型分片返回的调用重组、写入协议历史，然后循环内并发执行。
            tool_calls = []
            for index, call in sorted(tool_buffers.items()):
                call["id"] = call["id"] or f"stream-call-{step}-{index}"
                tool_calls.append(call)
            self.memory.append(session_id, "assistant", content=None, tool_calls=tool_calls)

            async def _exec_stream(call: dict) -> tuple[str, str, bool, str]:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                tracer.log("tool_call", name, {"args": args, "stream": True})
                yield_event = {"event": "tool_call", "data": {"name": name, "arguments": args}}
                # gather 内不能直接 yield，因此调用事件由外层在创建任务前发出。
                ok, observation = await self._run_tool(tracer, name, args)
                return call["id"], name, ok, observation

            parsed_calls = []
            for call in tool_calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                yield {"event": "tool_call", "data": {"name": name, "arguments": args}}
                parsed_calls.append(call)

            results = await asyncio.gather(*[_exec_stream(call) for call in parsed_calls])
            for call_id, name, ok, observation in results:
                self.memory.append(session_id, "tool", content=observation, name=name)
                self._patch_last_tool_call_id(session_id, call_id)
                yield {"event": "tool_result", "data": {"name": name, "ok": ok}}

            yield {
                "event": "status",
                "data": {"step": step, "message": "工具结果已返回，正在继续判断…"},
            }

        fallback = "抱歉，本次任务较复杂，我在允许的步数内没能完全完成。请拆分问题或补充信息后再试。"
        self.memory.append(session_id, "assistant", content=fallback)
        tracer.log("error", "max_steps_reached", {"max": self.max_steps, "stream": True}, ok=False)
        yield {"event": "final", "data": {"answer": fallback, "step": self.max_steps, "truncated": True}}

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
