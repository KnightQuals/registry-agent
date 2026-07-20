"""registry-agent v2.1 关键能力回归测试（stdlib unittest，无需额外测试框架）。"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.loop import Agent
from core.memory import SessionMemory
from tools import ToolRegistry
from tools.mcp_client import MCPManager
from tools.router import ToolRouter


def stream_chunk(content=None, tool_calls=None):
    return NS(choices=[NS(delta=NS(content=content, tool_calls=tool_calls or []))])


def stream_tool_call(index, call_id=None, name=None, arguments=None):
    return NS(index=index, id=call_id, function=NS(name=name, arguments=arguments))


class RouterTests(unittest.TestCase):
    def test_router_selects_related_tool_and_has_safe_fallback(self):
        registry = ToolRegistry()

        @registry.register
        def get_weather(city_name: str):
            """查询指定城市的实时天气和未来天气预报。"""
            return city_name

        @registry.register
        def search_media(query: str):
            """检索实验室服务器中的图片、视频、多媒体文件。"""
            return query

        async def unused(_):
            return ""

        for i in range(15):
            registry.register_mcp_tool(
                f"mcp__demo__tool_{i}", "无关测试工具", {"type": "object", "properties": {}},
                unused, source="mcp:demo",
            )

        router = ToolRouter(max_candidates=3)
        self.assertIn("get_weather", router.select("查武汉天气", registry).names)
        fallback = router.select("zz_not_a_tool_987", registry)
        self.assertTrue(fallback.used_fallback)
        self.assertEqual(len(fallback.names), 17)


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_l2_summary_preserves_recent_messages(self):
        with tempfile.TemporaryDirectory() as temp:
            db = os.path.join(temp, "memory.sqlite3")
            memory = SessionMemory(db)
            for i in range(4):
                memory.append("session", "user", f"问题{i}")
                memory.append("session", "assistant", f"回答{i}")

            async def summarizer(prompt: str) -> str:
                self.assertIn("问题0", prompt)
                return "已完成早期问题，保留后续上下文。"

            changed = await memory.summarize_if_needed(
                "session", summarizer, max_messages=6, keep_recent=3
            )
            self.assertTrue(changed)
            messages = memory.load("session")
            self.assertEqual(messages[0]["role"], "system")
            self.assertIn("已完成早期问题", messages[0]["content"])
            self.assertEqual(len(messages), 5)  # 1 L2 摘要 + 4 最近 L1 消息
            memory.clear("session")
            self.assertEqual(memory.load("session"), [])


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_react_reassembles_tool_call_then_streams_answer(self):
        class FakeModel:
            def __init__(self):
                self.turn = 0

            async def stream_chat(self, messages, tools=None):
                self.turn += 1
                if self.turn == 1:
                    # 模拟真实 API 将函数参数拆成多个 chunk。
                    yield stream_chunk(tool_calls=[stream_tool_call(0, "call_1", "echo", '{"text":')])
                    yield stream_chunk(tool_calls=[stream_tool_call(0, None, None, '"测试"}')])
                else:
                    yield stream_chunk(content="工具")
                    yield stream_chunk(content="完成")

        async def echo(text: str):
            return f"回声:{text}"

        with tempfile.TemporaryDirectory() as temp:
            registry = ToolRegistry()
            registry.register(echo)
            agent = Agent(
                registry=registry,
                model=FakeModel(),
                memory=SessionMemory(os.path.join(temp, "memory.sqlite3")),
                max_steps=3,
            )
            events = [event async for event in agent.run_stream("s", "请测试", to_console=False)]
            self.assertEqual(
                [event["event"] for event in events],
                ["status", "tool_call", "tool_result", "status", "status", "token", "token", "final"],
            )
            self.assertEqual(
                "".join(event["data"]["text"] for event in events if event["event"] == "token"),
                "工具完成",
            )
            loaded = agent.memory.load("s", include_summary=False)
            self.assertTrue(any(m["role"] == "tool" and m.get("tool_call_id") == "call_1" for m in loaded))


class MCPManagementTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_redacts_secrets_and_remove_unloads_source_tools(self):
        with tempfile.TemporaryDirectory() as temp:
            config = os.path.join(temp, "mcp_servers.json")
            with open(config, "w", encoding="utf-8") as f:
                json.dump({"mcpServers": {"demo": {
                    "type": "streamablehttp", "url": "https://example.com/mcp",
                    "headers": {"Authorization": "Bearer secret"},
                }}}, f)

            registry = ToolRegistry()

            async def fake_call(_):
                return "ok"

            registry.register_mcp_tool(
                "mcp__demo__search", "搜索", {"type": "object", "properties": {}}, fake_call,
                source="mcp:demo", original_name="search",
            )
            manager = MCPManager(registry, config_path=config)
            listed = manager.list_servers()
            self.assertTrue(listed[0]["config"]["has_headers"])
            self.assertNotIn("headers", listed[0]["config"])
            self.assertEqual(await manager.remove("demo"), ["mcp__demo__search"])
            self.assertFalse(registry.has("mcp__demo__search"))
            self.assertEqual(manager.list_servers(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
