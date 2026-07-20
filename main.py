"""
main.py — 命令行入口（v2）

用 ReAct Agent 内核跑一个交互式命令行会话。
启动时：加载内置工具 + 已学会的 MCP server + 知识库工具，然后进入对话循环。
"""

from __future__ import annotations

import asyncio
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

from core import Agent, ModelClient, ModelConfig, SessionMemory  # noqa: E402
from tools import registry, load_builtin_tools                    # noqa: E402
from tools.mcp_client import MCPManager                           # noqa: E402
from knowledge import NullKnowledgeBase, register_as_tool         # noqa: E402
from guardrails import Guardrail                                  # noqa: E402


async def bootstrap() -> tuple[Agent, MCPManager]:
    """组装 Agent：工具 + MCP + 知识库 + 护栏。"""
    load_builtin_tools()

    # 知识库（当前占位）注册为一个工具
    register_as_tool(registry, NullKnowledgeBase())

    # 加载已学会的 MCP server
    mcp = MCPManager(registry)
    try:
        loaded = await mcp.load_all_from_config()
        if loaded:
            print(f"[mcp] 已加载 MCP server: {loaded}")
    except Exception as e:  # noqa: BLE001
        print(f"[mcp] 加载 MCP 配置时出错（可忽略）：{e}")

    agent = Agent(
        registry=registry,
        model=ModelClient(ModelConfig.from_env()),
        memory=SessionMemory(),
        guardrail=Guardrail(),  # 护栏默认开
    )
    return agent, mcp


async def main() -> None:
    agent, mcp = await bootstrap()
    session_id = "cli-" + uuid.uuid4().hex[:8]

    print("=" * 46)
    print("Registry Agent v2.1（ReAct + Harness + Stream）已启动")
    print(f"   {agent.model.describe()}")
    print(f"   已加载工具: {[t['name'] for t in registry.list_tools()]}")
    print("   输入 exit / quit / 退出 结束")
    print("=" * 46)

    try:
        while True:
            try:
                user_input = input("\n👤 你: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "退出"):
                break
            print("\nAI: ", end="", flush=True)
            emitted_token = False
            async for event in agent.run_stream(session_id, user_input):
                event_type, data = event["event"], event["data"]
                if event_type == "status":
                    print(f"\n[状态] {data.get('message', '')}", file=sys.stderr)
                elif event_type == "tool_call":
                    print(f"\n[工具] 调用 {data.get('name', '')}", file=sys.stderr)
                elif event_type == "tool_result":
                    print(f"\n[工具] 完成 {data.get('name', '')}", file=sys.stderr)
                elif event_type == "token":
                    emitted_token = True
                    print(data.get("text", ""), end="", flush=True)
                elif event_type == "final" and not emitted_token:
                    print(data.get("answer", ""), end="", flush=True)
                elif event_type == "error":
                    print(f"\n[错误] {data.get('message', '')}", file=sys.stderr)
            print()
    finally:
        await mcp.close()


if __name__ == "__main__":
    asyncio.run(main())
