"""
web_server.py — Web + JSON API 入口（v2）

保留 v1 的三入口能力，适配 async ReAct 内核：
- FastAPI  : POST /api/chat 结构化对话；GET /api/tools 查看已装工具；
             POST /api/mcp/learn 让系统"学会"一个 MCP server（对话外的管理入口）。
- Gradio   : Web 聊天界面（含一个简单的技能管理说明）。
两个入口共享同一个 Agent 与 ToolRegistry。
"""

from __future__ import annotations

import uuid

import gradio as gr
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

from core import Agent, ModelClient, ModelConfig, SessionMemory  # noqa: E402
from tools import registry, load_builtin_tools                    # noqa: E402
from tools.mcp_client import MCPManager                           # noqa: E402
from knowledge import NullKnowledgeBase, register_as_tool         # noqa: E402
from guardrails import Guardrail                                  # noqa: E402

SERVER_PORT = 8500

# ---- 全局单例：工具/MCP/Agent ----
mcp_manager = MCPManager(registry)
agent: Agent | None = None


app = FastAPI(title="Registry Agent v2 · API")


@app.on_event("startup")
async def _startup() -> None:
    global agent
    load_builtin_tools()
    register_as_tool(registry, NullKnowledgeBase())
    try:
        loaded = await mcp_manager.load_all_from_config()
        if loaded:
            print(f"[mcp] 已加载: {loaded}")
    except Exception as e:  # noqa: BLE001
        print(f"[mcp] 加载配置出错（可忽略）：{e}")
    agent = Agent(
        registry=registry,
        model=ModelClient(ModelConfig.from_env()),
        memory=SessionMemory(),
        guardrail=Guardrail(),
    )
    print(f"[startup] {agent.model.describe()}；工具：{[t['name'] for t in registry.list_tools()]}")


# ---------- API 模型 ----------
class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


class LearnMCPRequest(BaseModel):
    name: str
    command: str
    args: list[str] = []


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """结构化对话入口。"""
    session_id = req.session_id or ("api-" + uuid.uuid4().hex[:8])
    try:
        answer = await agent.run(session_id, req.query, to_console=False)
        return {"status": "success", "session_id": session_id, "answer": answer}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": str(e)}


@app.get("/api/tools")
async def api_tools():
    """查看当前已装工具清单（技能管理）。"""
    return {"tools": registry.list_tools()}


@app.post("/api/mcp/learn")
async def api_learn_mcp(req: LearnMCPRequest):
    """
    让系统"学会"一个 MCP server（例如瑞幸点餐）。
    body: {"name":"luckin","command":"npx","args":["luckin-coffee-mcp"]}
    """
    try:
        learned = await mcp_manager.learn_stdio_server(req.name, req.command, req.args)
        return {"status": "success", "server": req.name, "learned_tools": learned}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": str(e)}


# ---------- Gradio Web ----------
async def chat_function(message, history):
    session_id = "web-shared"  # Web 界面用一个共享会话；如需隔离可按用户生成
    try:
        return await agent.run(session_id, message, to_console=False)
    except Exception as e:  # noqa: BLE001
        return f"❌ 出错：{e}"


def build_web():
    return gr.ChatInterface(
        fn=chat_function,
        title="🤖 Registry Agent v2（ReAct + Harness）",
        description="支持工具调用 / MCP 技能 / 知识库检索。用 POST /api/mcp/learn 可让它学会新的 MCP 技能。",
        textbox=gr.Textbox(placeholder="请输入问题...", container=False, scale=7),
    )


def main():
    print("=" * 46)
    print("🚀 Registry Agent v2 启动 (Web + JSON API)")
    print(f"👉 Web : http://0.0.0.0:{SERVER_PORT}")
    print(f"👉 API : http://0.0.0.0:{SERVER_PORT}/api/chat")
    print(f"👉 学技能: POST http://0.0.0.0:{SERVER_PORT}/api/mcp/learn")
    print("=" * 46)
    demo = build_web()
    mounted = gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(mounted, host="0.0.0.0", port=SERVER_PORT)


if __name__ == "__main__":
    main()
