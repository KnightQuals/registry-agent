"""
web_server.py — Web + JSON API 入口（v2）

保留 v1 的三入口能力，适配 async ReAct 内核：
- FastAPI  : POST /api/chat 结构化对话；GET /api/tools 查看已装工具；
             POST /api/mcp/learn 让系统"学会"一个 MCP server（对话外的管理入口）。
- Gradio   : Web 聊天界面（含一个简单的技能管理说明）。
两个入口共享同一个 Agent 与 ToolRegistry。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

import gradio as gr
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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
    """
    学习一个 MCP server。字段对齐官方 mcpServers 配置，支持三种传输：
    - streamablehttp / sse : type + url + headers（如瑞幸远程 MCP）
    - stdio                : type + command + args + env（本地进程）
    可直接把厂商给的 mcpServers 里某个 server 的配置体贴进来。
    """
    name: str
    type: str = "stdio"
    url: str | None = None
    headers: dict = Field(default_factory=dict)
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict = Field(default_factory=dict)

    def to_spec(self) -> dict:
        if self.type in ("streamablehttp", "streamable_http", "http", "sse"):
            return {"type": self.type, "url": self.url, "headers": self.headers}
        return {"type": "stdio", "command": self.command, "args": self.args, "env": self.env}


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """结构化对话入口（等待完整结果后一次返回，兼容已有客户端）。"""
    session_id = req.session_id or ("api-" + uuid.uuid4().hex[:8])
    try:
        answer = await agent.run(session_id, req.query, to_console=False)
        return {"status": "success", "session_id": session_id, "answer": answer}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": str(e)}


def _encode_sse(event: dict) -> str:
    """把 Agent 统一事件转成标准 SSE 帧。"""
    return f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"


@app.post("/api/chat/stream")
async def api_chat_stream(req: ChatRequest):
    """
    SSE 流式对话入口。

    返回 event-stream，事件依次可能为：status / tool_call / tool_result / token / final / error。
    前端可一边显示 Agent 正在做什么，一边逐 token 显示最终答案。
    """
    session_id = req.session_id or ("api-" + uuid.uuid4().hex[:8])

    async def event_source() -> AsyncIterator[str]:
        try:
            async for event in agent.run_stream(session_id, req.query, to_console=False):
                yield _encode_sse(event)
        except Exception as e:  # noqa: BLE001
            yield _encode_sse({"event": "error", "data": {"message": str(e)}})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/tools")
async def api_tools():
    """查看当前已装工具清单（技能管理）。"""
    return {"tools": registry.list_tools()}


@app.post("/api/mcp/learn")
async def api_learn_mcp(req: LearnMCPRequest):
    """
    让系统"学会"一个 MCP server。示例：
    - 远程(瑞幸这种)：{"name":"my-coffee","type":"streamablehttp",
                      "url":"https://gwmcp.lkcoffee.com/order/user/mcp",
                      "headers":{"Authorization":"Bearer <token>"}}
    - 本地进程：      {"name":"luckin","type":"stdio","command":"npx","args":["some-mcp"]}
    """
    try:
        learned = await mcp_manager.learn(req.name, req.to_spec())
        return {"status": "success", "server": req.name, "learned_tools": learned}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": str(e)}


@app.get("/api/mcp/servers")
async def api_list_mcp_servers():
    """列出已学习的 MCP server（配置中的 token/header 会自动脱敏）。"""
    return {"servers": mcp_manager.list_servers()}


@app.delete("/api/mcp/servers/{name}")
async def api_remove_mcp_server(name: str):
    """卸载指定 MCP server：关闭连接、移除其工具、删除持久化配置。"""
    known = {server["name"] for server in mcp_manager.list_servers()}
    if name not in known:
        raise HTTPException(status_code=404, detail=f"未找到 MCP server: {name}")
    removed = await mcp_manager.remove(name, persist=True)
    return {"status": "success", "server": name, "removed_tools": removed}


# ---------- Gradio Web ----------
def _tool_rows() -> list[list[str]]:
    return [
        [tool["name"], tool["kind"], tool["source"], tool["original_name"], tool["description"]]
        for tool in registry.list_tools()
    ]


def _server_rows() -> list[list[str]]:
    rows = []
    for server in mcp_manager.list_servers():
        cfg = server["config"]
        rows.append([
            server["name"],
            "已连接" if server["connected"] else "未连接",
            cfg.get("type", "stdio"),
            str(server["tool_count"]),
            cfg.get("url", cfg.get("command", "")),
        ])
    return rows


def _parse_json(value: str, expected_type: type, field_name: str):
    """技能面板中的 JSON 文本解析；空文本返回该类型的空值。"""
    value = (value or "").strip()
    if not value:
        return {} if expected_type is dict else []
    parsed = json.loads(value)
    if not isinstance(parsed, expected_type):
        raise ValueError(f"{field_name} 必须是合法的 JSON {expected_type.__name__}")
    return parsed


async def chat_stream_ui(message: str, history: list | None, session_id: str):
    """Gradio 聊天页：将 Agent 的 status/token/tool 事件边跑边渲染。"""
    if not message or not message.strip():
        yield history or [], "", "等待输入", session_id
        return
    session_id = session_id or ("web-" + uuid.uuid4().hex[:8])
    history = list(history or [])
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    status = "正在启动 ReAct 循环…"
    yield history, "", status, session_id

    try:
        async for event in agent.run_stream(session_id, message, to_console=False):
            event_type, data = event["event"], event["data"]
            if event_type == "status":
                status = data.get("message", "正在处理…")
            elif event_type == "tool_call":
                status = f"正在调用工具：{data.get('name', '')}"
            elif event_type == "tool_result":
                status = f"工具完成：{data.get('name', '')}"
            elif event_type == "token":
                history[-1]["content"] += data.get("text", "")
            elif event_type == "final":
                history[-1]["content"] = data.get("answer", history[-1]["content"])
                status = "已完成"
            elif event_type == "error":
                history[-1]["content"] += "\n\n[错误] " + data.get("message", "未知错误")
                status = "执行出错"
            yield history, "", status, session_id
    except Exception as e:  # noqa: BLE001
        history[-1]["content"] += f"\n\n[错误] {e}"
        yield history, "", "执行出错", session_id


async def clear_chat_ui(session_id: str):
    if session_id and agent is not None:
        agent.memory.clear(session_id)
    return [], "", "会话已清空", session_id


async def learn_mcp_ui(
    name: str,
    transport: str,
    url: str,
    headers_text: str,
    command: str,
    args_text: str,
    env_text: str,
):
    """技能管理面板：解析表单、学习 MCP、刷新已装工具与 server 列表。"""
    try:
        name = (name or "").strip()
        if not name:
            raise ValueError("请填写技能名称（用于本地管理，不一定等于远程工具名）")
        headers = _parse_json(headers_text, dict, "请求头")
        env = _parse_json(env_text, dict, "环境变量")
        args = _parse_json(args_text, list, "启动参数")
        if transport in ("streamablehttp", "sse"):
            if not (url or "").strip():
                raise ValueError("远程 MCP 需要填写 URL")
            spec = {"type": transport, "url": url.strip(), "headers": headers}
        else:
            if not (command or "").strip():
                raise ValueError("stdio MCP 需要填写 command，例如 npx")
            spec = {"type": "stdio", "command": command.strip(), "args": args, "env": env}
        learned = await mcp_manager.learn(name, spec)
        return (
            f"已学习 MCP server「{name}」，注册工具：{', '.join(learned) or '无'}",
            _tool_rows(),
            _server_rows(),
        )
    except Exception as e:  # noqa: BLE001
        return f"学习失败：{e}", _tool_rows(), _server_rows()


async def remove_mcp_ui(name: str):
    try:
        name = (name or "").strip()
        if not name:
            raise ValueError("请输入要卸载的 MCP server 名称")
        removed = await mcp_manager.remove(name, persist=True)
        return f"已卸载「{name}」，移除工具：{', '.join(removed) or '无'}", _tool_rows(), _server_rows()
    except Exception as e:  # noqa: BLE001
        return f"卸载失败：{e}", _tool_rows(), _server_rows()


def refresh_management_ui():
    return "已刷新", _tool_rows(), _server_rows()


def build_web():
    """Web 界面：流式聊天 + 技能管理两个 Tab，共享同一个 Agent 实例。"""
    with gr.Blocks(title="Registry Agent v2.1") as demo:
        gr.Markdown("# Registry Agent v2.1\nReAct + Harness · 流式回答 · MCP 技能管理")
        session_state = gr.State(value="")

        with gr.Tab("对话"):
            chatbot = gr.Chatbot(height=480, label="对话")
            status = gr.Markdown("就绪")
            with gr.Row():
                textbox = gr.Textbox(
                    placeholder="请输入问题…", label="", container=False, scale=8, autofocus=True,
                )
                send_button = gr.Button("发送", variant="primary", scale=1)
                clear_button = gr.Button("清空会话", scale=1)

            chat_outputs = [chatbot, textbox, status, session_state]
            send_button.click(chat_stream_ui, [textbox, chatbot, session_state], chat_outputs)
            textbox.submit(chat_stream_ui, [textbox, chatbot, session_state], chat_outputs)
            clear_button.click(clear_chat_ui, [session_state], chat_outputs)

        with gr.Tab("技能管理"):
            gr.Markdown(
                "管理本地工具和 MCP 技能。远程 MCP 使用 streamablehttp/SSE；本地 MCP 使用 stdio。"
                "请求头和环境变量仅在提交时使用，不会在列表中回显。"
            )
            tool_table = gr.Dataframe(
                value=_tool_rows(),
                headers=["暴露名称", "类型", "来源", "远程原名", "说明"],
                datatype=["str"] * 5,
                interactive=False,
                label="当前已装工具",
            )
            server_table = gr.Dataframe(
                value=_server_rows(),
                headers=["MCP 名称", "连接状态", "传输", "工具数", "地址/命令"],
                datatype=["str"] * 5,
                interactive=False,
                label="已学习 MCP server",
            )
            refresh_button = gr.Button("刷新列表")

            with gr.Accordion("添加 MCP 技能", open=False):
                mcp_name = gr.Textbox(label="技能名称", placeholder="例如 luckin")
                transport = gr.Dropdown(
                    choices=["streamablehttp", "sse", "stdio"], value="streamablehttp", label="传输方式",
                )
                mcp_url = gr.Textbox(label="远程 URL", placeholder="https://example.com/mcp")
                headers_json = gr.Textbox(
                    label="请求头 JSON（可选）", type="password", placeholder='{"Authorization":"Bearer <token>"}',
                )
                command = gr.Textbox(label="stdio command", placeholder="例如 npx")
                args_json = gr.Textbox(label="stdio args JSON（可选）", placeholder='["some-mcp"]')
                env_json = gr.Textbox(label="环境变量 JSON（可选）", type="password", placeholder='{"API_KEY":"..."}')
                learn_button = gr.Button("学习并安装技能", variant="primary")

            with gr.Accordion("卸载 MCP 技能", open=False):
                remove_name = gr.Textbox(label="MCP server 名称", placeholder="例如 luckin")
                remove_button = gr.Button("卸载技能", variant="stop")

            management_status = gr.Markdown("就绪")
            management_outputs = [management_status, tool_table, server_table]
            refresh_button.click(refresh_management_ui, outputs=management_outputs)
            learn_button.click(
                learn_mcp_ui,
                [mcp_name, transport, mcp_url, headers_json, command, args_json, env_json],
                management_outputs,
            )
            remove_button.click(remove_mcp_ui, [remove_name], management_outputs)

    return demo


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
