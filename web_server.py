import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import gradio as gr
from agent_engine import AgentEngine
import traceback

# ================= 配置 =================
SERVER_PORT = 8500


# ================= API 模型 =================
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    history: List[ChatMessage] = []


# ================= FastAPI 主应用 =================
app = FastAPI(title="智能体工具集 API")


@app.post("/api/chat")
async def api_chat(request: ChatRequest):
    """
    API 接口：返回结构化 JSON 数据
    """
    print(f"📡 [API] 收到请求: {request.query}")

    agent = AgentEngine()

    # 注入历史
    if request.history:
        for msg in request.history:
            agent.history.append({"role": msg.role, "content": msg.content})

    try:
        # 🔥 关键修改：开启 json_mode=True
        # 这样 response 就会是一个 字典(Dict)，而不是 字符串
        response = agent.chat(request.query, json_mode=True)

        return {
            "status": "success",
            "response": response
        }
    except Exception as e:
        error_msg = f"API Error: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "status": "error",
            "message": error_msg,
            "response": None
        }


# ================= Gradio 网页逻辑 =================
def chat_function(message, history):
    """
    网页回调：返回 Markdown 文本 (json_mode=False)
    """
    print(f"📥 [Web] 收到消息: {message}")

    agent = AgentEngine()

    if history:
        for turn in history:
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                user_msg = turn[0]
                bot_msg = turn[1]
                if user_msg and bot_msg:
                    agent.history.append({"role": "user", "content": str(user_msg)})
                    agent.history.append({"role": "assistant", "content": str(bot_msg)})

    try:
        # 🔥 网页端保持默认，返回 Markdown 字符串，方便渲染图片
        response = agent.chat(message, json_mode=False)
        print(f"📤 [Web] 回复: {str(response)[:30]}...")
        return response

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        return error_msg


# ================= 启动 =================
def main():
    print(f"========================================")
    print(f"🚀 服务启动中 (Web + JSON API)...")
    print(f"👉 Web: [http://0.0.0.0](http://0.0.0.0):{SERVER_PORT}")
    print(f"👉 API: [http://0.0.0.0](http://0.0.0.0):{SERVER_PORT}/api/chat")
    print(f"========================================")

    demo = gr.ChatInterface(
        fn=chat_function,
        title="🤖 智能体工具集 (演示版)",
        textbox=gr.Textbox(placeholder="请输入问题...", container=False, scale=7),
    )

    mounted_app = gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(mounted_app, host="0.0.0.0", port=SERVER_PORT)


if __name__ == "__main__":
    main()