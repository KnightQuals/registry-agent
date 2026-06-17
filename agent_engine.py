import requests
import json
import re
import os
from datetime import datetime
from tools import run_tools_with_params, get_tools_schema

# ================= 工具注册区域 =================
# 1. 导入天气工具
import tools.weather

# 2. 导入媒体检索工具 (修正：使用 tools.media)
import tools.media


# 3. SQL工具暂时移除
# import tools.sql_analysis
# ===============================================

class AgentEngine:
    def __init__(self):
        self.api_url = os.getenv("QWEN_API_URL", "http://127.0.0.1:8502/v1/chat/completions")
        self.model_name = os.getenv("QWEN_MODEL_NAME", "qwen")
        self.history = []
        # 获取所有工具定义 (此时只会包含 weather 和 media)
        self.tools_schema = get_tools_schema()

    def _call_model(self, messages, temperature=0.3):
        """发送请求给 Qwen 模型"""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048
        }
        try:
            resp = requests.post(self.api_url, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content']
            else:
                return f"模型服务报错: {resp.status_code}"
        except Exception as e:
            return f"模型连接失败: {e}"

    def _parse_intent(self, user_input: str) -> dict:
        """自然语言 -> 工具参数 JSON"""
        tools_desc = json.dumps(self.tools_schema, ensure_ascii=False, indent=2)

        system_prompt = f"""
你是一个精准的参数提取器。当前时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}
【可用工具】
{tools_desc}
【任务】
1. 分析用户输入，判断应该调用哪些工具。
2. 为每个需要调用的工具提取参数。
3. 如果用户意图不清晰或不需要工具，返回空JSON。
【输出格式】
只返回一个 JSON 对象，例如：{{ "工具名": {{ "参数名": "参数值" }} }}
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"用户输入：{user_input}"}
        ]

        # 使用低温度 (0.1) 保证 JSON 格式稳定
        response = self._call_model(messages, temperature=0.1)

        try:
            clean_json = response.replace("```json", "").replace("```", "").strip()
            params = json.loads(clean_json)
            return params
        except:
            return {}

    def chat(self, user_input: str, json_mode: bool = False):
        """
        核心对话逻辑
        :param json_mode: 是否强制输出 JSON 格式 (供 API 使用)
        """
        self.history.append({"role": "user", "content": user_input})
        print(f"🤖 用户输入: {user_input}")

        # 1. 解析意图
        tool_params = self._parse_intent(user_input)

        if not tool_params:
            tools_observation = "本次未执行任何工具。"
        else:
            # 2. 并行执行
            tools_observation = run_tools_with_params(tool_params)

        print("📥 工具执行完毕，正在生成回答...")

        # 3. 综合回答 (根据模式选择 Prompt)
        if not json_mode:
            # === 模式 A: 普通文本模式 (给网页看) ===
            synthesis_prompt = f"""
你是一个智能助手。系统根据用户指令运行了外部工具，结果如下。
【工具运行报告】
{tools_observation}
【任务】
请结合工具报告回答用户问题。
1. 如实反馈失败：如果工具未找到结果，请明确告知。
2. 多源融合：自然地融合多个工具的结果。
3. 保持链接：Markdown 图片或链接必须原样保留。
4. 语气自然：像真人一样交流。
"""
        else:
            # === 模式 B: JSON 结构化模式 (给 API 看) ===
            synthesis_prompt = f"""
你是一个API数据格式化助手。系统根据用户指令运行了外部工具，结果如下。
【工具运行报告】
{tools_observation}

【任务】
请将工具运行结果和你的回答整理成标准的 JSON 格式。

【输出 JSON 结构要求】
{{
    "answer": "用自然语言简要总结结果，不要包含Markdown链接",
    "data": {{
        "weather": "如果有天气信息，填这里，否则null",
        "media": [ 
            {{ "filename": "文件名", "url": "下载链接", "type": "image/video" }} 
        ],
        "other": "其他工具的返回结果"
    }},
    "error": "如果有错误或未找到，简述原因，否则null"
}}

注意：
1. 直接返回 JSON，**不要**使用 ```json 代码块包裹。
2. 必须提取出媒体文件的 URL 和文件名放入 media 列表。
"""

        messages = [
                       {"role": "system", "content": synthesis_prompt}
                   ] + self.history[-5:]

        # 在 JSON 模式下，使用低温度保证格式稳定
        temp = 0.1 if json_mode else 0.3
        final_response = self._call_model(messages, temperature=temp)

        # 4. 后处理
        if json_mode:
            try:
                # 尝试清洗并解析 JSON
                clean_resp = final_response.replace("```json", "").replace("```", "").strip()
                parsed_json = json.loads(clean_resp)
                # 历史记录只存文本，不存大段 JSON
                self.history.append({"role": "assistant", "content": parsed_json["answer"]})
                return parsed_json  # 返回字典对象
            except Exception as e:
                print(f"❌ JSON解析失败: {e}")
                fallback = {
                    "answer": final_response,
                    "data": {},
                    "error": "Model format error"
                }
                self.history.append({"role": "assistant", "content": final_response})
                return fallback
        else:
            # 文本模式直接返回字符串
            self.history.append({"role": "assistant", "content": final_response})
            return final_response
