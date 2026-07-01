import json
from datetime import datetime


def build_system_prompt(tools_schema: list) -> str:
    """
    构建系统提示词，告诉模型它有哪些工具可以使用，以及如何调用。
    """

    # 1. 基础设定
    prompt = f"""你是一个智能助手。
                当前时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}

                # 你的能力
                你可以直接回答用户的问题，也可以使用以下工具来辅助回答。
                如果用户的问题需要查询实时数据（如天气），请务必使用工具。

                # 可用工具列表
            """

    # 2. 动态插入工具描述
    for tool in tools_schema:
        prompt += f"\n## {tool['name']}\n"
        prompt += f"功能描述: {tool['description']}\n"
        # 将参数转换为易读的 JSON 字符串
        prompt += f"参数格式: {json.dumps(tool['params'], ensure_ascii=False)}\n"

    # 3. 强调调用规则 (这一步对 Qwen 很重要)
    prompt += """
            # 工具调用规则 (非常重要)
            如果你决定使用工具，**必须且只能**返回一个 JSON 格式的代码块，格式如下：

            ```json
            {
                "tool": "工具名称",
                "parameters": {
                    "参数名": "参数值"
                }
            }
            注意：不要返回任何 Python 代码，只返回 JSON。JSON 必须包裹在 json 和 之间。如果不需要使用工具，请直接用自然语言回答用户。 """
    return prompt