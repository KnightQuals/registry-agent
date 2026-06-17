import requests
import re
import os
from typing import Annotated
from . import register_tool  # 导入注册装饰器


@register_tool
def get_weather(
        city_name: Annotated[str, "需要查询的城市名称（支持中文或拼音）", True],
        days: Annotated[int, "查询未来几天的数据（默认3天）", False] = 3
) -> str:
    """
    根据城市名称查询天气预报。
    """
    api_key = os.getenv("SENIVERSE_API_KEY")
    if not api_key:
        return "天气 API Key 未配置，请设置环境变量 SENIVERSE_API_KEY。"

    # 简单的拼音转换映射（简化版，实际可使用第三方库）
    city_mapping = {
        "北京": "beijing", "上海": "shanghai", "广州": "guangzhou",
        "深圳": "shenzhen", "杭州": "hangzhou", "南京": "nanjing",
        "武汉": "wuhan", "成都": "chengdu", "重庆": "chongqing",
        "西安": "xian", "苏州": "suzhou", "天津": "tianjin",
        "长沙": "changsha"
    }

    # 如果是中文且在映射中，转换为拼音；否则直接使用（假设用户输入了拼音或API支持中文）
    location = city_mapping.get(city_name, city_name)

    url = f"https://api.seniverse.com/v3/weather/daily.json?key={api_key}&location={location}&language=zh-Hans&unit=c&start=0&days={days}"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            if not results:
                return "API返回数据为空，请检查城市名称。"

            weather_data = results[0].get('daily', [])
            location_name = results[0].get('location', {}).get('name', city_name)

            report = f"【{location_name}】未来 {days} 天天气预报：\n"
            for day in weather_data:
                date = day.get('date')
                text_day = day.get('text_day')
                high = day.get('high')
                low = day.get('low')
                report += f"- {date}: {text_day}, {low}°C - {high}°C\n"

            return report
        else:
            return f"API请求失败，状态码：{response.status_code}"

    except Exception as e:
        return f"请求发生错误：{str(e)}"
