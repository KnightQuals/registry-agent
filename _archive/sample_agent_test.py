import random
import time
import re
import json
import os
from openai import OpenAI
import requests
from typing import Dict, Any, List

class FunctionAgent:
    def __init__(self, api_key: str, weather_api_key: str = None):
        self.api_key = api_key
        self.weather_api_key = weather_api_key or os.getenv("SENIVERSE_API_KEY")
        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8502/v1"),
        )
        self.functions = self._register_functions()
        
    def _register_functions(self) -> Dict[str, Dict[str, Any]]:
        """注册所有可用函数及其元数据"""
        return {
            "generate_random_number": {
                "function": self._generate_random_number,
                "description": "生成1到100之间的随机整数",
                "parameters": {}
            },
            "get_current_time": {
                "function": self._get_current_time,
                "description": "获取当前系统时间",
                "parameters": {}
            },
            "correct_punctuation": {
                "function": self._correct_punctuation,
                "description": "修正中文标点符号",
                "parameters": {
                    "text": {"type": "string", "description": "需要修正标点的文本"}
                }
            },
            "simple_math_operation": {
                "function": self._simple_math_operation,
                "description": "执行简单的数学运算：加法(add)、减法(subtract)、乘法(multiply)、除法(divide)",
                "parameters": {
                    "operation": {"type": "string", "description": "运算类型"},
                    "num1": {"type": "number", "description": "第一个数字"},
                    "num2": {"type": "number", "description": "第二个数字"}
                }
            },
            "get_weather": {
                "function": self._get_weather,
                "description": "获取指定城市的天气信息，城市参数使用拼音如beijing、shanghai、wuhan",
                "parameters": {
                    "location": {"type": "string", "description": "城市名称拼音，如beijing、shanghai、wuhan"},
                    "days": {"type": "integer", "description": "查询天数，默认3天"}
                }
            }
        }
    
    # 功能函数作为类方法
    def _generate_random_number(self):
        return random.randint(1, 100)
    
    def _get_current_time(self):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    def _correct_punctuation(self, text: str):
        return text.replace("，", ",").replace("。", ".")
    
    def _simple_math_operation(self, operation: str, num1: float, num2: float):
        operations = {
            "add": num1 + num2,
            "subtract": num1 - num2,
            "multiply": num1 * num2,
            "divide": num1 / num2 if num2 != 0 else "除数不能为零"
        }
        return operations.get(operation, "不支持的运算")
    
    def _get_weather(self, location: str, days: int = 3):
        """获取天气信息 - 使用正确的API格式"""
        if not self.weather_api_key:
            return {"error": "天气API密钥未配置"}
            
        # 确保城市名称是拼音格式
        location_pinyin = self._convert_city_to_pinyin(location)
        print(f"🌍 查询城市: {location} -> {location_pinyin}")
        
        # 确保days是整数
        days_int = int(days)
        print(f"📅 查询天数: {days} -> {days_int}")
        
        # 使用正确的API格式，包含start=-1参数
        url = f"https://api.seniverse.com/v3/weather/daily.json?key={self.weather_api_key}&location={location_pinyin}&language=zh-Hans&unit=c&start=-1&days={days_int}"
        print(f"🔗 API请求URL: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            print(f"📡 API响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                results = data.get('results', [])
                if not results:
                    return {"error": "API返回数据为空"}
                
                weather_data = results[0].get('daily', [])
                
                if not weather_data:
                    return {"error": "无法获取天气数据"}
                
                forecasts = []
                for day in weather_data:
                    # 解析温度数据（API返回格式如 "28℃"）
                    high_temp = day.get('high', '').replace('℃', '').strip()
                    low_temp = day.get('low', '').replace('℃', '').strip()
                    
                    forecasts.append({
                        "date": day.get('date', ''),
                        "high": high_temp,
                        "low": low_temp,
                        "day_text": day.get('text_day', ''),
                        "night_text": day.get('text_night', ''),
                        "wind_direction": day.get('wind_direction', ''),
                        "wind_scale": day.get('wind_scale', '')
                    })
                
                location_name = results[0].get('location', {}).get('name', location)
                last_update = results[0].get('last_update', '')
                
                return {
                    "location": location_name,
                    "location_pinyin": location_pinyin,
                    "days": days_int,
                    "forecasts": forecasts,
                    "last_update": last_update
                }
            else:
                error_detail = ""
                try:
                    error_data = response.json()
                    error_detail = f"，错误信息：{error_data}"
                except:
                    error_detail = f"，响应内容：{response.text}"
                return {"error": f"API请求失败，状态码：{response.status_code}{error_detail}"}
                
        except Exception as e:
            return {"error": f"请求发生错误：{str(e)}"}
    
    def _convert_city_to_pinyin(self, city_name: str) -> str:
        """将中文城市名转换为拼音，如果是拼音则直接返回"""
        # 常见城市的中文到拼音映射
        city_mapping = {
            "北京": "beijing", "上海": "shanghai", "广州": "guangzhou", 
            "深圳": "shenzhen", "杭州": "hangzhou", "南京": "nanjing",
            "武汉": "wuhan", "成都": "chengdu", "重庆": "chongqing",
            "西安": "xian", "苏州": "suzhou", "天津": "tianjin",
            "长沙": "changsha", "郑州": "zhengzhou", "青岛": "qingdao",
            "大连": "dalian", "厦门": "xiamen", "宁波": "ningbo",
            "合肥": "hefei", "福州": "fuzhou", "济南": "jinan"
        }
        
        # 如果已经是拼音，直接返回
        if re.match(r'^[a-z]+$', city_name.lower()):
            return city_name.lower()
        
        # 如果是中文，查找映射
        if city_name in city_mapping:
            return city_mapping[city_name]
        
        # 如果不在映射中，尝试使用模型转换
        return self._convert_to_pinyin_with_model(city_name)
    
    def _convert_to_pinyin_with_model(self, chinese_text: str) -> str:
        """使用模型将中文转换为拼音"""
        try:
            prompt = f"""
            请将以下中文地名转换为拼音（小写，无空格）：
            {chinese_text}
            
            只返回拼音，不要返回其他内容。
            """
            
            response = self.client.chat.completions.create(
                model="glm-4",
                messages=[
                    {"role": "system", "content": "你是一个地名转换助手，请准确将中文地名转换为拼音。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            pinyin = response.choices[0].message.content.strip().lower()
            # 清理可能的额外字符
            pinyin = re.sub(r'[^a-z]', '', pinyin)
            return pinyin if pinyin else chinese_text.lower()
            
        except Exception as e:
            print(f"拼音转换失败: {e}")
            return chinese_text.lower()
    
    def _create_function_descriptions(self) -> List[Dict[str, Any]]:
        """创建函数描述供模型使用"""
        function_descriptions = []
        for name, info in self.functions.items():
            function_descriptions.append({
                "name": name,
                "description": info["description"],
                "parameters": {
                    "type": "object",
                    "properties": info["parameters"],
                    "required": list(info["parameters"].keys())
                }
            })
        return function_descriptions
    
    def _extract_parameters(self, function_name: str, user_input: str) -> Dict[str, Any]:
        """使用模型提取函数参数"""
        function_info = self.functions[function_name]
        
        if not function_info["parameters"]:
            return {}
        
        # 为天气功能特别优化提示词
        if function_name == "get_weather":
            prompt = f"""
            请从用户输入中提取天气查询所需的参数：
            
            用户输入：{user_input}
            
            需要提取的参数：
            - location: 城市名称（请提取中文城市名，如北京、上海、武汉等）
            - days: 查询天数（必须是整数，默认3）
            
            请以JSON格式返回，例如：
            {{"location": "北京", "days": 3}}
            
            注意：days参数必须是整数，不能是小数。
            """
        else:
            prompt = f"""
            请从用户输入中提取以下函数所需的参数：
            
            函数：{function_info['description']}
            所需参数：{json.dumps(function_info['parameters'], ensure_ascii=False, indent=2)}
            
            用户输入：{user_input}
            
            请以JSON格式返回参数，例如：{{"param1": "value1", "param2": "value2"}}
            注意：数字参数请转换为数字类型，字符串参数保持字符串类型。
            如果无法从输入中提取某些参数，请使用合理的默认值。
            """
        
        try:
            response = self.client.chat.completions.create(
                model="glm-4",
                messages=[
                    {"role": "system", "content": "你是一个参数提取助手，请准确提取用户输入中的参数并返回有效的JSON。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            params_text = response.choices[0].message.content
            print(f"原始参数响应: {params_text}")
            
            # 提取JSON部分
            json_match = re.search(r'\{[^{}]*\}', params_text)
            if json_match:
                params = json.loads(json_match.group())
                
                # 参数验证和转换
                validated_params = {}
                for param_name, param_config in function_info["parameters"].items():
                    if param_name in params:
                        if param_config["type"] == "number":
                            try:
                                validated_params[param_name] = float(params[param_name])
                            except (ValueError, TypeError):
                                validated_params[param_name] = params[param_name]
                        elif param_config["type"] == "integer":
                            try:
                                validated_params[param_name] = int(params[param_name])
                            except (ValueError, TypeError):
                                validated_params[param_name] = params[param_name]
                        else:
                            validated_params[param_name] = params[param_name]
                
                return validated_params
            else:
                return {}
                
        except Exception as e:
            print(f"参数提取失败: {e}")
            return {}
    
    def _get_model_beautified_response(self, user_input: str, function_name: str, function_result: Any) -> str:
        """使用大模型自动美化回复"""
        
        # 根据功能类型创建不同的美化提示
        if function_name == "generate_random_number":
            prompt = f"""
            用户请求：{user_input}
            生成的随机数：{function_result}
            
            请用友好、有趣的方式告诉用户这个随机数结果。
            """
            
        elif function_name == "get_current_time":
            prompt = f"""
            用户询问：{user_input}
            当前时间：{function_result}
            
            请用友好、自然的方式告诉用户当前时间。
            """
            
        elif function_name == "correct_punctuation":
            prompt = f"""
            用户请求：{user_input}
            原始文本：{user_input}
            修正结果：{function_result}
            
            请呈现修正结果，并简要说明修正了哪些标点。
            """
            
        elif function_name == "simple_math_operation":
            prompt = f"""
            用户进行数学运算：{user_input}
            运算结果：{function_result}
            
            请清晰地展示数学运算结果。
            """
            
        elif function_name == "get_weather":
            if "error" in function_result:
                prompt = f"""
                用户查询天气：{user_input}
                出现错误：{function_result['error']}
                
                请用友好的方式告知用户这个错误。
                """
            else:
                prompt = f"""
                用户查询天气：{user_input}
                查询城市：{function_result['location']}
                查询天数：{function_result['days']}天
                最后更新时间：{function_result.get('last_update', '未知')}
                详细天气数据：{json.dumps(function_result['forecasts'], ensure_ascii=False, indent=2)}
                
                请根据以上天气数据，为用户生成一个完整、实用的天气报告。
                包括：
                - 总体天气概况
                - 逐日详细天气预报（日期、白天天气、夜间天气、温度、风力）
                - 实用的出行和生活建议
                
                用中文回复，保持专业友好的语气，可以适当使用表情符号让报告更生动。
                """
        else:
            prompt = f"""
            用户输入：{user_input}
            功能执行结果：{function_result}
            
            请根据以上信息，为用户生成一个友好的回复。
            """
        
        try:
            response = self.client.chat.completions.create(
                model="glm-4",
                messages=[
                    {
                        "role": "system", 
                        "content": """你是一个友好的AI助手，负责将功能执行结果转化为自然、友好的回复。
                        根据用户问题和功能结果，生成贴切、有用的回复。
                        保持简洁明了，专业友好。"""
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"功能执行成功，但美化回复时出错：{str(e)}\n原始结果：{function_result}"
    
    def process_user_input(self, user_input: str) -> str:
        """处理用户输入的主函数"""
        print(f"👤 用户输入: {user_input}")
        
        # 第一步：让模型选择函数
        function_descriptions = self._create_function_descriptions()
        
        selection_prompt = f"""
        请根据用户输入选择合适的函数，并只返回函数名称。
        
        可用函数：
        {json.dumps([{"name": name, "description": info["description"]} for name, info in self.functions.items()], ensure_ascii=False, indent=2)}
        
        用户输入：{user_input}
        
        请只返回函数名称，不要返回其他任何内容。
        """
        
        try:
            selection_response = self.client.chat.completions.create(
                model="glm-4",
                messages=[
                    {"role": "system", "content": "你是一个函数选择助手，请准确选择最合适的函数名称。"},
                    {"role": "user", "content": selection_prompt}
                ],
                temperature=0.1
            )
            
            function_name = selection_response.choices[0].message.content.strip()
            print(f"✅ 选择的函数: {function_name}")
            
            # 验证函数是否存在
            if function_name not in self.functions:
                return "❌ 抱歉，我无法处理这个请求。请尝试用其他方式提问。"
            
            # 第二步：提取参数
            parameters = self._extract_parameters(function_name, user_input)
            print(f"📋 提取的参数: {parameters}")
            
            # 第三步：准备最终参数
            final_params = parameters.copy()
            
            # 特殊参数处理
            if function_name == "correct_punctuation" and "text" not in final_params:
                final_params["text"] = user_input
            
            # 第四步：执行函数
            function_info = self.functions[function_name]
            func = function_info["function"]
            
            start_time = time.time()
            try:
                function_result = func(**final_params)
                execution_time = time.time() - start_time
                print(f"🎯 函数执行结果: {function_result}")
                print(f"⏱️ 执行时间: {execution_time:.2f}秒")
                
                # 第五步：使用大模型美化回复
                beautify_start_time = time.time()
                final_response = self._get_model_beautified_response(user_input, function_name, function_result)
                beautify_time = time.time() - beautify_start_time
                print(f"🎨 美化处理时间: {beautify_time:.2f}秒")
                
                return final_response
                
            except Exception as e:
                return f"❌ 执行函数时出错: {str(e)}"
                
        except Exception as e:
            return f"❌ 处理请求时出错: {str(e)}"

def test_agent():
    """测试Agent的各种功能"""
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    weather_api_key = os.getenv("SENIVERSE_API_KEY")
    
    agent = FunctionAgent(api_key, weather_api_key)
    
    test_cases = [
        "武汉天气怎么样？",
        "北京未来三天的天气",
        "上海明天天气",
        "给我一个随机数",
        "现在几点了？",
    ]
    
    print("🚀 开始测试Function Agent...\n")
    
    for i, user_input in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"🧪 测试用例 {i}: {user_input}")
        print(f"{'='*60}")
        
        start_time = time.time()
        response = agent.process_user_input(user_input)
        total_time = time.time() - start_time
        
        print(f"💬 最终回复:\n{response}")
        print(f"\n⏰ 总处理时间: {total_time:.2f}秒")

if __name__ == "__main__":
    test_agent()
