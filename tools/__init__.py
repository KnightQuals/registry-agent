import inspect
import traceback
import concurrent.futures
import json
from typing import Dict, Any, List, Callable, get_origin, Annotated
from types import GenericAlias

# 全局变量存储工具
_TOOL_HOOKS: Dict[str, Callable] = {}
_TOOL_DESCRIPTIONS: List[Dict] = []


def register_tool(func: Callable):
    """装饰器：注册工具 (保持不变)"""
    tool_name = func.__name__
    tool_description = inspect.getdoc(func) or "No description provided."
    tool_description = tool_description.strip()

    python_params = inspect.signature(func).parameters
    tool_params = []

    for name, param in python_params.items():
        annotation = param.annotation
        typ_str = "str"
        description = ""
        required = True

        if annotation is not inspect.Parameter.empty:
            if get_origin(annotation) == Annotated:
                typ, metadata = annotation.__origin__, annotation.__metadata__
                description = metadata[0] if len(metadata) > 0 else ""
                required = metadata[1] if len(metadata) > 1 else True
            else:
                typ = annotation
            typ_str = str(typ) if isinstance(typ, GenericAlias) else typ.__name__

        tool_params.append({
            "name": name,
            "description": description,
            "type": typ_str,
            "required": required,
        })

    tool_def = {
        "name": tool_name,
        "description": tool_description,
        "params": tool_params,
    }

    _TOOL_HOOKS[tool_name] = func
    _TOOL_DESCRIPTIONS.append(tool_def)
    print(f"[System] 已注册工具: {tool_name}")
    return func


def get_tools_schema() -> List[Dict]:
    return _TOOL_DESCRIPTIONS


def run_tools_with_params(parsed_params: Dict[str, Any]) -> str:
    """
    【核心修改】根据解析好的参数，并发执行工具
    parsed_params 格式示例:
    {
        "get_weather": {"city_name": "武汉"},
        "search_remote_media": {"query": "办公室"}
    }
    """
    results = []
    print(f"\n⚡ [Parallel-Run] 接收到 {len(parsed_params)} 个工具的任务，开始执行...")

    def run_single_tool(name, func, kwargs):
        try:
            # 执行工具
            ret = func(**kwargs)
            return f"【工具 {name} 执行成功】\n参数: {kwargs}\n结果: {ret}\n"
        except Exception as e:
            return f"【工具 {name} 执行失败】\n参数: {kwargs}\n原因: {str(e)}\n"

    # 使用线程池并发执行
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for tool_name, args in parsed_params.items():
            if tool_name in _TOOL_HOOKS and args:  # 只有当 args 不为空/None 时才执行
                futures.append(executor.submit(run_single_tool, tool_name, _TOOL_HOOKS[tool_name], args))

        if not futures:
            return "没有工具被触发执行。"

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    return "\n".join(results)


# 保留旧接口兼容
def dispatch_tool(tool_name: str, tool_params: Dict[str, Any]) -> str:
    if tool_name not in _TOOL_HOOKS:
        return f"错误: 工具 {tool_name} 不存在"
    try:
        return str(_TOOL_HOOKS[tool_name](**tool_params))
    except Exception as e:
        return f"工具执行出错: {str(e)}"