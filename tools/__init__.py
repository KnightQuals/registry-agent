"""
tools/ — 工具层（Tool 层）：三来源统一注册

三种工具来源，统一归一化成 OpenAI 标准 tool schema，一起喂给模型：
  ① 本地内置工具   —— weather / media（保留 v1）
  ② 同门自研工具   —— 写个 py 函数 + @register_tool 装饰器，丢进 tools/ 自动注册
  ③ MCP 工具       —— 通过 tools/mcp_client 连接 MCP server，自动拉取其暴露的工具

关键升级（对比 v1）：
- v1 的 register_tool 生成的是自定义 schema（喂进 prompt 让模型手写 JSON）。
- v2 复用其 inspect 反射逻辑，但生成 **OpenAI function calling 标准 schema**，
  配合 core/loop 的原生 tool_calls。这也是接入 MCP 生态的前提。
- 统一 async 调用接口 registry.call()：本地同步函数用线程池包装，MCP 工具直接 await。
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Annotated, Any, Callable, Optional, get_origin

# ---- Python 类型 → JSON Schema 类型映射 ----
_PY_TO_JSON = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


class ToolRegistry:
    """工具注册表：本地函数工具 + MCP 工具统一管理。"""

    def __init__(self):
        # name -> {schema, func, kind, mcp_call, source, original_name}
        # source 用于展示/卸载某个 MCP server 时精确清理其工具。
        self._tools: dict[str, dict] = {}

    # ---------- ① / ② 本地 & 自研工具 ----------
    def register(self, func: Callable) -> Callable:
        """装饰器：注册本地/自研工具，从函数签名反射生成 OpenAI 标准 schema。"""
        name = func.__name__
        description = (inspect.getdoc(func) or "No description provided.").strip()

        properties: dict[str, Any] = {}
        required: list[str] = []

        for pname, param in inspect.signature(func).parameters.items():
            ann = param.annotation
            jtype, pdesc, is_required = "string", "", True
            if ann is not inspect.Parameter.empty:
                if get_origin(ann) is Annotated:
                    base = ann.__origin__
                    meta = ann.__metadata__
                    pdesc = meta[0] if len(meta) > 0 else ""
                    is_required = meta[1] if len(meta) > 1 else True
                    jtype = _PY_TO_JSON.get(getattr(base, "__name__", "str"), "string")
                else:
                    jtype = _PY_TO_JSON.get(getattr(ann, "__name__", "str"), "string")
            # 有默认值的参数视为可选
            if param.default is not inspect.Parameter.empty:
                is_required = False
            properties[pname] = {"type": jtype, "description": pdesc}
            if is_required:
                required.append(pname)

        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        self._tools[name] = {
            "schema": schema,
            "func": func,
            "kind": "local",
            "mcp_call": None,
            "source": "local",
            "original_name": name,
        }
        print(f"[tools] 注册本地工具: {name}")
        return func

    # ---------- ③ MCP 工具 ----------
    def register_mcp_tool(
        self,
        name: str,
        description: str,
        parameters: dict,
        mcp_call: Callable,
        source: str = "mcp",
        original_name: str | None = None,
    ) -> None:
        """
        注册一个来自 MCP server 的工具。
        - name: 暴露给模型的唯一工具名（多个 MCP 同名工具会做命名空间隔离）
        - parameters: MCP server 返回的 JSON Schema（inputSchema）
        - mcp_call: async 可调用，签名 async (arguments: dict) -> str
        - source: MCP server 名称，用于技能管理面板展示和卸载。
        """
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description or f"MCP tool {name}",
                "parameters": parameters or {"type": "object", "properties": {}},
            },
        }
        self._tools[name] = {
            "schema": schema,
            "func": None,
            "kind": "mcp",
            "mcp_call": mcp_call,
            "source": source,
            "original_name": original_name or name,
        }
        print(f"[tools] 注册 MCP 工具: {name}")

    # ---------- 查询 ----------
    def openai_schema(self, names: Optional[list[str]] = None) -> list[dict]:
        """
        返回喂给模型的工具 schema（OpenAI function calling 格式）。
        names=None 返回全部；传候选名列表时只返回路由后的子集。
        """
        if names is None:
            return [t["schema"] for t in self._tools.values()]
        return [self._tools[name]["schema"] for name in names if name in self._tools]

    def list_tools(self) -> list[dict]:
        """返回工具清单（供 API、路由器和 Web 技能管理面板展示）。"""
        return [
            {
                "name": n,
                "kind": t["kind"],
                "source": t["source"],
                "original_name": t["original_name"],
                "description": t["schema"]["function"]["description"],
            }
            for n, t in self._tools.items()
        ]

    def tool_search_text(self, name: str) -> str:
        """返回路由器用于粗筛的名称、说明和参数说明文本。"""
        entry = self._tools[name]
        func = entry["schema"]["function"]
        props = func.get("parameters", {}).get("properties", {})
        prop_text = " ".join(
            f"{param_name} {meta.get('description', '')}" for param_name, meta in props.items()
        )
        return f"{name} {entry['original_name']} {entry['source']} {func.get('description', '')} {prop_text}"

    def unregister(self, name: str) -> bool:
        """卸载一个已注册工具。仅由技能管理/MCP 卸载流程使用。"""
        return self._tools.pop(name, None) is not None

    def unregister_source(self, source: str) -> list[str]:
        """卸载某个 MCP server 贡献的全部工具，返回被移除的工具名。"""
        names = [name for name, entry in self._tools.items() if entry.get("source") == source]
        for name in names:
            self._tools.pop(name, None)
        return names

    def has(self, name: str) -> bool:
        return name in self._tools

    # ---------- 统一异步调用 ----------
    async def call(self, name: str, arguments: dict) -> Any:
        """
        统一调用入口。
        - 本地工具：可能是同步函数，用线程池执行避免阻塞事件循环。
        - MCP 工具：直接 await 其 async mcp_call。
        """
        if name not in self._tools:
            raise KeyError(f"工具 {name} 未注册")
        entry = self._tools[name]

        if entry["kind"] == "mcp":
            return await entry["mcp_call"](arguments)

        func = entry["func"]
        if inspect.iscoroutinefunction(func):
            return await func(**arguments)
        # 同步函数丢到线程池
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(**arguments))


# 全局默认注册表（本地工具通过装饰器注册到这里）
registry = ToolRegistry()


def register_tool(func: Callable) -> Callable:
    """向后兼容的装饰器名（同门自研工具沿用这个名字即可）。"""
    return registry.register(func)


def load_builtin_tools() -> None:
    """导入内置工具模块，触发装饰器注册。"""
    from . import weather  # noqa: F401
    from . import media    # noqa: F401
