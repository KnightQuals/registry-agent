"""
tools/mcp_client.py — MCP 客户端（"喂 MCP 就学会"的核心）

对应用户核心需求：
  同门自研工具能加；网上现成的 MCP（如瑞幸提供 MCP 接入方式）丢给系统，
  它连上后自动把该 server 暴露的工具学会，成为自己的技能之一。

支持三种标准传输方式（配置格式对齐 MCP 官方 `mcpServers`）：
  - streamablehttp : Streamable HTTP 远程模式（当前公有 MCP 服务最主流，如瑞幸点餐）
                     {"type":"streamablehttp","url":"https://.../mcp","headers":{"Authorization":"Bearer xxx"}}
  - sse            : SSE 远程模式
                     {"type":"sse","url":"https://.../sse","headers":{...}}
  - stdio          : 本地起子进程（如 `npx some-mcp`）
                     {"type":"stdio","command":"npx","args":["some-mcp"],"env":{...}}

关键实现说明（重要）：
  MCP 官方 SDK 基于 anyio task group，一条连接的"建立→使用→关闭"必须在同一个 async task 内完成，
  不能跨 task 进出上下文（否则报 "cancel scope in a different task"）。
  因此本模块为每个 server 起一个**常驻后台 task**：task 内用 `async with` 持有整条连接的生命周期，
  主流程通过 asyncio 队列向它投递工具调用请求、取回结果。这样连接稳定、关闭干净。
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    from mcp.client.streamable_http import streamablehttp_client
    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "config", "mcp_servers.json"
)


class _ServerConnection:
    """
    单个 MCP server 的常驻连接。整条连接的生命周期都在 _runner task 内，
    外部通过 call() 投递请求。这样规避 anyio 的跨 task cancel scope 限制。
    """

    def __init__(self, name: str, spec: dict):
        self.name = name
        self.spec = spec
        self._req_q: asyncio.Queue = asyncio.Queue()
        self._ready: asyncio.Future = asyncio.get_event_loop().create_future()
        self._task: Optional[asyncio.Task] = None
        self.tools: list = []  # list_tools 结果

    async def start(self) -> list:
        """启动后台连接 task，等待其就绪并返回工具列表。"""
        self._task = asyncio.create_task(self._runner())
        await self._ready  # 若连接失败，这里会抛出异常
        return self.tools

    def _open_transport(self):
        """按 type 返回对应的 transport 上下文管理器。"""
        stype = (self.spec.get("type") or "stdio").lower()
        if stype in ("streamablehttp", "streamable_http", "http"):
            return "http", streamablehttp_client(self.spec["url"], headers=self.spec.get("headers") or None)
        if stype == "sse":
            return "sse", sse_client(self.spec["url"], headers=self.spec.get("headers") or None)
        if stype == "stdio":
            params = StdioServerParameters(
                command=self.spec["command"], args=self.spec.get("args") or [], env=self.spec.get("env"),
            )
            return "stdio", stdio_client(params)
        raise ValueError(f"不支持的 MCP type: {stype}（支持 streamablehttp / sse / stdio）")

    async def _runner(self):
        """常驻 task：建立连接 → 拉工具 → 循环处理调用请求 → 收到关闭信号后退出。"""
        try:
            kind, transport_cm = self._open_transport()
            async with transport_cm as transport:
                # streamablehttp 返回 (read, write, get_session_id)，其余返回 (read, write)
                read, write = transport[0], transport[1]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    resp = await session.list_tools()
                    self.tools = list(resp.tools)
                    if not self._ready.done():
                        self._ready.set_result(True)

                    # 请求处理循环
                    while True:
                        item = await self._req_q.get()
                        if item is None:  # 关闭信号
                            break
                        tool_name, arguments, fut = item
                        try:
                            result = await session.call_tool(tool_name, arguments)
                            parts = []
                            for c in getattr(result, "content", []) or []:
                                parts.append(getattr(c, "text", None) or str(c))
                            fut.set_result("\n".join(parts) if parts else str(result))
                        except Exception as e:  # noqa: BLE001
                            if not fut.done():
                                fut.set_exception(e)
        except Exception as e:  # noqa: BLE001
            if not self._ready.done():
                self._ready.set_exception(e)

    async def call(self, tool_name: str, arguments: dict) -> str:
        """向常驻 task 投递一次工具调用，等待结果。"""
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._req_q.put((tool_name, arguments, fut))
        return await fut

    async def close(self):
        await self._req_q.put(None)
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except Exception:
                self._task.cancel()


class MCPManager:
    """管理多个 MCP server 连接（stdio / sse / streamablehttp），注册其工具进 ToolRegistry。"""

    def __init__(self, registry, config_path: str = DEFAULT_CONFIG):
        self.registry = registry
        self.config_path = config_path
        self._conns: dict[str, _ServerConnection] = {}

    # ---------- 配置持久化 ----------
    def _load_config(self) -> dict:
        if not os.path.exists(self.config_path):
            return {"mcpServers": {}}
        with open(self.config_path, encoding="utf-8") as f:
            return json.load(f)

    def _save_config(self, cfg: dict) -> None:
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    # ---------- 通用学习入口 ----------
    async def learn(self, name: str, spec: dict, persist: bool = True) -> list[str]:
        """
        按标准 mcpServers 配置学习一个 server，返回学到的工具名列表。
        spec 示例见模块 docstring。
        """
        if not _HAS_MCP:
            raise RuntimeError("未安装 mcp SDK，请先 pip install mcp")

        conn = _ServerConnection(name, spec)
        tools = await conn.start()  # 连接失败会在此抛出
        self._conns[name] = conn

        learned = []
        for tool in tools:
            tool_name = tool.name

            async def _mcp_call(arguments: dict, _conn=conn, _tname=tool_name) -> str:
                return await _conn.call(_tname, arguments)

            self.registry.register_mcp_tool(
                name=tool_name,
                description=tool.description or "",
                parameters=tool.inputSchema or {"type": "object", "properties": {}},
                mcp_call=_mcp_call,
            )
            learned.append(tool_name)

        if persist:
            cfg = self._load_config()
            cfg.setdefault("mcpServers", {})[name] = spec
            self._save_config(cfg)
        return learned

    # ---------- 便捷入口 ----------
    async def learn_stdio_server(self, name: str, command: str, args: Optional[list[str]] = None,
                                 env: Optional[dict] = None, persist: bool = True) -> list[str]:
        return await self.learn(
            name, {"type": "stdio", "command": command, "args": args or [], "env": env or {}}, persist
        )

    async def learn_http_server(self, name: str, url: str, headers: Optional[dict] = None,
                                persist: bool = True) -> list[str]:
        """学习一个 Streamable HTTP 远程 MCP（如瑞幸点餐）。"""
        return await self.learn(
            name, {"type": "streamablehttp", "url": url, "headers": headers or {}}, persist
        )

    async def load_all_from_config(self) -> dict[str, list[str]]:
        """启动时把配置里所有 MCP server 重新连上并注册工具。"""
        cfg = self._load_config()
        out: dict[str, list[str]] = {}
        for name, spec in cfg.get("mcpServers", {}).items():
            try:
                out[name] = await self.learn(name, spec, persist=False)
            except Exception as e:  # noqa: BLE001
                out[name] = [f"<连接失败: {e}>"]
        return out

    async def close(self) -> None:
        for conn in self._conns.values():
            try:
                await conn.close()
            except Exception:
                pass
        self._conns.clear()
