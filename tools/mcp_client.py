"""
tools/mcp_client.py — MCP 客户端（"喂 MCP 就学会"的核心）

对应用户核心需求：
  同门自研工具能加；网上现成的 MCP（如瑞幸提供 MCP 下载/连接方式）丢给系统，
  它连上后自动把该 server 暴露的工具学会，成为自己的技能之一。

机制：
  1. 用户给一个 MCP server 的连接方式（stdio: 一条命令；或 http/sse: 一个 URL）。
  2. MCP client 连上 server，调用标准的 list_tools()，拿到它暴露的全部工具及其 inputSchema。
  3. 逐个 register_mcp_tool() 注册进 ToolRegistry，与本地工具平起平坐。
  4. 连接配置写入 config/mcp_servers.json，下次启动自动重连、自动加载。

依赖：官方 mcp SDK（pip install mcp）。未安装时本模块可被 import，但连接会报错提示安装。
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "config", "mcp_servers.json"
)


class MCPManager:
    """管理多个 MCP server 连接，并把它们的工具注册进 ToolRegistry。"""

    def __init__(self, registry, config_path: str = DEFAULT_CONFIG):
        self.registry = registry
        self.config_path = config_path
        self._sessions: dict[str, Any] = {}     # server_name -> ClientSession
        self._ctxmgrs: dict[str, Any] = {}      # 保活的上下文管理器

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

    # ---------- 学习 / 连接 ----------
    async def learn_stdio_server(self, name: str, command: str, args: Optional[list[str]] = None,
                                 env: Optional[dict] = None, persist: bool = True) -> list[str]:
        """
        学习一个 stdio 型 MCP server（例如 `npx some-mcp`）。
        返回学到的工具名列表。persist=True 时写入配置，下次自动加载。

        用户场景：瑞幸给出 `npx luckin-coffee-mcp` → learn_stdio_server("luckin", "npx", ["luckin-coffee-mcp"])
        """
        if not _HAS_MCP:
            raise RuntimeError("未安装 mcp SDK，请先 pip install mcp")

        params = StdioServerParameters(command=command, args=args or [], env=env)
        ctx = stdio_client(params)
        read, write = await ctx.__aenter__()
        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()

        self._ctxmgrs[name] = (ctx, session)
        self._sessions[name] = session

        learned = await self._register_session_tools(name, session)

        if persist:
            cfg = self._load_config()
            cfg.setdefault("mcpServers", {})[name] = {
                "type": "stdio", "command": command, "args": args or [], "env": env or {},
            }
            self._save_config(cfg)
        return learned

    async def _register_session_tools(self, server_name: str, session) -> list[str]:
        """从一个已连接 session 拉取工具并注册。"""
        resp = await session.list_tools()
        learned = []
        for tool in resp.tools:
            tool_name = tool.name

            async def _mcp_call(arguments: dict, _sess=session, _tname=tool_name) -> str:
                result = await _sess.call_tool(_tname, arguments)
                # 归一化 MCP 返回内容为文本
                parts = []
                for c in getattr(result, "content", []) or []:
                    parts.append(getattr(c, "text", None) or str(c))
                return "\n".join(parts) if parts else str(result)

            self.registry.register_mcp_tool(
                name=tool_name,
                description=tool.description or "",
                parameters=tool.inputSchema or {"type": "object", "properties": {}},
                mcp_call=_mcp_call,
            )
            learned.append(tool_name)
        return learned

    async def load_all_from_config(self) -> dict[str, list[str]]:
        """启动时调用：把配置里所有已学会的 MCP server 重新连上并注册工具。"""
        cfg = self._load_config()
        out: dict[str, list[str]] = {}
        for name, spec in cfg.get("mcpServers", {}).items():
            if spec.get("type") == "stdio":
                try:
                    out[name] = await self.learn_stdio_server(
                        name, spec["command"], spec.get("args"), spec.get("env"), persist=False
                    )
                except Exception as e:  # noqa: BLE001
                    out[name] = [f"<连接失败: {e}>"]
        return out

    async def close(self) -> None:
        """关闭所有 MCP 连接。"""
        for name, (ctx, session) in self._ctxmgrs.items():
            try:
                await session.__aexit__(None, None, None)
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass
        self._ctxmgrs.clear()
        self._sessions.clear()
