"""
core/memory.py — 会话记忆层（Context 层）

设计要点（对应用户"记忆机制"讨论）：
- 这不是 workbuddy 那种 .md 个人笔记，而是面向"多用户内部服务"的会话记忆。
- 用 SQLite 按 session_id 持久化多轮消息，解决 v1 中每个请求 new AgentEngine() 
  导致上下文当场丢失的问题。零外部依赖、单文件、天然支持多会话。
- 预留三层记忆的扩展位（当前实现第一层：session 短期记忆）：
    L1 session 短期记忆   —— 已实现（本文件）
    L2 历史摘要           —— 预留（summarize 钩子）
    L3 长期向量记忆       —— 预留（交由 knowledge/ 层或后续接入）
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Optional

DEFAULT_DB = os.getenv(
    "MEMORY_DB",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "memory.sqlite3"),
)


class SessionMemory:
    """SQLite 会话记忆。线程/进程安全性依赖 sqlite 自身的文件锁，够内部服务场景用。"""

    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT,
                    tool_calls TEXT,
                    name       TEXT,
                    ts         REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id, id)"
            )

    def append(self, session_id: str, role: str, content: Optional[str] = None,
               tool_calls: Optional[list] = None, name: Optional[str] = None) -> None:
        """追加一条消息。tool_calls / name 用于承载原生 tool_calls 协议的消息。"""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages(session_id, role, content, tool_calls, name, ts) "
                "VALUES(?,?,?,?,?,?)",
                (
                    session_id,
                    role,
                    content,
                    json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
                    name,
                    time.time(),
                ),
            )

    def load(self, session_id: str, limit: Optional[int] = None) -> list[dict]:
        """
        读取某会话的消息，还原成 OpenAI messages 结构（含 tool_calls / tool 角色）。
        limit 为 None 时返回全部；否则返回最近 limit 条。
        """
        sql = "SELECT role, content, tool_calls, name FROM messages WHERE session_id=? ORDER BY id"
        with self._conn() as conn:
            rows = conn.execute(sql, (session_id,)).fetchall()
        msgs: list[dict] = []
        for r in rows:
            m: dict = {"role": r["role"]}
            if r["content"] is not None:
                m["content"] = r["content"]
            if r["name"]:
                m["name"] = r["name"]
            if r["tool_calls"]:
                payload = json.loads(r["tool_calls"])
                if r["role"] == "tool" and isinstance(payload, dict) and "tool_call_id" in payload:
                    # tool 角色消息：还原 OpenAI 协议要求的 tool_call_id 关联字段
                    m["tool_call_id"] = payload["tool_call_id"]
                else:
                    # assistant 角色消息：还原 tool_calls 数组
                    m["tool_calls"] = payload
            msgs.append(m)
        if limit is not None and len(msgs) > limit:
            msgs = msgs[-limit:]
        return msgs

    def clear(self, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))

    # --- 预留：L2 历史摘要钩子（当前不实现，接入后填充）---
    def summarize_if_needed(self, session_id: str, max_keep: int = 20) -> None:
        """当消息过多时把早期对话压缩成摘要。预留钩子，待接入摘要模型后实现。"""
        return None
