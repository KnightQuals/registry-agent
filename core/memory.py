"""
core/memory.py — 会话记忆层（Context 层）

三层记忆设计：
  L1 session 短期记忆：SQLite 保存最近、完整且符合 tool_calls 协议的消息。
  L2 历史摘要：会话过长时，将「已结束的早期回合」压缩成一条结构化摘要，
                保留最近窗口供模型继续使用，避免上下文无限增长。
  L3 长期向量记忆：预留给 knowledge/（真实 RAG 接入后实现）。

为什么不用直接截断：
- 盲目删消息会丢失用户偏好、已完成任务和工具结论；
- 更危险的是可能截断 assistant.tool_calls 与 tool 结果的配对，导致 OpenAI 协议无效。

本实现只在「一个自然语言 assistant 最终答复」处切分，保证被保留的最近消息
不会出现 orphan tool message；早期内容先由模型压缩，再在 SQLite 标记 compacted，
既减小每次请求的上下文，又保留原始记录供审计。
"""

from __future__ import annotations

import inspect
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Awaitable, Callable, Iterator, Optional

DEFAULT_DB = os.getenv(
    "MEMORY_DB",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "memory.sqlite3"),
)

SummaryFn = Callable[[str], Awaitable[str] | str]


class SessionMemory:
    """SQLite 会话记忆：L1 原消息 + L2 历史摘要。"""

    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """短连接上下文：提交/回滚后立即 close，避免 Windows 下 SQLite 文件长期被锁。"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

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
                    ts         REAL NOT NULL,
                    compacted  INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            # 兼容 v2.0 已创建的旧数据库：SQLite CREATE IF NOT EXISTS 不会新增列，故显式迁移。
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
            if "compacted" not in columns:
                conn.execute("ALTER TABLE messages ADD COLUMN compacted INTEGER NOT NULL DEFAULT 0")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id, id)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_summaries (
                    session_id       TEXT PRIMARY KEY,
                    summary          TEXT NOT NULL,
                    covered_until_id INTEGER NOT NULL,
                    updated_at       REAL NOT NULL
                )
                """
            )

    def append(
        self,
        session_id: str,
        role: str,
        content: Optional[str] = None,
        tool_calls: Optional[list | dict] = None,
        name: Optional[str] = None,
    ) -> None:
        """追加一条原始消息。tool_calls / name 承载原生 tool_calls 协议字段。"""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages(session_id, role, content, tool_calls, name, ts, compacted) "
                "VALUES(?,?,?,?,?,?,0)",
                (
                    session_id,
                    role,
                    content,
                    json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
                    name,
                    time.time(),
                ),
            )

    def _active_rows(self, session_id: str) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT id, role, content, tool_calls, name FROM messages "
                "WHERE session_id=? AND compacted=0 ORDER BY id",
                (session_id,),
            ).fetchall()

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> dict:
        """把数据库行还原为 OpenAI messages 结构。"""
        m: dict = {"role": row["role"]}
        if row["content"] is not None:
            m["content"] = row["content"]
        if row["name"]:
            m["name"] = row["name"]
        if row["tool_calls"]:
            payload = json.loads(row["tool_calls"])
            if row["role"] == "tool" and isinstance(payload, dict) and "tool_call_id" in payload:
                m["tool_call_id"] = payload["tool_call_id"]
            else:
                m["tool_calls"] = payload
        return m

    def get_summary(self, session_id: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT summary FROM session_summaries WHERE session_id=?", (session_id,)
            ).fetchone()
        return row["summary"] if row else None

    def load(
        self,
        session_id: str,
        limit: Optional[int] = None,
        include_summary: bool = True,
    ) -> list[dict]:
        """
        读取会话的「L2 摘要 + 未压缩 L1 消息」，还原成 OpenAI messages 结构。
        limit 仅限制未压缩 L1 消息；摘要永远保留，防止丢失早期关键信息。
        """
        rows = self._active_rows(session_id)
        if limit is not None and len(rows) > limit:
            rows = rows[-limit:]
        msgs = [self._row_to_message(row) for row in rows]

        if include_summary:
            summary = self.get_summary(session_id)
            if summary:
                msgs.insert(0, {
                    "role": "system",
                    "content": "以下是本会话较早历史的压缩摘要，请作为背景事实使用：\n" + summary,
                })
        return msgs

    @staticmethod
    def _is_safe_summary_boundary(row: sqlite3.Row) -> bool:
        """
        只有自然语言 assistant 最终回复后才能压缩。
        这样不会把 assistant.tool_calls 和紧随其后的 tool 消息拆开。
        """
        return row["role"] == "assistant" and not row["tool_calls"]

    @staticmethod
    def _format_rows_for_summary(rows: list[sqlite3.Row]) -> str:
        """把旧消息转成稳定、便于摘要模型理解的纯文本。"""
        parts: list[str] = []
        for row in rows:
            role = row["role"]
            name = f"({row['name']})" if row["name"] else ""
            content = row["content"] or ""
            if row["tool_calls"]:
                content += f"\n工具调用元数据: {row['tool_calls']}"
            parts.append(f"[{role}{name}]\n{content}")
        return "\n\n".join(parts)

    async def summarize_if_needed(
        self,
        session_id: str,
        summarizer: SummaryFn,
        max_messages: int = 30,
        keep_recent: int = 12,
    ) -> bool:
        """
        L2 历史摘要。

        仅当未压缩消息数超过 max_messages 时工作；保留至少 keep_recent 条原消息，
        将其前面已完整结束的回合压缩。summarizer 由 Agent 注入，通常调用同一模型。
        返回是否真的完成了一次压缩。
        """
        rows = self._active_rows(session_id)
        if len(rows) <= max_messages:
            return False

        # 候选区域不包含最新 keep_recent 条；再向前找一个安全切点。
        candidate_end = len(rows) - keep_recent
        safe_index: Optional[int] = None
        for i in range(candidate_end - 1, -1, -1):
            if self._is_safe_summary_boundary(rows[i]):
                safe_index = i
                break
        if safe_index is None:
            return False  # 当前都是未结束的工具交互，宁可暂不压缩也不破坏协议。

        rows_to_compact = rows[:safe_index + 1]
        old_summary = self.get_summary(session_id)
        transcript = self._format_rows_for_summary(rows_to_compact)
        prompt = (
            "请把下面的早期对话压缩成可供后续 Agent 使用的中文工作记忆。\n"
            "必须保留：用户目标/偏好、已确认的事实、关键工具结果、已完成或未完成事项、\n"
            "以及后续继续任务所需的约束。不要编造，不要复述客套话，控制在 600 字以内。\n\n"
        )
        if old_summary:
            prompt += "已有历史摘要（请与新内容合并更新）：\n" + old_summary + "\n\n"
        prompt += "需要压缩的新对话：\n" + transcript

        result = summarizer(prompt)
        summary = await result if inspect.isawaitable(result) else result
        summary = (summary or "").strip()
        if not summary:
            return False

        covered_until = rows_to_compact[-1]["id"]
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO session_summaries(session_id, summary, covered_until_id, updated_at)
                VALUES(?,?,?,?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary=excluded.summary,
                    covered_until_id=excluded.covered_until_id,
                    updated_at=excluded.updated_at
                """,
                (session_id, summary, covered_until, time.time()),
            )
            conn.execute(
                "UPDATE messages SET compacted=1 WHERE session_id=? AND id<=? AND compacted=0",
                (session_id, covered_until),
            )
        return True

    def clear(self, session_id: str) -> None:
        """清空会话的原始消息和 L2 摘要。"""
        with self._conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM session_summaries WHERE session_id=?", (session_id,))
