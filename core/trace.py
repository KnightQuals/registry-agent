"""
core/trace.py — 结构化可观测层（Observability）

设计要点（对应 Harness 认知）：
1. 每一步 Agent 决策 / 工具调用都记录一条结构化 trace，出问题可回溯。
2. "沉默即成功"（Silence is Success）：工具成功只回极简摘要，失败才输出完整细节，
   避免大量成功日志淹没关键失败信号、撑爆上下文（实证可显著提升长任务达成率）。

trace 以单行 JSON 写入日志文件，grep 友好；同时可选打印到控制台。
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


DEFAULT_TRACE_DIR = os.getenv("TRACE_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs"))


@dataclass
class TraceEvent:
    """一条结构化 trace 事件。"""
    session_id: str
    step: int
    kind: str  # reason | tool_call | tool_result | final | error | mcp | system
    name: str = ""
    detail: dict = field(default_factory=dict)
    ok: bool = True
    ts: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class Tracer:
    """会话级 tracer：按 session 累加 step，单行 JSON 落盘 + 可选控制台。"""

    def __init__(self, session_id: str, to_console: bool = True, trace_dir: str = DEFAULT_TRACE_DIR):
        self.session_id = session_id
        self.to_console = to_console
        self._step = 0
        os.makedirs(trace_dir, exist_ok=True)
        self._path = os.path.join(trace_dir, f"{session_id}.jsonl")

    def _next_step(self) -> int:
        self._step += 1
        return self._step

    def log(self, kind: str, name: str = "", detail: Optional[dict] = None, ok: bool = True) -> TraceEvent:
        ev = TraceEvent(
            session_id=self.session_id,
            step=self._next_step(),
            kind=kind,
            name=name,
            detail=detail or {},
            ok=ok,
        )
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(ev.to_line() + "\n")
        except Exception:
            pass  # 观测不应影响主流程
        if self.to_console:
            icon = "✓" if ok else "✗"
            print(f"[trace {icon}] step={ev.step} {kind}/{name}", file=sys.stderr)
        return ev


def summarize_tool_result(name: str, ok: bool, result: Any, max_ok_chars: int = 400) -> str:
    """
    "沉默即成功" 的结果摘要器。

    - 成功：只回极简结构（工具名 + 截断结果），减少上下文噪声。
    - 失败：回完整错误细节，让模型能据此自我修正。

    返回值是要回注给模型的 observation 文本。
    """
    if not ok:
        return f"[工具 {name} 失败] {result}"
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    if len(text) > max_ok_chars:
        text = text[:max_ok_chars] + f" …(已截断，共 {len(text)} 字符)"
    return f"[工具 {name} 成功] {text}"
