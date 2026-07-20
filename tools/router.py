"""
tools/router.py — 工具路由（Tool Routing）

当工具只有 2-3 个时，把全部 schema 发给模型最简单；但接入多个 MCP 后，
把几十甚至上百个工具全塞进上下文会带来两个问题：
1. 占 token，模型推理更慢；
2. 工具描述相互干扰，容易选错工具。

本模块是一个轻量、确定性的第一层路由：根据用户问题和工具的名称、描述、参数描述
做关键词/中文 n-gram 匹配，只把最相关的候选工具交给模型。

重要的安全回退：
- 工具总数不超过 max_candidates 时，直接保留全部工具；
- 一个工具都匹配不到时，也保留全部工具；
因此路由只在「工具很多且有明确匹配」时缩小上下文，绝不因为路由漏掉能力。

这不是让小模型替大模型做规划，而是 Context Engineering：先做便宜、可解释的粗筛，
最终仍由 LLM 从候选工具里做决定。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any


DEFAULT_MAX_CANDIDATES = int(os.getenv("TOOL_ROUTER_MAX_CANDIDATES", "12"))


@dataclass
class ToolRoute:
    """一次路由的可观测结果。"""

    names: list[str]
    scores: dict[str, float]
    used_fallback: bool
    reason: str


def _normalise(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def _latin_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]{2,}", (text or "").lower()))


def _cjk_ngrams(text: str) -> set[str]:
    """从中文文本抽 2-4 字 n-gram，避免必须依赖中文分词库。"""
    chunks = re.findall(r"[\u4e00-\u9fff]{2,}", text or "")
    out: set[str] = set()
    for chunk in chunks:
        for n in (2, 3, 4):
            for i in range(max(0, len(chunk) - n + 1)):
                out.add(chunk[i:i + n])
    return out


class ToolRouter:
    """基于工具元数据的轻量候选工具筛选器。"""

    def __init__(self, max_candidates: int = DEFAULT_MAX_CANDIDATES):
        self.max_candidates = max_candidates

    def select(self, query: str, registry) -> ToolRoute:
        """
        选择要暴露给模型的候选工具名。
        registry 只需要实现 list_tools() 和 tool_search_text(name)。
        """
        all_tools = registry.list_tools()
        all_names = [t["name"] for t in all_tools]

        if len(all_names) <= self.max_candidates:
            return ToolRoute(all_names, {}, False, "工具数量未超过路由阈值，保留全部")

        q_norm = _normalise(query)
        q_latin = _latin_tokens(query)
        q_cjk = _cjk_ngrams(query)
        scores: dict[str, float] = {}

        for tool in all_tools:
            name = tool["name"]
            corpus = _normalise(registry.tool_search_text(name))
            name_norm = _normalise(name)
            score = 0.0

            # 工具名命中信号最强；如用户直接说了工具/技能名，优先让它进候选。
            if name_norm and len(name_norm) >= 3 and name_norm in q_norm:
                score += 20.0

            for token in q_latin:
                if token in name_norm:
                    score += 8.0
                elif token in corpus:
                    score += 3.0

            # 中文 2-4 gram：命中描述或参数说明时累加；长片段权重更高。
            for gram in q_cjk:
                if gram in corpus:
                    score += 0.8 + (len(gram) - 2) * 0.6

            if score > 0:
                scores[name] = round(score, 2)

        if not scores:
            return ToolRoute(all_names, {}, True, "未找到可靠关键词匹配，安全回退为全部工具")

        ordered = sorted(scores, key=lambda n: (-scores[n], n))
        selected = ordered[:self.max_candidates]
        return ToolRoute(selected, {n: scores[n] for n in selected}, False, "按工具元数据关键词粗筛")
