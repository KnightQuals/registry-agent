"""
knowledge/base.py — 知识库 / RAG 接口层（占位）

对应用户需求：实验室未来会把内部全部文档整理成知识库。现在先留接口占位，
接入后（乐享 / FAISS / Milvus / 自建向量库均可）即拥有查知识库能力，不改主流程。

设计：
- KnowledgeBase 抽象接口：核心方法 retrieve(query, top_k) -> list[Passage]。
- NullKnowledgeBase：默认实现，返回空并提示"知识库尚未接入"。
- register_as_tool()：把知识库包装成一个名为 search_knowledge_base 的工具注册进 registry，
  这样模型可以像调其他工具一样查知识库。接入真实实现后工具立即生效。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Passage:
    """一条召回的知识片段。"""
    text: str
    source: str = ""       # 来源文档/路径
    score: float = 0.0     # 相关性分数
    meta: Optional[dict] = None


class KnowledgeBase:
    """知识库抽象接口。接入实验室知识库时实现本类。"""

    name = "knowledge_base"

    def retrieve(self, query: str, top_k: int = 5) -> list[Passage]:
        raise NotImplementedError

    def is_ready(self) -> bool:
        return False


class NullKnowledgeBase(KnowledgeBase):
    """占位实现：知识库尚未接入时使用，返回空结果。"""

    def retrieve(self, query: str, top_k: int = 5) -> list[Passage]:
        return []

    def is_ready(self) -> bool:
        return False


def register_as_tool(registry, kb: KnowledgeBase) -> None:
    """
    把知识库包装成 search_knowledge_base 工具注册进 registry。
    未接入（NullKnowledgeBase）时，工具会返回"知识库尚未接入"的提示，
    不影响系统运行；接入真实实现后，同一工具立即具备检索能力。
    """

    async def _search(arguments: dict) -> str:
        query = arguments.get("query", "")
        top_k = int(arguments.get("top_k", 5))
        if not kb.is_ready():
            return "知识库尚未接入。待实验室内部文档知识库上线后，本工具将返回相关文档片段。"
        passages = kb.retrieve(query, top_k=top_k)
        if not passages:
            return "未在知识库中检索到相关内容。"
        lines = []
        for i, p in enumerate(passages, 1):
            lines.append(f"[{i}] (来源:{p.source} 相关性:{p.score:.2f}) {p.text}")
        return "\n".join(lines)

    registry.register_mcp_tool(
        name="search_knowledge_base",
        description="检索实验室内部知识库（项目文档、FAQ、会议记录、运维文档等）。当问题涉及内部资料时使用。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词或问题"},
                "top_k": {"type": "integer", "description": "返回条数，默认5"},
            },
            "required": ["query"],
        },
        mcp_call=_search,
        source="knowledge",
        original_name="search_knowledge_base",
    )
