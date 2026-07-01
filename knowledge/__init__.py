"""knowledge 层：RAG / 知识库接口（当前为占位实现）。"""

from .base import KnowledgeBase, NullKnowledgeBase, Passage, register_as_tool

__all__ = ["KnowledgeBase", "NullKnowledgeBase", "Passage", "register_as_tool"]
