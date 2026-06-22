"""Mock Reranker：保持原顺序，仅做截断。"""

from __future__ import annotations

from ..vector_store.base import SearchResult
from .base import RerankerConfig, register_reranker


@register_reranker("mock")
class PassThroughReranker:
    """不做重排的 Reranker，仅用于占位和测试。"""

    def __init__(self, config: RerankerConfig):
        self._config = config

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        k = top_k or self._config.top_k
        return results[:k]


@register_reranker("none")
class NoOpReranker:
    """不重排。"""

    def __init__(self, config: RerankerConfig):
        self._config = config

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        k = top_k or self._config.top_k
        return results[:k]
