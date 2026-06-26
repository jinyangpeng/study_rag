"""Dense 检索引擎 — 向量语义检索。

流程：embedding -> vector search -> 可选 rerank
适用场景：语义理解要求高、查询自然语言化的场景。
"""

from __future__ import annotations

import time
from typing import Any

from ..embedding.base import Embedder
from ..reranker.base import Reranker
from ..vector_store.base import VectorStore
from .base import (
    DenseParams,
    RetrievalEngine,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalStrategy,
    register_retrieval_engine,
)


@register_retrieval_engine(RetrievalStrategy.DENSE)
class DenseRetrievalEngine(RetrievalEngine):
    """Dense 检索引擎：基于向量相似度的语义检索。

    工作流程:
      1. 将 query 编码为向量（embedder）
      2. 在向量库中检索 top_k * over_fetch_factor 个候选
      3. 可选 rerank 重排后截断到 top_k

    参数:
      over_fetch_factor: 多召回倍率（启用 rerank 时扩大候选数）
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_store: VectorStore,
        collection: str,
        reranker: Reranker | None = None,
        params: dict[str, Any] | None = None,
    ):
        self._embedder = embedder
        self._vector_store = vector_store
        self._collection = collection
        self._reranker = reranker
        self._params = DenseParams(**(params or {}))

    @property
    def strategy(self) -> RetrievalStrategy:
        return RetrievalStrategy.DENSE

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        start = time.perf_counter()

        # 1. Embedding
        query_vector = await self._embedder.embed_query(request.query)

        # 2. 向量检索（多召回便于 rerank）
        over_fetch = self._params.over_fetch_factor
        candidate_k = request.top_k * over_fetch if request.use_rerank and self._reranker else request.top_k
        candidates = await self._vector_store.search(
            collection=self._collection,
            query_vector=query_vector,
            top_k=candidate_k,
            filter_expr=request.filter_expr,
        )

        # 3. Rerank
        if request.use_rerank and self._reranker:
            results = await self._apply_rerank(
                reranker=self._reranker,
                query=request.query,
                candidates=candidates,
                top_k=request.top_k,
                fallback_top_k=request.top_k,
            )
        else:
            results = candidates[: request.top_k]

        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

        return RetrievalResponse(
            kb_id=request.kb_id,
            query=request.query,
            strategy=RetrievalStrategy.DENSE,
            results=results,
            meta={
                "duration_ms": elapsed_ms,
                "candidates_fetched": len(candidates),
                "over_fetch_factor": over_fetch if request.use_rerank and self._reranker else 1,
                "reranked": request.use_rerank and self._reranker is not None,
            },
        )
