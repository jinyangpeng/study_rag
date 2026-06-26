"""Milvus 2.5+ 原生 BM25 检索引擎。

利用 Milvus 2.5 的内置 Function(BM25) 能力，在服务端完成全文检索，
无需纯 Python BM25 索引、无需额外 Embedding 模型。

提供两种策略：
  - MilvusSparseRetrievalEngine  (SPARSE_MILVUS) — 纯 BM25 全文检索
  - MilvusHybridRetrievalEngine  (HYBRID_MILVUS) — Dense + BM25 + RRF 融合

前提条件：
  - Milvus >= 2.5
  - pymilvus >= 2.5
  - collection 须由 MilvusVectorStore.create_collection_with_bm25() 创建

优势（对比纯 Python SparseRetrievalEngine）：
  - 无内存索引，文档增删即时生效（无需 invalidate_index）
  - 无冷启动延迟（不需全量拉取文档构建索引）
  - 支持超大规模文档集（受限于 Milvus 而非内存）
  - Hybrid 在服务端 RRF 融合，减少网络传输
"""

from __future__ import annotations

import time
from typing import Any

from ...observability.logging import get_logger
from ..embedding.base import Embedder
from ..reranker.base import Reranker
from ..vector_store.base import VectorStore
from .base import (
    MilvusBM25Params,
    RetrievalEngine,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalStrategy,
    register_retrieval_engine,
)

logger = get_logger(__name__)


def _has_milvus_bm25_methods(vector_store: VectorStore) -> bool:
    """检查 vector_store 是否支持 Milvus BM25 方法。"""
    return (
        hasattr(vector_store, "search_sparse")
        and hasattr(vector_store, "hybrid_search")
    )


@register_retrieval_engine(RetrievalStrategy.SPARSE_MILVUS)
class MilvusSparseRetrievalEngine(RetrievalEngine):
    """Milvus 2.5+ 原生 BM25 全文检索引擎。

    工作流程:
      1. 直接将查询文本传给 Milvus，服务端自动分词 + BM25 评分
      2. 可选 rerank 重排后截断到 top_k

    优势:
      - 无需纯 Python BM25 索引（无内存占用、无冷启动）
      - 文档增删即时生效（Milvus 服务端维护倒排索引）
      - 支持超大规模文档集
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
        if not _has_milvus_bm25_methods(vector_store):
            raise ValueError(
                "MilvusSparseRetrievalEngine requires a MilvusVectorStore with "
                "BM25 support (Milvus 2.5+). Got: "
                f"{vector_store.__class__.__name__}"
            )
        self._embedder = embedder  # 保留用于 rerank（BM25 本身不需要 embedding）
        self._vector_store = vector_store
        self._collection = collection
        self._reranker = reranker
        self._params = MilvusBM25Params(**(params or {}))

    @property
    def strategy(self) -> RetrievalStrategy:
        return RetrievalStrategy.SPARSE_MILVUS

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        start = time.perf_counter()

        # 多召回数（为 rerank 准备候选）
        over_fetch = self._params.over_fetch_factor
        candidate_k = (
            request.top_k * over_fetch
            if request.use_rerank and self._reranker
            else request.top_k
        )

        # Milvus BM25 全文检索（直接传文本，无需 embedding）
        candidates = await self._vector_store.search_sparse(
            collection=self._collection,
            query_text=request.query,
            top_k=candidate_k,
            filter_expr=request.filter_expr,
        )

        # Rerank
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
            strategy=RetrievalStrategy.SPARSE_MILVUS,
            results=results,
            meta={
                "duration_ms": elapsed_ms,
                "candidates_fetched": len(candidates),
                "over_fetch_factor": over_fetch if request.use_rerank and self._reranker else 1,
                "reranked": request.use_rerank and self._reranker is not None,
                "backend": "milvus_bm25",
            },
        )


@register_retrieval_engine(RetrievalStrategy.HYBRID_MILVUS)
class MilvusHybridRetrievalEngine(RetrievalEngine):
    """Milvus 2.5+ 原生 Dense + BM25 混合检索引擎。

    工作流程:
      1. 将 query 文本同时做 dense embedding 和 BM25 检索
      2. 用 Milvus 服务端 RRFRanker 融合两路结果
      3. 可选 rerank 重排后截断到 top_k

    优势（对比纯 Python HybridRetrievalEngine）:
      - RRF 融合在 Milvus 服务端完成，减少网络传输
      - 无需客户端维护 BM25 索引
      - 单次 hybrid_search 调用完成两路检索 + 融合
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
        if not _has_milvus_bm25_methods(vector_store):
            raise ValueError(
                "MilvusHybridRetrievalEngine requires a MilvusVectorStore with "
                "BM25 support (Milvus 2.5+). Got: "
                f"{vector_store.__class__.__name__}"
            )
        self._embedder = embedder
        self._vector_store = vector_store
        self._collection = collection
        self._reranker = reranker
        self._params = MilvusBM25Params(**(params or {}))

    @property
    def strategy(self) -> RetrievalStrategy:
        return RetrievalStrategy.HYBRID_MILVUS

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        start = time.perf_counter()

        # Dense embedding
        query_vector = await self._embedder.embed_query(request.query)

        # 多召回数
        over_fetch = self._params.over_fetch_factor
        candidate_k = (
            request.top_k * over_fetch
            if request.use_rerank and self._reranker
            else request.top_k
        )

        # Milvus 服务端 Dense + BM25 + RRF 融合
        try:
            candidates = await self._vector_store.hybrid_search(
                collection=self._collection,
                query_vector=query_vector,
                query_text=request.query,
                top_k=candidate_k,
                filter_expr=request.filter_expr,
                dense_weight=self._params.dense_weight,
                rrf_k=self._params.rrf_k,
            )
        except Exception as e:
            # hybrid_search 可能因 Milvus 版本不支持而失败
            # 降级为纯 dense 检索（保证可用性）
            logger.warning(
                "milvus_hybrid_search_failed_fallback_to_dense",
                error=str(e),
            )
            candidates = await self._vector_store.search(
                collection=self._collection,
                query_vector=query_vector,
                top_k=candidate_k,
                filter_expr=request.filter_expr,
            )

        # Rerank
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
            strategy=RetrievalStrategy.HYBRID_MILVUS,
            results=results,
            meta={
                "duration_ms": elapsed_ms,
                "candidates_fetched": len(candidates),
                "over_fetch_factor": over_fetch if request.use_rerank and self._reranker else 1,
                "dense_weight": self._params.dense_weight,
                "rrf_k": self._params.rrf_k,
                "reranked": request.use_rerank and self._reranker is not None,
                "backend": "milvus_hybrid",
            },
        )
