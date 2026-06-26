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


def _is_bm25_schema_error(err: Exception) -> bool:
    """判断异常是否因 collection 缺少 BM25 字段（sparse_bm25）导致。"""
    msg = str(err).lower()
    return (
        "sparse_bm25" in msg
        or "field schema by name" in msg
        or ("failed to create query plan" in msg and "not found" in msg)
    )


def _wrap_bm25_schema_error(collection: str, err: Exception) -> Exception:
    """若 err 是 BM25 schema 缺失错误，返回清晰可操作的错误；否则原样返回。"""
    if _is_bm25_schema_error(err):
        return ValueError(
            f"Collection '{collection}' 不支持 BM25 全文检索（缺少 sparse_bm25 字段）。"
            f"该 collection 是 dense-only schema，无法使用 sparse_milvus / hybrid_milvus 策略。"
            f"请调用 POST /admin/kbs/{{kb_id}}/recreate-collection 重建为 BM25 schema"
            f"（会保留已有文档数据），或将检索策略改为 dense / sparse / hybrid。"
        )
    return err


def _raise_if_schema_mismatch(collection: str, err: Exception) -> None:
    """若是 BM25 schema 缺失错误，抛出清晰错误；否则不抛（交给调用方降级）。"""
    wrapped = _wrap_bm25_schema_error(collection, err)
    if wrapped is not err:
        raise wrapped


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
        try:
            candidates = await self._vector_store.search_sparse(
                collection=self._collection,
                query_text=request.query,
                top_k=candidate_k,
                filter_expr=request.filter_expr,
            )
        except Exception as e:
            raise _wrap_bm25_schema_error(self._collection, e) from e

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
            # 若是 BM25 schema 缺失（collection 没有 sparse_bm25 字段），
            # 给出清晰错误而非静默降级到 dense（否则用户以为 hybrid 生效了实际没生效）
            _raise_if_schema_mismatch(self._collection, e)
            # 其它错误（如 Milvus 版本不支持 hybrid_search）降级为纯 dense 检索
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
