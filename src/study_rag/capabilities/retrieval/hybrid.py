"""Hybrid 检索引擎 — Dense + Sparse 融合检索。

流程：Dense 检索 + Sparse 检索 -> Reciprocal Rank Fusion (RRF) -> 可选 rerank
适用场景：需要兼顾语义理解和关键词精确匹配的通用场景。

融合算法：Reciprocal Rank Fusion (RRF)
  score(d) = Σ 1 / (k + rank_i(d)) * weight_i
  其中 k 为常数（默认 60），rank_i 为文档在第 i 路检索中的排名，weight_i 为权重。
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any

from ...observability.logging import get_logger
from ..embedding.base import Embedder
from ..reranker.base import Reranker
from ..vector_store.base import SearchResult, VectorStore
from .base import (
    HybridParams,
    RetrievalEngine,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalStrategy,
    register_retrieval_engine,
)
from .dense import DenseRetrievalEngine
from .sparse import SparseRetrievalEngine

logger = get_logger(__name__)


def _reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    weights: list[float],
    k: int = 60,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion (RRF)。

    将多路检索结果按排名融合为统一排序。

    Args:
        result_lists: 多路检索结果列表
        weights: 每路权重（与 result_lists 一一对应）
        k: RRF 常数（越大分数越平滑，推荐 60）

    Returns:
        融合后的结果列表（按 RRF 分数降序）
    """
    if not result_lists:
        return []

    # doc_id -> RRF 分数
    doc_scores: dict[str, float] = defaultdict(float)
    # doc_id -> 最佳 SearchResult（保留最完整的信息）
    doc_best: dict[str, SearchResult] = {}

    for results, weight in zip(result_lists, weights, strict=True):
        for rank, result in enumerate(results):
            rrf_score = weight / (k + rank + 1)  # rank 从 0 开始，+1 保证分母 > 0
            doc_scores[result.id] += rrf_score
            # 保留 score 最高那路的结果（取 max 而非覆盖，保留 metadata 最丰富的）
            if result.id not in doc_best or result.score > doc_best[result.id].score:
                doc_best[result.id] = result

    # 按 RRF 分数降序排列
    sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    fused: list[SearchResult] = []
    for doc_id in sorted_ids:
        original = doc_best[doc_id]
        fused.append(
            SearchResult(
                id=original.id,
                text=original.text,
                score=doc_scores[doc_id],  # 用 RRF 分数替换原始分数
                metadata={**original.metadata, "_rrf_score": doc_scores[doc_id]},
            )
        )

    return fused


@register_retrieval_engine(RetrievalStrategy.HYBRID)
class HybridRetrievalEngine(RetrievalEngine):
    """Hybrid 检索引擎：Dense + Sparse 融合检索。

    工作流程:
      1. 并行执行 Dense 检索（语义向量）和 Sparse 检索（BM25 关键词）
      2. 用 Reciprocal Rank Fusion (RRF) 融合两路结果
      3. 可选 rerank 重排后截断到 top_k

    优势:
      - 兼顾语义理解（"如何优化性能" ≈ "性能调优"）和关键词精确匹配
      - RRF 融合无需分数归一化，简单可靠
      - 对不同量纲的分数（向量余弦 vs BM25）天然兼容

    参数:
      dense_weight: Dense 结果权重（0~1），Sparse 权重 = 1 - dense_weight
      rrf_k: RRF 常数 k
      over_fetch_factor: Dense 多召回倍率
      k1, b: BM25 参数
      use_jieba: 是否使用 jieba 中文分词
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
        self._params = HybridParams(**(params or {}))

        # 构造子引擎
        sparse_params = {
            "k1": self._params.k1,
            "b": self._params.b,
            "use_jieba": self._params.use_jieba,
            "stop_words": self._params.stop_words,
        }
        dense_params = {"over_fetch_factor": self._params.over_fetch_factor}

        self._dense_engine = DenseRetrievalEngine(
            embedder=embedder,
            vector_store=vector_store,
            collection=collection,
            reranker=None,  # rerank 在 Hybrid 层统一处理
            params=dense_params,
        )
        self._sparse_engine = SparseRetrievalEngine(
            embedder=embedder,
            vector_store=vector_store,
            collection=collection,
            reranker=None,  # rerank 在 Hybrid 层统一处理
            params=sparse_params,
        )

    @property
    def strategy(self) -> RetrievalStrategy:
        return RetrievalStrategy.HYBRID

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        start = time.perf_counter()

        # 多召回数量（两路各自多召回，融合后再截断）
        fetch_k = request.top_k * max(
            self._params.over_fetch_factor, 4
        )

        # 构造子请求（禁用子引擎的 rerank，融合后统一 rerank）
        dense_request = RetrievalRequest(
            kb_id=request.kb_id,
            query=request.query,
            top_k=fetch_k,
            use_rerank=False,
            filter_expr=request.filter_expr,
        )
        sparse_request = RetrievalRequest(
            kb_id=request.kb_id,
            query=request.query,
            top_k=fetch_k,
            use_rerank=False,
            filter_expr=request.filter_expr,
        )

        # 并行执行两路检索
        dense_task = asyncio.create_task(self._dense_engine.retrieve(dense_request))
        sparse_task = asyncio.create_task(self._sparse_engine.retrieve(sparse_request))
        dense_response, sparse_response = await asyncio.gather(
            dense_task, sparse_task, return_exceptions=True
        )

        # 容错：单路失败不影响整体检索
        dense_results: list[SearchResult] = []
        sparse_results: list[SearchResult] = []
        if isinstance(dense_response, Exception):
            logger.warning("hybrid_dense_failed", error=str(dense_response))
        else:
            dense_results = dense_response.results

        if isinstance(sparse_response, Exception):
            logger.warning("hybrid_sparse_failed", error=str(sparse_response))
        else:
            sparse_results = sparse_response.results

        # RRF 融合
        fused = _reciprocal_rank_fusion(
            result_lists=[dense_results, sparse_results],
            weights=[self._params.dense_weight, 1.0 - self._params.dense_weight],
            k=self._params.rrf_k,
        )

        # 截断到 top_k * 4（为 rerank 准备候选）
        fused_candidates = fused[: fetch_k]

        # Rerank
        if request.use_rerank and self._reranker:
            results = await self._apply_rerank(
                reranker=self._reranker,
                query=request.query,
                candidates=fused_candidates,
                top_k=request.top_k,
                fallback_top_k=request.top_k,
            )
        else:
            results = fused_candidates[: request.top_k]

        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

        return RetrievalResponse(
            kb_id=request.kb_id,
            query=request.query,
            strategy=RetrievalStrategy.HYBRID,
            results=results,
            meta={
                "duration_ms": elapsed_ms,
                "dense_count": len(dense_results),
                "sparse_count": len(sparse_results),
                "fused_count": len(fused),
                "dense_weight": self._params.dense_weight,
                "rrf_k": self._params.rrf_k,
                "reranked": request.use_rerank and self._reranker is not None,
            },
        )
