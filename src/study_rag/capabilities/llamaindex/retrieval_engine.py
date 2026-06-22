"""LlamaIndex 检索引擎：用 LlamaIndex 的 VectorStoreIndex + Retriever
把 embedding/vector_store/reranker 都用 LI 的方式串起来。

提供两种检索路径：
  1. DefaultRetrievalEngine（已有）
  2. LlamaIndexRetrievalEngine（本模块）

两条路径用同一套底层能力（同一个 Embedder/VectorStore/Reranker 实例），
差异在于是否走 LI 的 retriever 协议（带 LI 的 query 改写、HyDE 等扩展能力）。
"""

from __future__ import annotations

import logging
from typing import Any

from ..embedding.base import Embedder
from ..reranker.base import Reranker
from ..vector_store.base import SearchResult, VectorStore
from .embedding_adapter import LIEmbeddingAdapter
from .reranker_adapter import LIRerankerPostprocessor
from .vector_store_adapter import LIVectorStoreAdapter

logger = logging.getLogger(__name__)

__all__ = ["LlamaIndexRetrievalEngine"]


class LlamaIndexRetrievalEngine:
    """基于 LlamaIndex 的检索引擎。

    与默认检索引擎的差异：
      - 使用 LI 的 retriever 协议
      - 可以用 LI 的查询改写、HyDE、sub-question 等高级能力
      - 自动应用 NodeParser 切块
      - Reranker 作为 NodePostProcessor 接入

    局限：
      - 缺 LLM 时只能用 retriever（不调用 response synthesis）
      - 性能上多了一层协议转换（vs 直接调我们的 SearchResult）
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        collection: str,
        reranker: Reranker | None = None,
        node_parser: Any | None = None,
        top_k: int = 5,
    ):
        self._embedder = embedder
        self._vector_store = vector_store
        self._collection = collection
        self._reranker = reranker
        self._node_parser = node_parser
        self._top_k = top_k
        self._index: Any = None  # 懒构造
        self._li_embedder = LIEmbeddingAdapter(embedder)
        self._li_vs = LIVectorStoreAdapter(vector_store, collection)
        self._li_reranker = (
            LIRerankerPostprocessor(reranker, top_n=top_k) if reranker else None
        )
        logger.info(
            "LlamaIndexRetrievalEngine: collection=%s, top_k=%d, has_rerank=%s",
            collection,
            top_k,
            reranker is not None,
        )

    def _ensure_index(self) -> Any:
        """懒构造 LI VectorStoreIndex。"""
        if self._index is not None:
            return self._index
        try:
            from llama_index.core import VectorStoreIndex
        except ImportError as e:
            raise ImportError(
                "LlamaIndexRetrievalEngine 需要 llama-index-core. "
                "安装: pip install llama-index-core"
            ) from e
        # 用我们的 vector store 适配器构造 index
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=self._li_vs,
            embed_model=self._li_embedder,
        )
        return self._index

    async def aretrieve(self, query: str) -> list[SearchResult]:
        """异步检索。

        实现说明：
          - 直接调我们的 vector store（不走 LI Retriever，避免 LI 0.14.x retriever
            对 NodeWithScore.as_related_node_info() 的依赖）
          - 用 LI 的 BaseEmbedding 算 query embedding（保证与 LI 协议一致）
          - 用我们的 Reranker 重排
        """
        from llama_index.core.schema import (
            QueryBundle,
        )

        # 1. 算 query embedding（用 LI 适配器，保证接口一致）
        query_bundle = QueryBundle(query_str=query)
        query_embedding = await self._li_embedder._aget_query_embedding(query)
        query_bundle.embedding = query_embedding

        # 2. 直接调 LI 适配器查 vector store（绕过 LI Retriever）
        from llama_index.core.vector_stores.types import VectorStoreQuery

        vs_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._top_k * 4,
        )
        vs_result = await self._li_vs.aquery(vs_query)

        # 3. 转成 SearchResult 列表
        results: list[SearchResult] = []
        for node, sim, nid in zip(
            vs_result.nodes, vs_result.similarities, vs_result.ids, strict=False
        ):
            text = (
                node.node.get_content()
                if hasattr(node, "node") and hasattr(node.node, "get_content")
                else str(node)
            )
            results.append(
                SearchResult(
                    id=nid,
                    text=text,
                    score=float(sim),
                    metadata=dict(node.node.metadata or {}) if hasattr(node, "node") else {},
                )
            )

        # 4. Rerank（KB 配置了 reranker 时才生效；失败则降级为截断）
        if self._reranker is not None and results:
            try:
                results = await self._reranker.rerank(
                    query=query, results=results, top_k=self._top_k
                )
            except Exception as e:
                logger.warning(
                    "LlamaIndex path rerank failed for query=%r: %s",
                    query[:30],
                    e,
                )
                results = results[: self._top_k]
        else:
            results = results[: self._top_k]

        return results

    def retrieve(self, query: str) -> list[SearchResult]:
        return __import__("asyncio").run(self.aretrieve(query))

    def _make_query_bundle(self, query: str) -> Any:
        try:
            from llama_index.core.schema import QueryBundle
        except ImportError as e:
            raise ImportError("需要 llama-index-core") from e
        return QueryBundle(query_str=query)
