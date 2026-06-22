"""Retrieval 能力抽象。

封装端到端的检索流程：embedding -> vector search -> rerank。
后续可接入 LlamaIndex 的 QueryEngine 做更复杂的索引与响应合成。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ..embedding.base import Embedder, EmbeddingConfig, create_embedder
from ..reranker.base import Reranker, RerankerConfig, create_reranker
from ..vector_store.base import SearchResult, VectorStore, VectorStoreConfig, create_vector_store


class RetrievalRequest(BaseModel):
    """检索请求。"""

    kb_id: str
    query: str
    top_k: int = 5
    use_rerank: bool = True


class RetrievalResponse(BaseModel):
    """检索响应。"""

    kb_id: str
    query: str
    results: list[SearchResult]


@runtime_checkable
class RetrievalEngine(Protocol):
    """检索引擎接口。"""

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse: ...


class DefaultRetrievalEngine:
    """默认检索引擎：embedder + vector_store + 可选 reranker。"""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        reranker: Reranker | None = None,
    ):
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        # 1. Embedding
        query_vector = await self._embedder.embed_query(request.query)

        # 2. 检索（多召回一些，便于 rerank）
        candidate_k = request.top_k * 4 if request.use_rerank and self._reranker else request.top_k
        candidates = await self._vector_store.search(
            collection=request.kb_id,
            query_vector=query_vector,
            top_k=candidate_k,
        )

        # 3. Rerank（可选）
        if request.use_rerank and self._reranker:
            results = await self._reranker.rerank(
                query=request.query,
                results=candidates,
                top_k=request.top_k,
            )
        else:
            results = candidates[: request.top_k]

        return RetrievalResponse(
            kb_id=request.kb_id,
            query=request.query,
            results=results,
        )


# 工厂：根据配置组装
def build_default_retrieval_engine(
    embedder_config: EmbeddingConfig,
    vector_store_config: VectorStoreConfig,
    reranker_config: RerankerConfig | None = None,
) -> RetrievalEngine:
    """构建默认检索引擎。"""
    embedder = create_embedder(embedder_config)
    vector_store = create_vector_store(vector_store_config)
    reranker = create_reranker(reranker_config) if reranker_config else None
    return DefaultRetrievalEngine(embedder, vector_store, reranker)
