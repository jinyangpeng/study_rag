"""Retrieval 能力包 — 策略模式架构。

提供三种检索策略：
  - DenseRetrievalEngine   — 向量语义检索
  - SparseRetrievalEngine  — BM25 关键词检索
  - HybridRetrievalEngine  — Dense + Sparse 融合检索

使用方式:
    from study_rag.capabilities.retrieval import (
        create_retrieval_engine,
        RetrievalStrategy,
        RetrievalRequest,
    )

    engine = create_retrieval_engine(
        strategy=RetrievalStrategy.HYBRID,
        embedder=embedder,
        vector_store=vector_store,
        collection="kb_rd_frontend",
        reranker=reranker,
        params={"dense_weight": 0.6, "rrf_k": 60},
    )
    response = await engine.retrieve(RetrievalRequest(
        kb_id="rd_frontend",
        query="React 性能优化",
        top_k=5,
    ))
"""

from .base import (
    DefaultRetrievalEngine,
    DenseParams,
    HybridParams,
    MilvusBM25Params,
    RetrievalConfig,
    RetrievalEngine,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalStrategy,
    SparseParams,
    build_default_retrieval_engine,
    create_retrieval_engine,
    list_retrieval_strategies,
    register_retrieval_engine,
)
from .dense import DenseRetrievalEngine
from .hybrid import HybridRetrievalEngine
from .milvus_bm25 import MilvusHybridRetrievalEngine, MilvusSparseRetrievalEngine
from .sparse import SparseRetrievalEngine

__all__ = [
    # 枚举 & 配置
    "RetrievalStrategy",
    "RetrievalConfig",
    "DenseParams",
    "SparseParams",
    "HybridParams",
    "MilvusBM25Params",
    # 请求/响应
    "RetrievalRequest",
    "RetrievalResponse",
    # 引擎
    "RetrievalEngine",
    "DenseRetrievalEngine",
    "SparseRetrievalEngine",
    "HybridRetrievalEngine",
    "MilvusSparseRetrievalEngine",
    "MilvusHybridRetrievalEngine",
    "DefaultRetrievalEngine",
    # 工厂
    "create_retrieval_engine",
    "build_default_retrieval_engine",
    "register_retrieval_engine",
    "list_retrieval_strategies",
]
