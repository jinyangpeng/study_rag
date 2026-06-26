"""Retrieval 能力抽象 — 策略模式架构。

提供三种主流检索策略的统一接口：
  1. Dense   — 向量语义检索（embedding + vector search + 可选 rerank）
  2. Sparse  — BM25 关键词检索（jieba 分词 + 倒排索引）
  3. Hybrid  — Dense + Sparse 融合（Reciprocal Rank Fusion）

架构：
  RetrievalEngine (ABC)
    ├── DenseRetrievalEngine   — 默认，语义理解强
    ├── SparseRetrievalEngine  — 关键词精确匹配
    └── HybridRetrievalEngine  — 两者融合，兼顾语义 + 关键词
"""

from __future__ import annotations

import abc
import enum
from typing import Any

from pydantic import BaseModel, Field

from ...observability.logging import get_logger
from ..embedding.base import Embedder, EmbeddingConfig, create_embedder
from ..reranker.base import Reranker, RerankerConfig, create_reranker
from ..vector_store.base import SearchResult, VectorStore, VectorStoreConfig, create_vector_store

logger = get_logger(__name__)


# ====================================================================
#  配置模型
# ====================================================================


class RetrievalStrategy(str, enum.Enum):
    """检索策略枚举。

    纯 Python 实现（无需 Milvus 2.5+）：
      - DENSE   向量语义检索
      - SPARSE  BM25 关键词检索（纯 Python 内存索引）
      - HYBRID  Dense + Sparse 融合（客户端 RRF）

    Milvus 2.5+ 原生实现（需 Milvus 2.5+ + pymilvus 2.5+）：
      - SPARSE_MILVUS  BM25 全文检索（Milvus 服务端）
      - HYBRID_MILVUS  Dense + BM25 融合（Milvus 服务端 RRF）
    """

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    SPARSE_MILVUS = "sparse_milvus"
    HYBRID_MILVUS = "hybrid_milvus"


class DenseParams(BaseModel):
    """Dense 检索参数。"""

    over_fetch_factor: int = Field(
        default=4,
        ge=1,
        le=20,
        description="多召回倍率（启用 rerank 时 candidate_k = top_k * over_fetch_factor）",
    )


class SparseParams(BaseModel):
    """BM25 Sparse 检索参数。"""

    k1: float = Field(default=1.5, ge=0.0, le=3.0, description="BM25 词频饱和参数 k1")
    b: float = Field(default=0.75, ge=0.0, le=1.0, description="BM25 文档长度归一化参数 b")
    use_jieba: bool = Field(default=True, description="是否使用 jieba 中文分词")
    stop_words: list[str] = Field(
        default_factory=list,
        description="自定义停用词列表（可选）",
    )


class HybridParams(BaseModel):
    """Hybrid 融合检索参数。"""

    dense_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Dense 结果权重（0~1）；Sparse 权重 = 1 - dense_weight",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        le=200,
        description="Reciprocal Rank Fusion 常数 k（越大分数越平滑）",
    )
    over_fetch_factor: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Dense 多召回倍率",
    )
    k1: float = Field(default=1.5, ge=0.0, le=3.0, description="BM25 k1")
    b: float = Field(default=0.75, ge=0.0, le=1.0, description="BM25 b")
    use_jieba: bool = Field(default=True, description="是否使用 jieba 中文分词")
    stop_words: list[str] = Field(
        default_factory=list,
        description="自定义停用词列表（可选）",
    )


class MilvusBM25Params(BaseModel):
    """Milvus 2.5+ BM25 全文检索参数（sparse_milvus / hybrid_milvus 通用）。"""

    analyzer_type: str = Field(
        default="chinese",
        description="Milvus 分词器类型：chinese / english / standard",
    )
    dense_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="（仅 hybrid_milvus）Dense 权重；Sparse 权重 = 1 - dense_weight",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        le=200,
        description="（仅 hybrid_milvus）RRF 常数 k",
    )
    over_fetch_factor: int = Field(
        default=4,
        ge=1,
        le=20,
        description="多召回倍率（启用 rerank 时扩大候选数）",
    )


class RetrievalConfig(BaseModel):
    """检索引擎全局配置（对应 retrieval.yaml）。"""

    default_strategy: RetrievalStrategy = Field(
        default=RetrievalStrategy.DENSE,
        description="KB 未指定策略时使用的默认检索策略",
    )
    dense: DenseParams = Field(default_factory=DenseParams)
    sparse: SparseParams = Field(default_factory=SparseParams)
    hybrid: HybridParams = Field(default_factory=HybridParams)
    milvus_bm25: MilvusBM25Params = Field(default_factory=MilvusBM25Params)


# ====================================================================
#  请求 / 响应模型
# ====================================================================


class RetrievalRequest(BaseModel):
    """检索请求。"""

    kb_id: str = Field(..., description="知识库 ID")
    query: str = Field(..., min_length=1, description="检索查询")
    top_k: int = Field(default=5, ge=1, le=100, description="返回结果数")
    use_rerank: bool = Field(default=True, description="是否启用 rerank")
    strategy: RetrievalStrategy | None = Field(
        default=None,
        description="检索策略（None = 使用 KB 配置 / 全局默认）",
    )
    strategy_params: dict[str, Any] = Field(
        default_factory=dict,
        description="策略参数覆盖（覆盖 retrieval.yaml 中的默认值）",
    )
    filter_expr: dict[str, Any] | None = Field(
        default=None,
        description="metadata 过滤条件",
    )
    reranker_name: str | None = Field(
        default=None,
        description="显式指定 reranker 配置名（覆盖 KB 默认）",
    )


class RetrievalResponse(BaseModel):
    """检索响应。"""

    kb_id: str = Field(..., description="知识库 ID")
    query: str = Field(..., description="原始查询")
    strategy: RetrievalStrategy = Field(..., description="实际使用的检索策略")
    results: list[SearchResult] = Field(default_factory=list, description="检索结果")
    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="检索元信息（耗时、候选数等）",
    )


# ====================================================================
#  抽象基类
# ====================================================================


class RetrievalEngine(abc.ABC):
    """检索引擎抽象基类。

    所有检索策略必须继承此类并实现 retrieve 方法。
    """

    @property
    @abc.abstractmethod
    def strategy(self) -> RetrievalStrategy:
        """返回当前引擎的检索策略类型。"""
        ...

    @abc.abstractmethod
    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """执行检索。

        Args:
            request: 检索请求（含查询、top_k、过滤条件等）

        Returns:
            RetrievalResponse: 检索结果
        """
        ...

    async def _apply_rerank(
        self,
        reranker: Reranker | None,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
        fallback_top_k: int,
    ) -> list[SearchResult]:
        """通用 rerank 逻辑（失败降级为截断）。

        子类可直接调用此方法，无需重复实现。

        Args:
            reranker: reranker 实例（None 时不重排，直接截断）
            query: 查询文本
            candidates: 候选结果列表（已多召回）
            top_k: UI 传入的 Top K（= embedding 召回数，非最终返回数）
            fallback_top_k: reranker 不可用/失败时的截断数

        语义:
            - 启用 reranker 时传 top_k=None，让 reranker 用自身配置的 top_k
              （即 reranker.yaml 里配置的 top_k，如 local_bge_reranker_base 的 3）
            - 这样 UI 的 Top K 控制 embedding 召回数，reranker 配置控制最终返回数
        """
        if not candidates:
            return []

        if reranker is None:
            return candidates[:fallback_top_k]

        try:
            return await reranker.rerank(query=query, results=candidates, top_k=None)
        except Exception as e:
            logger.warning(
                "rerank_failed_fallback_to_truncation",
                strategy=self.strategy.value,
                error=str(e),
            )
            return candidates[:fallback_top_k]


# ====================================================================
#  工厂 + 注册
# ====================================================================

_RETRIEVAL_ENGINE_REGISTRY: dict[str, type[RetrievalEngine]] = {}


def register_retrieval_engine(strategy: RetrievalStrategy):
    """装饰器：注册 RetrievalEngine 实现类。"""

    def decorator(cls: type[RetrievalEngine]) -> type[RetrievalEngine]:
        _RETRIEVAL_ENGINE_REGISTRY[strategy.value] = cls
        return cls

    return decorator


def create_retrieval_engine(
    strategy: RetrievalStrategy,
    *,
    embedder: Embedder,
    vector_store: VectorStore,
    collection: str,
    reranker: Reranker | None = None,
    params: dict[str, Any] | None = None,
) -> RetrievalEngine:
    """根据策略创建检索引擎实例。

    Args:
        strategy: 检索策略
        embedder: Embedding 模型实例
        vector_store: 向量库实例
        collection: 向量库 collection 名
        reranker: 可选的 reranker 实例
        params: 策略参数覆盖

    Returns:
        对应策略的 RetrievalEngine 实例
    """
    engine_cls = _RETRIEVAL_ENGINE_REGISTRY.get(strategy.value)
    if engine_cls is None:
        raise ValueError(
            f"Unknown retrieval strategy: {strategy.value}. "
            f"Available: {list(_RETRIEVAL_ENGINE_REGISTRY.keys())}"
        )

    return engine_cls(
        embedder=embedder,
        vector_store=vector_store,
        collection=collection,
        reranker=reranker,
        params=params or {},
    )


def list_retrieval_strategies() -> list[str]:
    """列出已注册的检索策略。"""
    return list(_RETRIEVAL_ENGINE_REGISTRY.keys())


# ====================================================================
#  向后兼容别名
# ====================================================================


class DefaultRetrievalEngine:
    """向后兼容：默认检索引擎（= DenseRetrievalEngine 的旧接口）。

    保留此类以兼容已有的 `build_default_retrieval_engine` 工厂函数。
    新代码请直接使用 create_retrieval_engine + RetrievalStrategy.DENSE。
    """

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
        query_vector = await self._embedder.embed_query(request.query)
        candidate_k = (
            request.top_k * 4
            if request.use_rerank and self._reranker
            else request.top_k
        )
        candidates = await self._vector_store.search(
            collection=request.kb_id,
            query_vector=query_vector,
            top_k=candidate_k,
        )

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
            strategy=RetrievalStrategy.DENSE,
            results=results,
        )


def build_default_retrieval_engine(
    embedder_config: EmbeddingConfig,
    vector_store_config: VectorStoreConfig,
    reranker_config: RerankerConfig | None = None,
) -> DefaultRetrievalEngine:
    """构建默认检索引擎（向后兼容）。"""
    embedder = create_embedder(embedder_config)
    vector_store = create_vector_store(vector_store_config)
    reranker = create_reranker(reranker_config) if reranker_config else None
    return DefaultRetrievalEngine(embedder, vector_store, reranker)
