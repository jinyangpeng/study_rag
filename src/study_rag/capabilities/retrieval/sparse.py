"""Sparse 检索引擎 — BM25 关键词检索。

流程：分词 -> 倒排索引 -> BM25 评分 -> 可选 rerank
适用场景：精确关键词匹配、专有名词检索、代码/API 名称搜索。

BM25 公式：
  score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections import Counter
from typing import Any

from ...observability.logging import get_logger
from ..embedding.base import Embedder
from ..reranker.base import Reranker
from ..vector_store.base import SearchResult, VectorStore
from .base import (
    RetrievalEngine,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalStrategy,
    SparseParams,
    register_retrieval_engine,
)

logger = get_logger(__name__)


# ====================================================================
#  分词器
# ====================================================================


def _tokenize(
    text: str,
    use_jieba: bool = True,
    stop_words: list[str] | None = None,
) -> list[str]:
    """对文本进行分词。

    - 有 jieba 时用 jieba 精确模式
    - 无 jieba 时用简单的正则分词（按非中英文数字切分 + unigram 中文字符）
    - stop_words 中的词会被过滤掉（大小写不敏感）
    """
    if not text:
        return []

    if use_jieba:
        try:
            import jieba

            tokens = [w.strip() for w in jieba.lcut(text) if w.strip()]
        except ImportError:
            logger.debug("jieba not available, falling back to regex tokenizer")
            tokens = _regex_tokenize(text)
    else:
        tokens = _regex_tokenize(text)

    if stop_words:
        stop_lower = {sw.lower() for sw in stop_words}
        tokens = [t for t in tokens if t.lower() not in stop_lower]
    return tokens


def _regex_tokenize(text: str) -> list[str]:
    """简单正则分词：英文/数字连续串 + 中文字符逐字 unigram。"""
    tokens: list[str] = []
    for m in re.finditer(r"[a-zA-Z0-9_]+", text):
        tokens.append(m.group().lower())
    for ch in re.findall(r"[\u4e00-\u9fff]", text):
        tokens.append(ch)
    return tokens


# ====================================================================
#  BM25 核心
# ====================================================================


class _BM25Index:
    """轻量 BM25 内存索引。

    不依赖外部库（rank_bm25 / Elasticsearch），纯 Python 实现。
    适合中小规模文档集（< 10 万条）；大规模场景建议换 Elasticsearch。
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        use_jieba: bool = True,
        stop_words: list[str] | None = None,
    ):
        self._k1 = k1
        self._b = b
        self._use_jieba = use_jieba
        self._stop_words = stop_words or []
        # doc_id -> token list
        self._doc_tokens: dict[str, list[str]] = {}
        # token -> set of doc_ids（倒排索引）
        self._inverted_index: dict[str, set[str]] = {}
        # doc_id -> 文档原始信息（用于返回）
        self._doc_store: dict[str, dict[str, Any]] = {}
        self._avgdl: float = 0.0
        self._total_docs: int = 0
        self._idf_cache: dict[str, float] = {}

    def add_doc(self, doc_id: str, text: str, metadata: dict[str, Any]) -> None:
        """添加文档到索引。"""
        tokens = _tokenize(
            text, use_jieba=self._use_jieba, stop_words=self._stop_words
        )
        self._doc_tokens[doc_id] = tokens
        self._doc_store[doc_id] = {"text": text, "metadata": metadata, "id": doc_id}

        # 更新倒排索引
        unique_tokens = set(tokens)
        for t in unique_tokens:
            self._inverted_index.setdefault(t, set()).add(doc_id)

        # 更新统计量
        self._total_docs += 1
        total_len = sum(len(v) for v in self._doc_tokens.values())
        self._avgdl = total_len / self._total_docs if self._total_docs > 0 else 1.0
        # IDF 缓存失效
        self._idf_cache.clear()

    def _idf(self, token: str) -> float:
        """计算 IDF（逆文档频率）。"""
        if token in self._idf_cache:
            return self._idf_cache[token]
        df = len(self._inverted_index.get(token, set()))
        # BM25 IDF 公式：log((N - df + 0.5) / (df + 0.5) + 1)
        idf = math.log((self._total_docs - df + 0.5) / (df + 0.5) + 1.0)
        self._idf_cache[token] = idf
        return idf

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """BM25 检索，返回 (doc_id, score) 列表。"""
        if not self._doc_tokens:
            return []

        query_tokens = _tokenize(
            query, use_jieba=self._use_jieba, stop_words=self._stop_words
        )
        if not query_tokens:
            return []

        # 收集候选文档（至少包含一个查询词）
        candidate_ids: set[str] = set()
        for t in query_tokens:
            candidate_ids.update(self._inverted_index.get(t, set()))

        if not candidate_ids:
            return []

        # 计算 BM25 分数
        scores: list[tuple[str, float]] = []
        for doc_id in candidate_ids:
            tokens = self._doc_tokens[doc_id]
            doc_len = len(tokens)
            token_counts = Counter(tokens)
            score = 0.0
            for qt in query_tokens:
                if qt not in token_counts:
                    continue
                tf = token_counts[qt]
                idf = self._idf(qt)
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (
                    1.0 - self._b + self._b * doc_len / self._avgdl
                )
                score += idf * numerator / denominator
            if score > 0:
                scores.append((doc_id, score))

        # 按分数降序排列
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_doc(self, doc_id: str) -> dict[str, Any] | None:
        return self._doc_store.get(doc_id)

    @property
    def doc_count(self) -> int:
        return self._total_docs


# ====================================================================
#  Sparse 检索引擎
# ====================================================================


@register_retrieval_engine(RetrievalStrategy.SPARSE)
class SparseRetrievalEngine(RetrievalEngine):
    """Sparse 检索引擎：基于 BM25 的关键词检索。

    工作流程:
      1. 从向量库加载所有文档，构建 BM25 倒排索引（懒加载，首次检索时构建）
      2. 对 query 分词后在倒排索引中检索
      3. 可选 rerank 重排后截断到 top_k

    适用:
      - 精确关键词匹配（如 API 名、错误码、产品型号）
      - 代码/配置类文档检索
      - 作为 Hybrid 策略的 Sparse 组成部分

    局限:
      - 语义理解弱（"如何优化性能" 无法匹配 "性能调优"）
      - 首次检索需加载文档构建索引（冷启动）
      - 内存索引，不适合超大规模文档集
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
        self._embedder = embedder  # 保留用于 rerank
        self._vector_store = vector_store
        self._collection = collection
        self._reranker = reranker
        self._params = SparseParams(**(params or {}))
        self._index: _BM25Index | None = None
        # 保护 BM25 索引构建的并发锁（double-checked locking）
        self._index_lock = asyncio.Lock()

    @property
    def strategy(self) -> RetrievalStrategy:
        return RetrievalStrategy.SPARSE

    async def _ensure_index(self) -> _BM25Index:
        """懒加载：首次检索时从向量库构建 BM25 索引。

        使用 double-checked locking 避免并发首查时重复构建：
          - 第一次检查无锁，命中则直接返回（快路径）
          - 未命中时获取锁，再次检查（防止等待期间已被其他协程构建）
          - 仍为 None 才真正构建
        """
        # Fast path：已构建则直接返回（无锁）
        if self._index is not None:
            return self._index

        async with self._index_lock:
            # Double-check：等待锁期间可能已被其他协程构建
            if self._index is not None:
                return self._index

            logger.info("building_bm25_index", collection=self._collection)
            start = time.perf_counter()

            idx = _BM25Index(
                k1=self._params.k1,
                b=self._params.b,
                use_jieba=self._params.use_jieba,
                stop_words=self._params.stop_words,
            )

            # 从向量库拉取所有文档（分批，每批 500）
            offset = 0
            batch_size = 500
            total = 0
            while True:
                records = await self._vector_store.query(
                    self._collection,
                    limit=batch_size,
                    offset=offset,
                )
                if not records:
                    break
                for rec in records:
                    idx.add_doc(
                        doc_id=rec.id,
                        text=rec.text,
                        metadata=rec.metadata,
                    )
                total += len(records)
                offset += batch_size

            elapsed = round((time.perf_counter() - start) * 1000, 1)
            logger.info(
                "bm25_index_built",
                collection=self._collection,
                doc_count=total,
                duration_ms=elapsed,
            )

            self._index = idx
            return idx

    async def invalidate_index(self) -> None:
        """使 BM25 索引失效（下次检索时重建）。

        文档增删后应调用此方法，避免索引与向量库数据不一致。
        Manager 在 add_document / delete_document 后会批量清理检索引擎缓存，
        但如果直接操作向量库（绕过 Manager），需手动调用此方法。

        注意：此方法是 async 的，确保与 _ensure_index 的锁互斥，
        避免索引正在被构建时被清空。
        """
        async with self._index_lock:
            if self._index is not None:
                logger.info("bm25_index_invalidated", collection=self._collection)
                self._index = None

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        start = time.perf_counter()

        # 1. 确保 BM25 索引已构建
        index = await self._ensure_index()

        # 2. BM25 检索
        raw_hits = index.search(request.query, top_k=request.top_k * 4)
        candidates: list[SearchResult] = []
        for doc_id, score in raw_hits:
            doc = index.get_doc(doc_id)
            if doc is None:
                continue
            candidates.append(
                SearchResult(
                    id=doc_id,
                    text=doc["text"],
                    score=score,
                    metadata=doc["metadata"],
                )
            )

        # 3. Rerank（BM25 分数量纲与 reranker 不同，rerank 可进一步提升）
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
            strategy=RetrievalStrategy.SPARSE,
            results=results,
            meta={
                "duration_ms": elapsed_ms,
                "index_size": index.doc_count,
                "candidates_fetched": len(candidates),
                "reranked": request.use_rerank and self._reranker is not None,
            },
        )
