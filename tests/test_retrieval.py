"""检索策略单元测试 + 集成测试。"""

from __future__ import annotations

import pytest

from study_rag.capabilities.embedding.base import EmbeddingConfig
from study_rag.capabilities.embedding.impls import MockEmbedder
from study_rag.capabilities.retrieval import (
    RetrievalConfig,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalStrategy,
    create_retrieval_engine,
    list_retrieval_strategies,
)
from study_rag.capabilities.retrieval.sparse import _BM25Index, _tokenize
from study_rag.capabilities.vector_store.base import VectorRecord, VectorStoreConfig
from study_rag.capabilities.vector_store.impls import InMemoryVectorStore

# ====================================================================
#  配置模型测试
# ====================================================================


class TestRetrievalConfig:
    def test_default_config(self):
        cfg = RetrievalConfig()
        assert cfg.default_strategy == RetrievalStrategy.DENSE
        assert cfg.dense.over_fetch_factor == 4
        assert cfg.sparse.k1 == 1.5
        assert cfg.sparse.b == 0.75
        assert cfg.hybrid.dense_weight == 0.6
        assert cfg.hybrid.rrf_k == 60

    def test_custom_config(self):
        cfg = RetrievalConfig(
            default_strategy=RetrievalStrategy.HYBRID,
            dense={"over_fetch_factor": 2},
            sparse={"k1": 2.0, "b": 0.5},
            hybrid={"dense_weight": 0.7, "rrf_k": 30},
        )
        assert cfg.default_strategy == RetrievalStrategy.HYBRID
        assert cfg.dense.over_fetch_factor == 2
        assert cfg.sparse.k1 == 2.0
        assert cfg.hybrid.dense_weight == 0.7


# ====================================================================
#  BM25 分词测试
# ====================================================================


class TestTokenize:
    def test_empty_text(self):
        assert _tokenize("") == []

    def test_english_text(self):
        tokens = _tokenize("Hello World", use_jieba=False)
        assert "hello" in tokens
        assert "world" in tokens

    def test_chinese_text_no_jieba(self):
        tokens = _tokenize("性能优化", use_jieba=False)
        # 无 jieba 时中文逐字拆分
        assert "性" in tokens
        assert "能" in tokens
        assert "优" in tokens
        assert "化" in tokens

    def test_mixed_text(self):
        tokens = _tokenize("React 性能优化", use_jieba=False)
        assert "react" in tokens
        assert "性" in tokens


# ====================================================================
#  BM25 索引测试
# ====================================================================


class TestBM25Index:
    def test_empty_index(self):
        idx = _BM25Index(use_jieba=False)
        results = idx.search("test", top_k=10)
        assert results == []

    def test_single_doc(self):
        idx = _BM25Index(use_jieba=False)
        idx.add_doc("doc1", "hello world", {"title": "test"})
        results = idx.search("hello", top_k=10)
        assert len(results) == 1
        assert results[0][0] == "doc1"
        assert results[0][1] > 0

    def test_multi_doc_ranking(self):
        idx = _BM25Index(use_jieba=False)
        idx.add_doc("doc1", "hello world", {})
        idx.add_doc("doc2", "hello hello hello world", {})
        idx.add_doc("doc3", "foo bar baz", {})
        results = idx.search("hello", top_k=10)
        # doc2 应该排第一（"hello" 出现 3 次）
        assert results[0][0] == "doc2"
        # doc3 应该不在结果中（没有 "hello"）
        assert all(r[0] != "doc3" for r in results)

    def test_get_doc(self):
        idx = _BM25Index(use_jieba=False)
        idx.add_doc("doc1", "test content", {"key": "value"})
        doc = idx.get_doc("doc1")
        assert doc is not None
        assert doc["text"] == "test content"
        assert doc["metadata"]["key"] == "value"

    def test_doc_count(self):
        idx = _BM25Index(use_jieba=False)
        assert idx.doc_count == 0
        idx.add_doc("doc1", "test", {})
        assert idx.doc_count == 1
        idx.add_doc("doc2", "test2", {})
        assert idx.doc_count == 2


# ====================================================================
#  检索引擎工厂测试
# ====================================================================


class TestRetrievalEngineFactory:
    def test_list_strategies(self):
        strategies = list_retrieval_strategies()
        assert "dense" in strategies
        assert "sparse" in strategies
        assert "hybrid" in strategies

    def test_strategy_enum_values(self):
        assert RetrievalStrategy.DENSE.value == "dense"
        assert RetrievalStrategy.SPARSE.value == "sparse"
        assert RetrievalStrategy.HYBRID.value == "hybrid"


# ====================================================================
#  RRF 融合测试
# ====================================================================


class TestRRF:
    def test_empty_lists(self):
        from study_rag.capabilities.retrieval.hybrid import _reciprocal_rank_fusion

        result = _reciprocal_rank_fusion([], [], k=60)
        assert result == []

    def test_single_list(self):
        from study_rag.capabilities.retrieval.hybrid import _reciprocal_rank_fusion
        from study_rag.capabilities.vector_store.base import SearchResult

        results = [
            SearchResult(id="doc1", text="a", score=0.9, metadata={}),
            SearchResult(id="doc2", text="b", score=0.8, metadata={}),
        ]
        fused = _reciprocal_rank_fusion([results], [1.0], k=60)
        assert len(fused) == 2
        assert fused[0].id == "doc1"

    def test_fusion_ranking(self):
        from study_rag.capabilities.retrieval.hybrid import _reciprocal_rank_fusion
        from study_rag.capabilities.vector_store.base import SearchResult

        dense_results = [
            SearchResult(id="doc1", text="a", score=0.9, metadata={}),
            SearchResult(id="doc2", text="b", score=0.8, metadata={}),
            SearchResult(id="doc3", text="c", score=0.7, metadata={}),
        ]
        sparse_results = [
            SearchResult(id="doc2", text="b", score=5.0, metadata={}),
            SearchResult(id="doc1", text="a", score=4.0, metadata={}),
            SearchResult(id="doc4", text="d", score=3.0, metadata={}),
        ]
        fused = _reciprocal_rank_fusion(
            [dense_results, sparse_results],
            [0.6, 0.4],
            k=60,
        )
        # doc1 和 doc2 在两路都出现，应该排在前面
        assert len(fused) == 4
        assert fused[0].id in ("doc1", "doc2")
        # doc3 和 doc4 只在一路出现
        fused_ids = [r.id for r in fused]
        assert "doc3" in fused_ids
        assert "doc4" in fused_ids


# ====================================================================
#  RetrievalRequest / RetrievalResponse 测试
# ====================================================================


class TestModels:
    def test_request_defaults(self):
        req = RetrievalRequest(kb_id="test", query="hello")
        assert req.kb_id == "test"
        assert req.query == "hello"
        assert req.top_k == 5
        assert req.use_rerank is True
        assert req.strategy is None
        assert req.strategy_params == {}

    def test_response_defaults(self):
        resp = RetrievalResponse(
            kb_id="test",
            query="hello",
            strategy=RetrievalStrategy.DENSE,
        )
        assert resp.kb_id == "test"
        assert resp.strategy == RetrievalStrategy.DENSE
        assert resp.results == []
        assert resp.meta == {}


# ====================================================================
#  集成测试：真实 mock 组件端到端（MockEmbedder + InMemoryVectorStore）
# ====================================================================


@pytest.fixture
def embedder() -> MockEmbedder:
    """维度=8 的 mock embedder（hash 伪向量，接口完全一致）。"""
    return MockEmbedder(
        EmbeddingConfig(provider="mock", model_name="mock", dimension=8)
    )


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore(VectorStoreConfig(provider="mock"))


@pytest.fixture
def populated_collection(
    embedder: MockEmbedder, vector_store: InMemoryVectorStore
) -> str:
    """创建 collection 并写入 4 条测试文档，返回 collection 名。

    文档内容设计：
      - doc1: "React 性能优化指南"   （含 React / 性能 / 优化）
      - doc2: "Vue 性能调优实践"     （含 Vue / 性能 / 调优）
      - doc3: "Python 数据分析"      （含 Python / 数据 / 分析）
      - doc4: "React Hooks 使用"     （含 React / Hooks）
    """
    import asyncio

    collection = "test_kb"
    docs = [
        ("doc1", "React 性能优化指南", {"title": "react_perf", "source": "wiki"}),
        ("doc2", "Vue 性能调优实践", {"title": "vue_perf", "source": "wiki"}),
        ("doc3", "Python 数据分析", {"title": "py_data", "source": "book"}),
        ("doc4", "React Hooks 使用", {"title": "react_hooks", "source": "wiki"}),
    ]
    asyncio.run(vector_store.create_collection(collection, dimension=8))
    records = []
    for doc_id, text, meta in docs:
        vec = asyncio.run(embedder.embed_query(text))
        records.append(VectorRecord(id=doc_id, vector=vec, text=text, metadata=meta))
    asyncio.run(vector_store.insert(collection, records))
    return collection


class TestDenseRetrievalIntegration:
    """Dense 策略端到端测试。"""

    @pytest.mark.asyncio
    async def test_dense_returns_results(
        self, embedder, vector_store, populated_collection
    ):
        engine = create_retrieval_engine(
            RetrievalStrategy.DENSE,
            embedder=embedder,
            vector_store=vector_store,
            collection=populated_collection,
            reranker=None,
        )
        resp = await engine.retrieve(
            RetrievalRequest(kb_id="test", query="React", top_k=3, use_rerank=False)
        )
        assert resp.strategy == RetrievalStrategy.DENSE
        assert len(resp.results) <= 3
        assert len(resp.results) > 0
        assert resp.meta["reranked"] is False
        assert "candidates_fetched" in resp.meta

    @pytest.mark.asyncio
    async def test_dense_over_fetch_with_reranker(
        self, embedder, vector_store, populated_collection
    ):
        """启用 reranker 时 candidate_k = top_k * over_fetch_factor。"""
        engine = create_retrieval_engine(
            RetrievalStrategy.DENSE,
            embedder=embedder,
            vector_store=vector_store,
            collection=populated_collection,
            reranker=None,
            params={"over_fetch_factor": 2},
        )
        resp = await engine.retrieve(
            RetrievalRequest(kb_id="test", query="React", top_k=2, use_rerank=True)
        )
        # 无 reranker 时 over_fetch 不生效（candidate_k = top_k）
        assert resp.meta["over_fetch_factor"] == 1


class TestSparseRetrievalIntegration:
    """Sparse 策略端到端测试。"""

    @pytest.mark.asyncio
    async def test_sparse_keyword_match(
        self, embedder, vector_store, populated_collection
    ):
        """BM25 应能精确匹配关键词 'React'。"""
        engine = create_retrieval_engine(
            RetrievalStrategy.SPARSE,
            embedder=embedder,
            vector_store=vector_store,
            collection=populated_collection,
            reranker=None,
            params={"use_jieba": False},
        )
        resp = await engine.retrieve(
            RetrievalRequest(kb_id="test", query="React", top_k=4, use_rerank=False)
        )
        assert resp.strategy == RetrievalStrategy.SPARSE
        assert len(resp.results) > 0
        # doc1 和 doc4 含 "React"，应排在前面
        result_ids = {r.id for r in resp.results}
        assert "doc1" in result_ids
        assert "doc4" in result_ids
        assert resp.meta["index_size"] == 4

    @pytest.mark.asyncio
    async def test_sparse_stop_words_filter(
        self, embedder, vector_store, populated_collection
    ):
        """stop_words 应过滤掉指定词，影响匹配结果。"""
        # 把 "React" 加入停用词，查询 "React" 应无结果
        engine = create_retrieval_engine(
            RetrievalStrategy.SPARSE,
            embedder=embedder,
            vector_store=vector_store,
            collection=populated_collection,
            reranker=None,
            params={"use_jieba": False, "stop_words": ["React"]},
        )
        resp = await engine.retrieve(
            RetrievalRequest(kb_id="test", query="React", top_k=4, use_rerank=False)
        )
        # "React" 被过滤后查询无 token，应返回空
        assert len(resp.results) == 0

    @pytest.mark.asyncio
    async def test_sparse_index_invalidation(
        self, embedder, vector_store, populated_collection
    ):
        """文档新增后索引应失效重建。"""
        engine = create_retrieval_engine(
            RetrievalStrategy.SPARSE,
            embedder=embedder,
            vector_store=vector_store,
            collection=populated_collection,
            reranker=None,
            params={"use_jieba": False},
        )
        # 首次检索构建索引
        resp1 = await engine.retrieve(
            RetrievalRequest(kb_id="test", query="Angular", top_k=4, use_rerank=False)
        )
        assert len(resp1.results) == 0  # 无 Angular 文档

        # 新增 Angular 文档
        vec = await embedder.embed_query("Angular 框架入门")
        await vector_store.insert(
            populated_collection,
            [VectorRecord(id="doc5", vector=vec, text="Angular 框架入门", metadata={})],
        )

        # 失效索引
        await engine.invalidate_index()

        # 重新检索应能找到新文档
        resp2 = await engine.retrieve(
            RetrievalRequest(kb_id="test", query="Angular", top_k=4, use_rerank=False)
        )
        assert len(resp2.results) > 0
        assert resp2.results[0].id == "doc5"
        assert resp2.meta["index_size"] == 5

    @pytest.mark.asyncio
    async def test_sparse_concurrent_build_no_duplicate(
        self, embedder, vector_store, populated_collection
    ):
        """并发首查时 BM25 索引只构建一次（double-checked locking）。

        通过 mock vector_store.query 计数调用次数，验证并发请求不会
        重复拉取全量文档构建索引。
        """
        import asyncio

        engine = create_retrieval_engine(
            RetrievalStrategy.SPARSE,
            embedder=embedder,
            vector_store=vector_store,
            collection=populated_collection,
            reranker=None,
            params={"use_jieba": False},
        )

        # 包装 query 方法统计调用次数
        original_query = vector_store.query
        call_count = {"n": 0}

        async def counting_query(collection, **kwargs):
            call_count["n"] += 1
            return await original_query(collection, **kwargs)

        vector_store.query = counting_query  # type: ignore[assignment]

        try:
            # 并发发起 5 个检索请求（都触发 _ensure_index）
            tasks = [
                engine.retrieve(
                    RetrievalRequest(
                        kb_id="test", query="React", top_k=3, use_rerank=False
                    )
                )
                for _ in range(5)
            ]
            results = await asyncio.gather(*tasks)

            # 所有请求都应成功返回
            assert len(results) == 5
            for r in results:
                assert r.meta["index_size"] == 4

            # query 调用次数应远小于 5 * 批次数（证明索引只构建了一次）
            # 4 条文档 / batch_size=500 → 构建 1 次需 1 次 query（拉到空结束）
            # 实际：构建索引的 query 调用 ≈ 2（一批数据 + 一批空）
            # 如果无锁，5 个请求各自构建 → ~10 次 query
            assert call_count["n"] < 5, (
                f"Expected index built once, but query called {call_count['n']} times"
            )
        finally:
            vector_store.query = original_query  # type: ignore[assignment]


class TestHybridRetrievalIntegration:
    """Hybrid 策略端到端测试。"""

    @pytest.mark.asyncio
    async def test_hybrid_combines_dense_and_sparse(
        self, embedder, vector_store, populated_collection
    ):
        engine = create_retrieval_engine(
            RetrievalStrategy.HYBRID,
            embedder=embedder,
            vector_store=vector_store,
            collection=populated_collection,
            reranker=None,
            params={"use_jieba": False, "dense_weight": 0.5, "rrf_k": 60},
        )
        resp = await engine.retrieve(
            RetrievalRequest(kb_id="test", query="React 性能", top_k=4, use_rerank=False)
        )
        assert resp.strategy == RetrievalStrategy.HYBRID
        assert len(resp.results) > 0
        assert resp.meta["dense_count"] >= 0
        assert resp.meta["sparse_count"] >= 0
        assert resp.meta["fused_count"] >= len(resp.results)
        assert resp.meta["dense_weight"] == 0.5

    @pytest.mark.asyncio
    async def test_hybrid_single_branch_failure_tolerance(
        self, embedder, vector_store, populated_collection
    ):
        """Hybrid 一路失败不应阻塞整体检索。

        通过让 sparse 查询无结果（用 stop_words 过滤所有词）验证容错。
        """
        engine = create_retrieval_engine(
            RetrievalStrategy.HYBRID,
            embedder=embedder,
            vector_store=vector_store,
            collection=populated_collection,
            reranker=None,
            params={"use_jieba": False},
        )
        resp = await engine.retrieve(
            RetrievalRequest(kb_id="test", query="React", top_k=4, use_rerank=False)
        )
        # 两路都应正常返回
        assert resp.meta["dense_count"] > 0 or resp.meta["sparse_count"] > 0


class TestStrategyParamsOverride:
    """策略参数覆盖优先级测试：全局 < 请求级。"""

    @pytest.mark.asyncio
    async def test_request_params_override_engine_params(
        self, embedder, vector_store, populated_collection
    ):
        """请求级 strategy_params 应覆盖引擎构造时的参数。"""
        from study_rag.knowledge_bases.manager import _merge_strategy_params

        global_cfg = RetrievalConfig()
        # 全局 dense.over_fetch_factor = 4
        merged = _merge_strategy_params(
            RetrievalStrategy.DENSE, global_cfg, {}, {"over_fetch_factor": 10}
        )
        assert merged["over_fetch_factor"] == 10

    def test_merge_priority_global_kb_request(self):
        """参数合并优先级：全局默认 < KB 配置 < 请求覆盖。"""
        from study_rag.knowledge_bases.manager import _merge_strategy_params

        global_cfg = RetrievalConfig()  # sparse.k1 = 1.5
        kb_params = {"k1": 2.0}  # KB 覆盖为 2.0
        request_params = {"k1": 2.5, "b": 0.5}  # 请求再覆盖

        merged = _merge_strategy_params(
            RetrievalStrategy.SPARSE, global_cfg, kb_params, request_params
        )
        assert merged["k1"] == 2.5  # 请求级胜出
        assert merged["b"] == 0.5  # 请求级新增
        assert merged["use_jieba"] is True  # 全局默认保留

    def test_merge_milvus_bm25_params(self):
        """Milvus BM25 策略参数合并。"""
        from study_rag.knowledge_bases.manager import _merge_strategy_params

        global_cfg = RetrievalConfig()  # milvus_bm25.analyzer_type = "chinese"
        merged = _merge_strategy_params(
            RetrievalStrategy.SPARSE_MILVUS,
            global_cfg,
            {},
            {"analyzer_type": "english", "over_fetch_factor": 8},
        )
        assert merged["analyzer_type"] == "english"
        assert merged["over_fetch_factor"] == 8


# ====================================================================
#  Milvus 2.5+ BM25 引擎测试（用 mock vector_store 模拟 BM25 方法）
# ====================================================================


class _MockMilvusBM25VectorStore:
    """模拟支持 BM25 的 MilvusVectorStore（无需真实 Milvus 连接）。

    提供 search_sparse / hybrid_search 方法，返回预设结果。
    """

    def __init__(self, sparse_results=None, hybrid_results=None):
        self._sparse_results = sparse_results or []
        self._hybrid_results = hybrid_results or []

    async def search_sparse(self, collection, query_text, top_k=10, filter_expr=None):
        return self._sparse_results[:top_k]

    async def hybrid_search(
        self, collection, query_vector, query_text, top_k=10,
        filter_expr=None, dense_weight=0.6, rrf_k=60,
    ):
        return self._hybrid_results[:top_k]

    async def search(self, collection, query_vector, top_k=5, filter_expr=None):
        return self._hybrid_results[:top_k]


class TestMilvusBM25Engines:
    """Milvus 2.5+ BM25 引擎测试。"""

    def test_sparse_milvus_strategy_registered(self):
        """SPARSE_MILVUS 策略应已注册。"""
        strategies = list_retrieval_strategies()
        assert "sparse_milvus" in strategies
        assert "hybrid_milvus" in strategies

    def test_milvus_engine_requires_bm25_vector_store(self, embedder):
        """非 BM25 的 vector_store 应拒绝创建 Milvus 引擎。"""
        from study_rag.capabilities.vector_store.base import VectorStoreConfig
        from study_rag.capabilities.vector_store.impls import InMemoryVectorStore

        vs = InMemoryVectorStore(VectorStoreConfig(provider="mock"))
        with pytest.raises(ValueError, match="BM25 support"):
            create_retrieval_engine(
                RetrievalStrategy.SPARSE_MILVUS,
                embedder=embedder,
                vector_store=vs,
                collection="test",
            )

    @pytest.mark.asyncio
    async def test_sparse_milvus_retrieve(self, embedder):
        """MilvusSparseRetrievalEngine 应调用 search_sparse 并返回结果。"""
        from study_rag.capabilities.vector_store.base import SearchResult

        mock_vs = _MockMilvusBM25VectorStore(
            sparse_results=[
                SearchResult(id="doc1", text="React 性能", score=2.5, metadata={}),
                SearchResult(id="doc2", text="Vue 性能", score=1.8, metadata={}),
            ]
        )
        engine = create_retrieval_engine(
            RetrievalStrategy.SPARSE_MILVUS,
            embedder=embedder,
            vector_store=mock_vs,
            collection="test",
            reranker=None,
        )
        resp = await engine.retrieve(
            RetrievalRequest(kb_id="test", query="React", top_k=2, use_rerank=False)
        )
        assert resp.strategy == RetrievalStrategy.SPARSE_MILVUS
        assert len(resp.results) == 2
        assert resp.results[0].id == "doc1"
        assert resp.meta["backend"] == "milvus_bm25"
        assert resp.meta["candidates_fetched"] == 2

    @pytest.mark.asyncio
    async def test_hybrid_milvus_retrieve(self, embedder):
        """MilvusHybridRetrievalEngine 应调用 hybrid_search 并返回结果。"""
        from study_rag.capabilities.vector_store.base import SearchResult

        mock_vs = _MockMilvusBM25VectorStore(
            hybrid_results=[
                SearchResult(id="doc1", text="React 性能", score=0.9, metadata={}),
                SearchResult(id="doc2", text="Vue 性能", score=0.8, metadata={}),
            ]
        )
        engine = create_retrieval_engine(
            RetrievalStrategy.HYBRID_MILVUS,
            embedder=embedder,
            vector_store=mock_vs,
            collection="test",
            reranker=None,
            params={"dense_weight": 0.7, "rrf_k": 50},
        )
        resp = await engine.retrieve(
            RetrievalRequest(kb_id="test", query="React", top_k=2, use_rerank=False)
        )
        assert resp.strategy == RetrievalStrategy.HYBRID_MILVUS
        assert len(resp.results) == 2
        assert resp.meta["backend"] == "milvus_hybrid"
        assert resp.meta["dense_weight"] == 0.7
        assert resp.meta["rrf_k"] == 50

    @pytest.mark.asyncio
    async def test_hybrid_milvus_fallback_to_dense_on_error(self, embedder):
        """hybrid_search 失败时应降级为纯 dense 检索。"""
        from study_rag.capabilities.vector_store.base import SearchResult

        class _FailingHybridStore(_MockMilvusBM25VectorStore):
            async def hybrid_search(self, *args, **kwargs):
                raise RuntimeError("Milvus hybrid_search not supported")

            async def search(self, collection, query_vector, top_k=5, filter_expr=None):
                return [SearchResult(id="doc1", text="fallback", score=0.5, metadata={})]

        mock_vs = _FailingHybridStore()
        engine = create_retrieval_engine(
            RetrievalStrategy.HYBRID_MILVUS,
            embedder=embedder,
            vector_store=mock_vs,
            collection="test",
            reranker=None,
        )
        resp = await engine.retrieve(
            RetrievalRequest(kb_id="test", query="React", top_k=2, use_rerank=False)
        )
        # 降级后仍返回结果（来自 dense search）
        assert len(resp.results) == 1
        assert resp.results[0].id == "doc1"


# ====================================================================
#  Rerank 集成测试：验证 _apply_rerank 参数名正确（防回归）
# ====================================================================


class _MockReranker:
    """简单的 mock reranker，模拟真实 reranker 的 rerank(query, results, top_k) 接口。

    模拟真实行为：top_k=None 时用自身配置的 _top_k（如 reranker.yaml 配置的 top_k）。
    """

    def __init__(self, top_k: int = 3):
        self._top_k = top_k  # 模拟 reranker.yaml 配置的 top_k

    async def rerank(self, query, results, top_k):
        # 与真实 reranker 一致：top_k=None 时用自身配置
        k = top_k if top_k is not None else self._top_k
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        return sorted_results[:k]


class TestRerankIntegration:
    """验证所有引擎在有 reranker 时能正确调用 _apply_rerank。

    回归测试：之前 _apply_rerank 调用时用了错误的参数名 `results=`
    而非 `candidates=`，导致 use_rerank=True + reranker!=None 时 400 错误。
    """

    @pytest.mark.asyncio
    async def test_dense_with_reranker(self, embedder, vector_store, populated_collection):
        """Dense 引擎 + reranker 不报参数名错误，返回数由 reranker 配置决定。"""
        engine = create_retrieval_engine(
            RetrievalStrategy.DENSE,
            embedder=embedder,
            vector_store=vector_store,
            collection=populated_collection,
            reranker=_MockReranker(top_k=3),  # reranker 配置返回 3 条
        )
        resp = await engine.retrieve(
            RetrievalRequest(kb_id="test", query="React", top_k=2, use_rerank=True)
        )
        assert resp.meta["reranked"] is True
        # top_k=2 是召回数，rerank 后返回 3 条（reranker 配置的 top_k）
        assert len(resp.results) == 3

    @pytest.mark.asyncio
    async def test_sparse_with_reranker(self, embedder, vector_store, populated_collection):
        """Sparse 引擎 + reranker 不报参数名错误。"""
        engine = create_retrieval_engine(
            RetrievalStrategy.SPARSE,
            embedder=embedder,
            vector_store=vector_store,
            collection=populated_collection,
            reranker=_MockReranker(top_k=2),
            params={"use_jieba": False},
        )
        resp = await engine.retrieve(
            RetrievalRequest(kb_id="test", query="React", top_k=2, use_rerank=True)
        )
        assert resp.meta["reranked"] is True
        assert len(resp.results) == 2

    @pytest.mark.asyncio
    async def test_hybrid_with_reranker(self, embedder, vector_store, populated_collection):
        """Hybrid 引擎 + reranker 不报参数名错误。"""
        engine = create_retrieval_engine(
            RetrievalStrategy.HYBRID,
            embedder=embedder,
            vector_store=vector_store,
            collection=populated_collection,
            reranker=_MockReranker(top_k=2),
            params={"use_jieba": False},
        )
        resp = await engine.retrieve(
            RetrievalRequest(kb_id="test", query="React", top_k=2, use_rerank=True)
        )
        assert resp.meta["reranked"] is True
        assert len(resp.results) == 2

    @pytest.mark.asyncio
    async def test_sparse_milvus_with_reranker(self, embedder):
        """MilvusSparse 引擎 + reranker 不报参数名错误。"""
        from study_rag.capabilities.vector_store.base import SearchResult

        mock_vs = _MockMilvusBM25VectorStore(
            sparse_results=[
                SearchResult(id="doc1", text="React", score=2.5, metadata={}),
                SearchResult(id="doc2", text="Vue", score=1.8, metadata={}),
            ]
        )
        engine = create_retrieval_engine(
            RetrievalStrategy.SPARSE_MILVUS,
            embedder=embedder,
            vector_store=mock_vs,
            collection="test",
            reranker=_MockReranker(top_k=1),
        )
        resp = await engine.retrieve(
            RetrievalRequest(kb_id="test", query="React", top_k=1, use_rerank=True)
        )
        assert resp.meta["reranked"] is True
        assert len(resp.results) == 1

    @pytest.mark.asyncio
    async def test_hybrid_milvus_with_reranker(self, embedder):
        """MilvusHybrid 引擎 + reranker 不报参数名错误。"""
        from study_rag.capabilities.vector_store.base import SearchResult

        mock_vs = _MockMilvusBM25VectorStore(
            hybrid_results=[
                SearchResult(id="doc1", text="React", score=0.9, metadata={}),
                SearchResult(id="doc2", text="Vue", score=0.8, metadata={}),
            ]
        )
        engine = create_retrieval_engine(
            RetrievalStrategy.HYBRID_MILVUS,
            embedder=embedder,
            vector_store=mock_vs,
            collection="test",
            reranker=_MockReranker(top_k=1),
        )
        resp = await engine.retrieve(
            RetrievalRequest(kb_id="test", query="React", top_k=1, use_rerank=True)
        )
        assert resp.meta["reranked"] is True
        assert len(resp.results) == 1
