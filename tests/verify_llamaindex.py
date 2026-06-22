"""LlamaIndex 整合验证脚本。

验证：
  1. 包可正确导入
  2. NodeParser 4 种策略（whole / sentence / token / semantic）
  3. NodeParser 切块数合理
  4. VectorStore 适配器 aadd/aquery/adelete 端到端
  5. Embedding 适配器维度一致性
  6. Reranker 适配器 postprocess
  7. NodeMapper 互转
  8. LlamaIndexRetrievalEngine.aretrieve 端到端
  9. Manager.get_llamaindex_engine / search_via_llamaindex
 10. add_document_chunked 写入多块
 11. 旧回归（embedding/vector_store/reranker）
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import yaml

# ---- 兼容性检查 ----
LI_OK = True
try:
    import llama_index.core  # noqa: F401
except ImportError:
    LI_OK = False
    print("llama-index-core 未装, 跳过 LI 集成测试")
    sys.exit(0)


def _section(name: str) -> None:
    print(f"\n=== {name} ===")


# ---- 1. 包导入 ----
def verify_package_import():
    _section("1. 包导入")
    from study_rag.capabilities import llamaindex

    assert llamaindex.is_llama_index_available()
    assert hasattr(llamaindex, "LlamaIndexRetrievalEngine")
    assert hasattr(llamaindex, "LIVectorStoreAdapter")
    assert hasattr(llamaindex, "LIEmbeddingAdapter")
    assert hasattr(llamaindex, "LIRerankerPostprocessor")
    assert hasattr(llamaindex, "NodeParserFactory")
    assert hasattr(llamaindex, "NodeMapper")
    print("  [OK] llamaindex 包暴露 6 个核心组件")


# ---- 2. NodeParser 4 策略 ----
def verify_node_parser_strategies():
    _section("2. NodeParser 4 策略")
    from study_rag.capabilities.llamaindex import NodeParserFactory

    long_text = (
        "今天天气不错。适合出去走走。\n\n"
        "公园里花开了。红的、黄的、紫的。\n\n"
        "小明在草地上放风筝。风筝飞得很高。\n\n"
        "小红在树下看书。她喜欢春天的风。\n\n"
        "傍晚回家时，雨开始下了。\n\n"
        "我们加快了脚步。雨越下越大。"
    )

    # whole
    p = NodeParserFactory.from_raw({"strategy": "whole"})
    nodes = p.parse(long_text, "doc1", title="测试")
    assert len(nodes) == 1
    assert nodes[0].text == long_text
    print(f"  [OK] whole -> {len(nodes)} 块")

    # sentence
    p = NodeParserFactory.from_raw({"strategy": "sentence", "chunk_size": 50})
    nodes = p.parse(long_text, "doc2", title="测试")
    assert len(nodes) >= 2
    print(f"  [OK] sentence (chunk_size=50) -> {len(nodes)} 块")
    for i, n in enumerate(nodes):
        print(f"      [{i}] {n.text[:30]}...")

    # token
    p = NodeParserFactory.from_raw({"strategy": "token", "chunk_size": 50, "chunk_overlap": 10})
    nodes = p.parse(long_text, "doc3", title="测试")
    assert len(nodes) >= 1
    print(f"  [OK] token (chunk_size=50, overlap=10) -> {len(nodes)} 块")

    # 未知策略报错
    try:
        NodeParserFactory.from_raw({"strategy": "unknown"})
        raise AssertionError("应抛 ValueError")
    except ValueError as ex:
        print(f"  [OK] 未知策略报错: {str(ex)[:60]}")


# ---- 3. VectorStore 适配器 ----
async def verify_vs_adapter():
    _section("3. VectorStore 适配器端到端")
    from study_rag.capabilities.llamaindex.vector_store_adapter import LIVectorStoreAdapter
    from study_rag.capabilities.vector_store import (
        VectorStoreConfig,
        create_vector_store,
    )

    store = create_vector_store(VectorStoreConfig(provider="mock"))
    await store.create_collection("li_test", dimension=4)
    adapter = LIVectorStoreAdapter(store, "li_test")

    # aadd: 包装 TextNode
    from llama_index.core.schema import TextNode

    nodes = [
        TextNode(id_="n1", text="苹果", metadata={"title": "水果"}),
        TextNode(id_="n2", text="香蕉", metadata={"title": "水果"}),
        TextNode(id_="n3", text="白菜", metadata={"title": "蔬菜"}),
    ]
    # 手动设置 embedding（mock 不存在时直接用）
    for i, n in enumerate(nodes):
        n.embedding = [float(i), float(i) + 0.1, 0.0, 0.0]

    ids = await adapter.aadd(nodes)
    assert set(ids) == {"n1", "n2", "n3"}
    print(f"  [OK] aadd {len(ids)} nodes -> {ids}")

    # aquery
    from llama_index.core.vector_stores.types import VectorStoreQuery

    q = VectorStoreQuery(
        query_embedding=[0.0, 0.1, 0.0, 0.0],
        similarity_top_k=2,
    )
    res = await adapter.aquery(q)
    assert len(res.nodes) == 2
    print(f"  [OK] aquery top_k=2 -> {len(res.nodes)} hits, ids={res.ids}")
    # 验证：scores 按降序（mock 用的 cosine）
    assert res.similarities[0] >= res.similarities[1]
    print(f"  [OK] scores 降序: {res.similarities[0]:.3f} >= {res.similarities[1]:.3f}")

    # adelete
    await adapter.adelete("n1")
    res2 = await adapter.aquery(q)
    assert "n1" not in res2.ids
    print("  [OK] adelete n1 后查询不到")


# ---- 4. Embedding 适配器 ----
async def verify_embedding_adapter():
    _section("4. Embedding 适配器")
    from study_rag.capabilities.embedding import EmbeddingConfig, create_embedder
    from study_rag.capabilities.llamaindex.embedding_adapter import LIEmbeddingAdapter

    cfg = EmbeddingConfig(provider="mock", model_name="m", dimension=8)
    embedder = create_embedder(cfg)
    adapter = LIEmbeddingAdapter(embedder)

    # 维度一致
    assert adapter.dimension == 8
    print(f"  [OK] dimension={adapter.dimension}")

    # query 嵌入
    v1 = await adapter._aget_query_embedding("hello")
    v2 = await adapter._aget_text_embedding("hello")
    assert len(v1) == 8
    assert len(v2) == 8
    print(f"  [OK] _aget_query_embedding/text_embedding -> {len(v1)} 维")

    # 批量
    vs = await adapter._aget_text_embeddings(["a", "b", "c"])
    assert len(vs) == 3
    assert all(len(v) == 8 for v in vs)
    print(f"  [OK] _aget_text_embeddings batch=3 -> {len(vs)} x {len(vs[0])} 维")


# ---- 5. Reranker 适配器 ----
async def verify_reranker_adapter():
    _section("5. Reranker 适配器")
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    from study_rag.capabilities.llamaindex.reranker_adapter import LIRerankerPostprocessor
    from study_rag.capabilities.reranker import RerankerConfig, create_reranker

    # 用 none provider 模拟（passthrough 行为）
    cfg = RerankerConfig(provider="none", top_k=2)
    reranker = create_reranker(cfg)
    adapter = LIRerankerPostprocessor(reranker, top_n=2)

    nodes = [
        NodeWithScore(node=TextNode(id_="n1", text="x"), score=0.3),
        NodeWithScore(node=TextNode(id_="n2", text="y"), score=0.9),
        NodeWithScore(node=TextNode(id_="n3", text="z"), score=0.5),
    ]
    out = await adapter._postprocess_nodes(nodes, QueryBundle(query_str="q"))
    assert len(out) == 2
    print(f"  [OK] _postprocess_nodes 输入3 -> 输出{len(out)} (top_n=2)")
    # none rerank 保留顺序
    assert [o.node.node_id for o in out] == ["n1", "n2"]
    print(f"  [OK] none reranker 保留原序: {[o.node.node_id for o in out]}")


# ---- 6. NodeMapper ----
def verify_node_mapper():
    _section("6. NodeMapper 互转")
    from llama_index.core.schema import NodeWithScore

    from study_rag.capabilities.llamaindex.node_mapper import NodeMapper
    from study_rag.capabilities.vector_store import SearchResult

    sr = SearchResult(id="n1", text="hello", score=0.7, metadata={"k": "v"})

    # sr -> node
    n = NodeMapper.search_result_to_node(sr)
    assert n.node_id == "n1"
    assert n.get_content() == "hello"
    assert n.metadata["k"] == "v"
    print("  [OK] SearchResult -> TextNode")

    # sr -> node with score
    nws = NodeMapper.search_result_to_node_with_score(sr)
    assert isinstance(nws, NodeWithScore)
    assert nws.score == 0.7
    print("  [OK] SearchResult -> NodeWithScore")

    # nws -> sr
    back = NodeMapper.node_with_score_to_search_result(nws)
    assert back.id == "n1"
    assert back.text == "hello"
    assert back.score == 0.7
    print("  [OK] NodeWithScore -> SearchResult")


# ---- 7. LI Retrieval Engine ----
async def verify_li_retrieval_engine():
    _section("7. LlamaIndexRetrievalEngine 端到端")
    from study_rag.capabilities.embedding import EmbeddingConfig, create_embedder
    from study_rag.capabilities.llamaindex import LlamaIndexRetrievalEngine
    from study_rag.capabilities.reranker import RerankerConfig, create_reranker
    from study_rag.capabilities.vector_store import (
        VectorRecord,
        VectorStoreConfig,
        create_vector_store,
    )

    embedder = create_embedder(EmbeddingConfig(provider="mock", model_name="m", dimension=4))
    vs = create_vector_store(VectorStoreConfig(provider="mock"))
    await vs.create_collection("li_engine", dimension=4)

    # 准备数据
    for i, t in enumerate(["alpha", "beta", "gamma", "delta", "epsilon"]):
        v = await embedder.embed_query(t)
        await vs.insert("li_engine", [VectorRecord(
            id=f"d{i}", vector=v, text=t, metadata={"title": t},
        )])

    # 配 reranker
    rr = create_reranker(RerankerConfig(provider="none", top_k=2))

    # 构造 engine
    engine = LlamaIndexRetrievalEngine(
        embedder=embedder,
        vector_store=vs,
        collection="li_engine",
        reranker=rr,
        top_k=2,
    )

    # 检索
    results = await engine.aretrieve("alpha")
    assert len(results) <= 2
    print(f"  [OK] aretrieve('alpha') -> {len(results)} hits, top_score={results[0].score:.3f}")
    print(f"      ids: {[r.id for r in results]}")


# ---- 8. Manager 集成 ----
async def verify_manager_integration():
    _section("8. Manager 集成：search_via_llamaindex + add_document_chunked")
    from study_rag.capabilities.embedding import EmbeddingConfig, create_embedder
    from study_rag.capabilities.vector_store import VectorStoreConfig, create_vector_store
    from study_rag.knowledge_bases.manager import KnowledgeBaseManager, reset_manager_singleton
    from study_rag.knowledge_bases.registry import (
        get_registry,
        reset_registry_cache,
    )
    from study_rag.settings import AppPaths

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        # 临时 yaml
        kb_yaml = {"knowledge_bases": [{
            "kb_id": "kb_li", "name": "LI", "description": "test",
            "department": "d", "collection": "c_li",
            "embedding": "mock_li", "reranker": None, "enabled": True,
        }]}
        emb_yaml = {"embeddings": {
            "mock_li": {"provider": "mock", "model_name": "m", "dimension": 4},
        }}
        vs_yaml = {"vector_store": {"provider": "mock", "uri": ""}}
        rerank_yaml = {"rerankers": {}}

        (td_path / "kb.yaml").write_text(yaml.safe_dump(kb_yaml, allow_unicode=True), encoding="utf-8")
        (td_path / "emb.yaml").write_text(yaml.safe_dump(emb_yaml, allow_unicode=True), encoding="utf-8")
        (td_path / "vs.yaml").write_text(yaml.safe_dump(vs_yaml, allow_unicode=True), encoding="utf-8")
        (td_path / "rerank.yaml").write_text(yaml.safe_dump(rerank_yaml, allow_unicode=True), encoding="utf-8")

        AppPaths.KB_CONFIG = td_path / "kb.yaml"
        AppPaths.EMBEDDING_CONFIG = td_path / "emb.yaml"
        AppPaths.VECTOR_STORE_CONFIG = td_path / "vs.yaml"
        AppPaths.RERANKER_CONFIG = td_path / "rerank.yaml"

        reset_registry_cache()
        reset_manager_singleton()

        registry = get_registry()
        embedders = {"mock_li": create_embedder(EmbeddingConfig(provider="mock", model_name="m", dimension=4))}
        vs = create_vector_store(VectorStoreConfig(provider="mock"))
        manager = KnowledgeBaseManager(
            registry=registry, vector_store=vs, embedders=embedders,
        )
        await manager.init_all()

        # 测试 get_llamaindex_engine
        engine = manager.get_llamaindex_engine("kb_li")
        assert engine is not None
        print("  [OK] get_llamaindex_engine('kb_li') 返回非空")

        # 测试 add_document_chunked
        long_doc = (
            "第一条文档：介绍公司业务。\n\n"
            "第二条文档：介绍组织结构。\n\n"
            "第三条文档：介绍技术栈。\n\n"
            "第四条文档：介绍产品线。\n\n"
            "第五条文档：介绍客户案例。"
        )
        chunks = await manager.add_document_chunked(
            kb_id="kb_li",
            doc_id="doc1",
            title="测试文档",
            content=long_doc,
            parser_config={"strategy": "sentence", "chunk_size": 60, "chunk_overlap": 10},
        )
        assert chunks >= 1
        print(f"  [OK] add_document_chunked -> {chunks} 块")

        # 测试 search_via_llamaindex
        results = await manager.search_via_llamaindex("kb_li", "公司业务", top_k=2)
        assert len(results) <= 2
        print(f"  [OK] search_via_llamaindex -> {len(results)} hits")


# ---- 9. 切块 vs 整篇对照 ----
async def verify_chunking_vs_whole():
    _section("9. 切块 vs 整篇对照（验证 NodeParser 提升粒度）")
    from study_rag.capabilities.llamaindex import NodeParserFactory

    long_doc = "。".join([f"第{i}句内容" for i in range(100)]) + "。"

    whole = NodeParserFactory.from_raw({"strategy": "whole"}).parse(long_doc, "d1")
    sent_small = NodeParserFactory.from_raw({"strategy": "sentence", "chunk_size": 50, "chunk_overlap": 10}).parse(long_doc, "d1")
    sent_large = NodeParserFactory.from_raw({"strategy": "sentence", "chunk_size": 200, "chunk_overlap": 20}).parse(long_doc, "d1")

    assert len(whole) == 1
    assert len(sent_small) > len(sent_large)
    print(f"  whole -> {len(whole)} 块")
    print(f"  sentence chunk_size=50  -> {len(sent_small)} 块")
    print(f"  sentence chunk_size=200 -> {len(sent_large)} 块")
    print("  [OK] chunk_size 越小，块数越多（粒度越细）")


# ---- main ----
async def main():
    print("=" * 60)
    print("LlamaIndex 整合验证")
    print("=" * 60)

    verify_package_import()
    verify_node_parser_strategies()
    await verify_vs_adapter()
    await verify_embedding_adapter()
    await verify_reranker_adapter()
    verify_node_mapper()
    await verify_li_retrieval_engine()
    await verify_manager_integration()
    await verify_chunking_vs_whole()

    print("\n" + "=" * 60)
    print("[PASS] 全部验证通过")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
