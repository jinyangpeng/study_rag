"""Manager.list_chunks() 测试：从 vector store 按 doc_id 过滤拿 chunks。"""
from __future__ import annotations

import pytest

from study_rag.capabilities.vector_store import (
    VectorRecord,
    VectorStoreConfig,
    create_vector_store,
)
from study_rag.knowledge_bases.manager import KnowledgeBaseManager
from study_rag.knowledge_bases.models import KnowledgeBaseConfig
from study_rag.knowledge_bases.registry import KnowledgeBaseRegistry


class _StubEmbedder:
    """最小 embedder stub：返回固定维度 4 的向量。"""

    class _Cfg:
        model_name = "stub"
        batch_size = 8

    @property
    def dimension(self) -> int:
        return 4

    @property
    def _config(self) -> "_StubEmbedder._Cfg":
        return self._Cfg()

    async def embed_query(self, text: str) -> list[float]:
        return [0.1] * 4

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 4 for _ in texts]


@pytest.fixture
def setup():
    """建一个 KB + manager 配 in-memory vector store + stub embedder。"""
    registry = KnowledgeBaseRegistry([])
    registry.add(
        KnowledgeBaseConfig(
            kb_id="test_kb",
            name="Test KB",
            description="test",
            department="test",
            collection="kb_test",
            embedding="stub",
            reranker=None,
            enabled=True,
        )
    )
    vector_store = create_vector_store(VectorStoreConfig(provider="mock", uri=""))
    manager = KnowledgeBaseManager(
        registry=registry,
        vector_store=vector_store,
        embedders={"stub": _StubEmbedder()},  # type: ignore[arg-type]
        docs_index_path=None,
    )
    return manager, vector_store


@pytest.mark.asyncio
async def test_list_chunks_returns_all_chunks_of_doc(setup):
    manager, vs = setup
    # 插 3 个 chunks，2 个属于 doc_a，1 个属于 doc_b
    await vs.insert(
        "kb_test",
        [
            VectorRecord(id="a-0", vector=[0.1] * 4, text="chunk a 0",
                         metadata={"doc_id": "doc_a", "chunk_index": 0, "title": "A"}),
            VectorRecord(id="a-1", vector=[0.1] * 4, text="chunk a 1",
                         metadata={"doc_id": "doc_a", "chunk_index": 1, "title": "A"}),
            VectorRecord(id="b-0", vector=[0.1] * 4, text="chunk b 0",
                         metadata={"doc_id": "doc_b", "chunk_index": 0, "title": "B"}),
        ],
    )
    chunks = await manager.list_chunks("test_kb", "doc_a")
    assert len(chunks) == 2
    # 按 chunk_index 排序
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert all(c.metadata.get("doc_id") == "doc_a" for c in chunks)


@pytest.mark.asyncio
async def test_list_chunks_empty_for_missing_doc(setup):
    manager, _ = setup
    chunks = await manager.list_chunks("test_kb", "nonexistent")
    assert chunks == []


@pytest.mark.asyncio
async def test_list_chunks_kb_not_found_raises(setup):
    manager, _ = setup
    with pytest.raises(KeyError, match="nonexistent_kb"):
        await manager.list_chunks("nonexistent_kb", "doc_a")


@pytest.mark.asyncio
async def test_list_chunks_pagination(setup):
    manager, vs = setup
    records = [
        VectorRecord(
            id=f"a-{i}",
            vector=[0.1] * 4,
            text=f"chunk {i}",
            metadata={"doc_id": "doc_a", "chunk_index": i, "title": "A"},
        )
        for i in range(10)
    ]
    await vs.insert("kb_test", records)

    page1 = await manager.list_chunks("test_kb", "doc_a", limit=3, offset=0)
    page2 = await manager.list_chunks("test_kb", "doc_a", limit=3, offset=3)
    assert len(page1) == 3
    assert len(page2) == 3
    # 排序：page1 应该是 0, 1, 2
    assert [c.chunk_index for c in page1] == [0, 1, 2]
    assert [c.chunk_index for c in page2] == [3, 4, 5]


@pytest.mark.asyncio
async def test_get_chunk_count(setup):
    manager, vs = setup
    await vs.insert(
        "kb_test",
        [
            VectorRecord(id=f"a-{i}", vector=[0.1] * 4, text=f"t{i}",
                         metadata={"doc_id": "doc_a", "chunk_index": i})
            for i in range(7)
        ],
    )
    count = await manager.get_chunk_count("test_kb", "doc_a")
    assert count == 7


@pytest.mark.asyncio
async def test_get_doc_total_chars(setup):
    """get_doc_total_chars 返回所有 chunk 文本总字符数。"""
    manager, vs = setup
    await vs.insert(
        "kb_test",
        [
            VectorRecord(id="a-0", vector=[0.1] * 4, text="hello",
                         metadata={"doc_id": "doc_a", "chunk_index": 0}),
            VectorRecord(id="a-1", vector=[0.1] * 4, text="world!",
                         metadata={"doc_id": "doc_a", "chunk_index": 1}),
            VectorRecord(id="b-0", vector=[0.1] * 4, text="other-doc",
                         metadata={"doc_id": "doc_b", "chunk_index": 0}),
        ],
    )
    # doc_a = "hello" (5) + "world!" (6) = 11
    assert await manager.get_doc_total_chars("test_kb", "doc_a") == 11
    # doc_b 只有 1 个 chunk
    assert await manager.get_doc_total_chars("test_kb", "doc_b") == 9


@pytest.mark.asyncio
async def test_get_doc_total_chars_handles_chinese_chars(setup):
    """中文/Unicode 字符应被正确计数（Python str len 算 codepoint）。"""
    manager, vs = setup
    await vs.insert(
        "kb_test",
        [
            VectorRecord(id="c-0", vector=[0.1] * 4, text="你好世界",
                         metadata={"doc_id": "doc_c", "chunk_index": 0}),
        ],
    )
    # "你好世界" = 4 个 codepoint
    assert await manager.get_doc_total_chars("test_kb", "doc_c") == 4


@pytest.mark.asyncio
async def test_get_doc_parser(setup):
    """get_doc_parser 从 vector store chunks metadata 拿真实 parser 标签。"""
    manager, vs = setup
    await vs.insert(
        "kb_test",
        [
            VectorRecord(id="a-0", vector=[0.1] * 4, text="chunk0",
                         metadata={"doc_id": "doc_a", "chunk_index": 0, "parser": "sentence_512"}),
            VectorRecord(id="a-1", vector=[0.1] * 4, text="chunk1",
                         metadata={"doc_id": "doc_a", "chunk_index": 1, "parser": "sentence_512"}),
        ],
    )
    # doc_a 的所有 chunk 都有 parser='sentence_512'
    assert await manager.get_doc_parser("test_kb", "doc_a") == "sentence_512"


@pytest.mark.asyncio
async def test_get_doc_parser_returns_none_for_missing_doc(setup):
    """doc 不存在 → 返回 None（不抛错）。"""
    manager, _ = setup
    assert await manager.get_doc_parser("test_kb", "nonexistent") is None


@pytest.mark.asyncio
async def test_get_doc_parser_returns_none_when_chunks_have_no_parser(setup):
    """老数据 chunks metadata 里没有 parser 字段 → 返回 None。"""
    manager, vs = setup
    await vs.insert(
        "kb_test",
        [
            VectorRecord(id="a-0", vector=[0.1] * 4, text="chunk0",
                         metadata={"doc_id": "doc_a", "chunk_index": 0}),  # 没 parser 字段
        ],
    )
    assert await manager.get_doc_parser("test_kb", "doc_a") is None


# ---- Phase 6.6: get_total_chunk_count + summary.chunk_count ----


@pytest.mark.asyncio
async def test_get_total_chunk_count(setup):
    """get_total_chunk_count 走 count()，不拉数据。"""
    manager, vs = setup
    await vs.insert(
        "kb_test",
        [
            VectorRecord(id=f"a-{i}", vector=[0.1] * 4, text=f"t{i}",
                         metadata={"doc_id": f"d{i % 3}"})
            for i in range(15)
        ],
    )
    n = await manager.get_total_chunk_count("test_kb")
    assert n == 15


@pytest.mark.asyncio
async def test_summary_includes_chunk_count(setup):
    """get_summary 返回的 summary 里有 chunk_count > 0。"""
    manager, vs = setup
    await vs.insert(
        "kb_test",
        [
            VectorRecord(id=f"a-{i}", vector=[0.1] * 4, text=f"t{i}", metadata={})
            for i in range(5)
        ],
    )
    summary = await manager.get_summary("test_kb")
    assert summary is not None
    assert summary.chunk_count == 5
    assert summary.document_count == 0  # 没 add_document，所以 0


@pytest.mark.asyncio
async def test_get_total_chunk_count_nonexistent_kb_returns_zero(setup):
    manager, _ = setup
    n = await manager.get_total_chunk_count("nonexistent_kb")
    assert n == 0


@pytest.mark.asyncio
async def test_list_summaries_includes_chunk_count(setup):
    """list_summaries 返回的每个 summary 都有正确的 chunk_count。"""
    manager, vs = setup
    await vs.insert(
        "kb_test",
        [
            VectorRecord(id=f"a-{i}", vector=[0.1] * 4, text=f"t{i}", metadata={})
            for i in range(4)
        ],
    )
    summaries = await manager.list_summaries()
    assert len(summaries) == 1
    assert summaries[0].chunk_count == 4
    assert summaries[0].kb_id == "test_kb"


# ---- Phase: add_document 必须设置 chunk_count / char_count / parser ----
# 之前 DocumentMeta 没有这三个字段，写入时全部漏设 → list 永远是 0。
# 回归测试覆盖 add_document + add_document_chunked + add_document_from_upload。

@pytest.mark.asyncio
async def test_add_document_sets_chunk_count_char_count_parser(setup):
    """add_document 写入后 DocumentMeta 必须带 chunk_count / char_count / parser。"""
    from study_rag.knowledge_bases.models import DocumentCreate

    manager, _vs = setup
    await manager.add_document(
        DocumentCreate(
            kb_id="test_kb",
            doc_id="doc-1",
            title="Doc 1",
            content="Hello, world!",
            source=None,
            metadata={},
        )
    )
    docs = manager.list_documents("test_kb")
    assert len(docs) == 1
    d = docs[0]
    # add_document 是 whole 模式：1 个 chunk, parser='whole'
    assert d.chunk_count == 1
    assert d.char_count == len("Hello, world!")
    assert d.parser == "whole"


@pytest.mark.asyncio
async def test_add_document_chunked_persists_document_meta(setup):
    """add_document_chunked 必须把 DocumentMeta 存到 _docs（之前漏存 → list 看不到）。

    同时验证 chunk_count = 实际切块数，parser 来自 NodeParserFactory。
    """
    from study_rag.knowledge_bases.models import DocumentCreate

    manager, _vs = setup
    # 走 add_document_chunked：注意 _StubEmbedder 返回 4 维向量，
    # LI 的 sentence splitter 会按 chunk_size=512 切，对 20 字符的短文本通常 1 块。
    # 用 chunk_size=5 强制多块
    from study_rag.knowledge_bases.models import DocumentCreate as DC

    n_chunks = await manager.add_document_chunked(
        kb_id="test_kb",
        doc_id="chunked-1",
        title="Chunked Doc",
        content="alpha beta gamma delta epsilon zeta eta theta",
        source="",
        parser_config={"strategy": "sentence", "chunk_size": 10, "chunk_overlap": 0},
    )
    assert n_chunks >= 1

    docs = manager.list_documents("test_kb")
    assert len(docs) == 1, "add_document_chunked 漏存 DocumentMeta → list 看不到"
    d = docs[0]
    assert d.chunk_count == n_chunks
    assert d.char_count == len("alpha beta gamma delta epsilon zeta eta theta")
    assert d.parser in ("sentence", "token", "semantic", "whole")


@pytest.mark.asyncio
async def test_list_documents_enriches_chunk_count_from_vector_store(setup):
    """list_documents API 必须实时从 vector store 拿 chunk_count（不能漏）。

    模拟：往 _docs 写 DocumentMeta（chunk_count=0），但 vector store 已有 3 个 chunk。
    验证 enrich 逻辑：用并发 asyncio.gather 拉实际 chunk_count，替换占位值。
    """
    from study_rag.knowledge_bases.models import DocumentMeta

    manager, vs = setup
    # 1. 写 3 个 chunks 进 vector store（doc_id='x'）
    await vs.insert(
        "kb_test",
        [
            VectorRecord(id=f"x-{i}", vector=[0.1] * 4, text=f"t{i}",
                         metadata={"doc_id": "x", "chunk_index": i})
            for i in range(3)
        ],
    )
    # 2. 在 _docs 里塞一个 chunk_count=0 的占位 DocumentMeta
    async with manager._lock:
        manager._docs.setdefault("test_kb", {})["x"] = DocumentMeta(
            doc_id="x",
            kb_id="test_kb",
            title="X",
            source=None,
            content="abc",
            metadata={},
            chunk_count=0,  # 占位值
            char_count=0,
        )
    # 3. 模拟 list_documents endpoint 的 enrich 逻辑
    docs = manager.list_documents("test_kb")

    async def _enrich(doc: DocumentMeta) -> DocumentMeta:
        try:
            n = await manager.get_chunk_count("test_kb", doc.doc_id)
        except Exception:  # noqa: BLE001
            n = doc.chunk_count
        # char_count 兜底：仅在 0 时用 content 长度回填
        char_count = doc.char_count
        if char_count == 0 and doc.content:
            char_count = len(doc.content)
        return doc.model_copy(update={
            "chunk_count": n,
            "char_count": char_count,
        })

    import asyncio
    enriched = await asyncio.gather(*[_enrich(d) for d in docs])
    assert len(enriched) == 1
    assert enriched[0].chunk_count == 3   # vector store 实际值覆盖占位 0
    assert enriched[0].char_count == 3
    assert enriched[0].doc_id == "x"


@pytest.mark.asyncio
async def test_list_documents_enrich_uses_vector_store_for_char_count(setup):
    """回归测试：list_documents enrich **必须**用 vector store 真实字符数覆盖 char_count。

    历史 bug：admin.py list_documents 里写 `char_count=len(doc.content)`，但
    pipeline.py 写入时 content 被截断到 1000 字符，导致 char_count 被错误地
    限制在 1000。即使重启后从 _docs_index.json 加载，老数据也是 1000。

    修复：enrich 用 manager.get_doc_total_chars() 拿所有 chunk 文本的真实总字符数。
    这样无论历史数据怎么错（被旧 enrich 覆盖、被 content 截断），
    都会返回真实值。
    """
    from study_rag.knowledge_bases.models import DocumentMeta

    manager, vs = setup
    # 1. vector store 写 2 个 chunk：doc_id='big'，总文本长度 3000
    text1 = "x" * 1500
    text2 = "y" * 1500
    await vs.insert(
        "kb_test",
        [
            VectorRecord(id="big-0", vector=[0.1] * 4, text=text1,
                         metadata={"doc_id": "big", "chunk_index": 0,
                                   "parser": "sentence_512"}),
            VectorRecord(id="big-1", vector=[0.1] * 4, text=text2,
                         metadata={"doc_id": "big", "chunk_index": 1,
                                   "parser": "sentence_512"}),
        ],
    )
    # 2. _docs 里塞一个历史错误数据：char_count=1000（旧 enrich 用截断 content 算的），
    #    content 也被截断到 1000 字符，parser 也漏存
    async with manager._lock:
        manager._docs.setdefault("test_kb", {})["big"] = DocumentMeta(
            doc_id="big",
            kb_id="test_kb",
            title="Big Doc",
            source=None,
            content="z" * 1000,  # 模拟 pipeline 的 content 截断
            metadata={},
            chunk_count=1,  # 历史错误
            char_count=1000,  # 历史错误（旧 enrich 用 len(content) 算的）
            parser=None,  # 历史漏存
        )

    # 3. enrich（模拟 admin.py list_documents 的 enrich 逻辑）
    docs = manager.list_documents("test_kb")

    async def _enrich(doc: DocumentMeta) -> DocumentMeta:
        try:
            n = await manager.get_chunk_count("test_kb", doc.doc_id)
        except Exception:  # noqa: BLE001
            n = doc.chunk_count
        try:
            total_chars = await manager.get_doc_total_chars("test_kb", doc.doc_id)
        except Exception:  # noqa: BLE001
            total_chars = doc.char_count
        try:
            real_parser = await manager.get_doc_parser("test_kb", doc.doc_id)
        except Exception:  # noqa: BLE001
            real_parser = doc.parser
        return doc.model_copy(update={
            "chunk_count": n,
            "char_count": total_chars,
            "parser": real_parser if real_parser else doc.parser,
        })

    import asyncio
    enriched = await asyncio.gather(*[_enrich(d) for d in docs])
    assert len(enriched) == 1
    # 关键不变量：char_count 必须是 3000（vector store 真实字符数），不能是历史 1000
    assert enriched[0].char_count == 3000, (
        f"enrich char_count={enriched[0].char_count}，应该是 3000"
    )
    # chunk_count 也必须被纠正为真实值 2
    assert enriched[0].chunk_count == 2, (
        f"enrich chunk_count={enriched[0].chunk_count}，应该是 2"
    )
    # parser 也必须从 vector store 拿（不能是历史 None）
    assert enriched[0].parser == "sentence_512", (
        f"enrich parser={enriched[0].parser!r}，应该是 'sentence_512'"
    )
