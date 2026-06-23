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

    @property
    def dimension(self) -> int:
        return 4

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
