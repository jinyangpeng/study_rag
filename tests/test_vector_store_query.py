"""VectorStore.query() 测试：按 metadata 过滤拿 chunks，不需要 query_vector。"""
from __future__ import annotations

import pytest

from study_rag.capabilities.vector_store import (
    VectorRecord,
    VectorStoreConfig,
    create_vector_store,
)


@pytest.fixture
def mock_store():
    cfg = VectorStoreConfig(provider="mock", uri="")
    return create_vector_store(cfg)


@pytest.mark.asyncio
async def test_query_no_filter_returns_all(mock_store):
    """无 filter 时返回所有记录。"""
    await mock_store.create_collection("test", dimension=4)
    await mock_store.insert(
        "test",
        [
            VectorRecord(id="1", vector=[0.1] * 4, text="a", metadata={"doc_id": "d1"}),
            VectorRecord(id="2", vector=[0.2] * 4, text="b", metadata={"doc_id": "d1"}),
            VectorRecord(id="3", vector=[0.3] * 4, text="c", metadata={"doc_id": "d2"}),
        ],
    )
    res = await mock_store.query("test")
    assert len(res) == 3
    assert sorted(r.id for r in res) == ["1", "2", "3"]


@pytest.mark.asyncio
async def test_query_filter_by_metadata(mock_store):
    """按 metadata 过滤。"""
    await mock_store.create_collection("test", dimension=4)
    await mock_store.insert(
        "test",
        [
            VectorRecord(id="1", vector=[0.1] * 4, text="a", metadata={"doc_id": "d1"}),
            VectorRecord(id="2", vector=[0.2] * 4, text="b", metadata={"doc_id": "d1"}),
            VectorRecord(id="3", vector=[0.3] * 4, text="c", metadata={"doc_id": "d2"}),
        ],
    )
    res = await mock_store.query("test", filter_expr={"doc_id": "d1"})
    assert len(res) == 2
    assert all(r.metadata["doc_id"] == "d1" for r in res)


@pytest.mark.asyncio
async def test_query_pagination(mock_store):
    """limit + offset 分页。"""
    await mock_store.create_collection("test", dimension=4)
    await mock_store.insert(
        "test",
        [
            VectorRecord(id=str(i), vector=[0.1] * 4, text=f"t{i}", metadata={"idx": i})
            for i in range(10)
        ],
    )
    page1 = await mock_store.query("test", limit=3, offset=0)
    page2 = await mock_store.query("test", limit=3, offset=3)
    assert len(page1) == 3
    assert len(page2) == 3
    # 排序按 id 不保证，但 page1 和 page2 不重叠
    page1_ids = {r.id for r in page1}
    page2_ids = {r.id for r in page2}
    assert page1_ids.isdisjoint(page2_ids)


@pytest.mark.asyncio
async def test_query_nonexistent_collection_returns_empty(mock_store):
    """collection 不存在 → 返回空 list。"""
    res = await mock_store.query("nonexistent")
    assert res == []


@pytest.mark.asyncio
async def test_query_complex_filter(mock_store):
    """复杂 filter：doc_id + chunk_index range。"""
    await mock_store.create_collection("test", dimension=4)
    await mock_store.insert(
        "test",
        [
            VectorRecord(id=str(i), vector=[0.1] * 4, text=f"t{i}",
                         metadata={"doc_id": "d1", "chunk_index": i})
            for i in range(5)
        ] + [
            VectorRecord(id=f"d2-{i}", vector=[0.1] * 4, text=f"d2-t{i}",
                         metadata={"doc_id": "d2", "chunk_index": i})
            for i in range(5)
        ],
    )
    res = await mock_store.query(
        "test", filter_expr={"doc_id": "d1", "chunk_index__gte": 2}
    )
    assert len(res) == 3  # idx 2, 3, 4
    assert sorted(r.metadata["chunk_index"] for r in res) == [2, 3, 4]
