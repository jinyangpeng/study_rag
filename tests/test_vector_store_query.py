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


# ---- Phase 6.5: metadata JSON path 行为（Milvus filter 翻译） ----


@pytest.mark.asyncio
async def test_query_filter_metadata_subfield(mock_store):
    """mock 按 metadata 子字段过滤时，行为要等价于 r.metadata["doc_id"] == X。

    之前 filter 翻译 bug：mock 用 matches_filter 走 r.metadata["doc_id"]
    实际是对的（因为 mock 看的就是 metadata 字段），但 milvus 生成的是
    `doc_id == X`（顶层字段语法），两边行为不一致。
    """
    await mock_store.create_collection("test", dimension=4)
    await mock_store.insert(
        "test",
        [
            VectorRecord(id="1", vector=[0.1] * 4, text="a",
                         metadata={"doc_id": "d1", "title": "Doc 1"}),
            VectorRecord(id="2", vector=[0.1] * 4, text="b",
                         metadata={"doc_id": "d1", "title": "Doc 1"}),
            VectorRecord(id="3", vector=[0.1] * 4, text="c",
                         metadata={"doc_id": "d2", "title": "Doc 2"}),
        ],
    )
    # 按 doc_id 过滤（最常见）
    res = await mock_store.query("test", filter_expr={"doc_id": "d1"})
    assert len(res) == 2
    assert all(r.metadata["doc_id"] == "d1" for r in res)


def test_milvus_filter_uses_json_path_for_metadata_fields():
    """to_milvus_expr 对 metadata 字段要用 JSON path 语法。

    之前：to_milvus_expr({"doc_id": "X"}) -> 'doc_id == "X"'（错误）
    修复后：to_milvus_expr({"doc_id": "X"}) -> 'metadata["doc_id"] == "X"'（正确）
    """
    from study_rag.capabilities.vector_store.filters import to_milvus_expr

    expr = to_milvus_expr({"doc_id": "X"})
    # 必须包含 metadata["doc_id"] 而不是顶层 doc_id
    assert "metadata" in expr
    assert "doc_id" in expr
    # 不能是 'doc_id == "X"' 单独的顶层引用
    assert not expr.strip().startswith("doc_id ")


def test_milvus_filter_numeric_metadata():
    """chunk_index 是数字类型，生成不带引号的 JSON path。"""
    from study_rag.capabilities.vector_store.filters import to_milvus_expr

    expr = to_milvus_expr({"chunk_index": 5})
    assert 'metadata["chunk_index"]' in expr
    assert "5" in expr
    # 不能是字符串 "5"（除非格式化有意）
    # 通常 Milvus JSON path 数字字段可以不带引号


def test_milvus_filter_complex_with_metadata():
    """复杂 filter：doc_id + chunk_index range。"""
    from study_rag.capabilities.vector_store.filters import to_milvus_expr

    expr = to_milvus_expr(
        {"doc_id": "d1", "chunk_index__gte": 2}
    )
    assert 'metadata["doc_id"]' in expr
    assert 'metadata["chunk_index"]' in expr
    # 验证包含 AND
    assert " and " in expr.lower()


# ---- Phase 6.6: count() 方法：O(1) 拿 collection 记录数 ----


@pytest.mark.asyncio
async def test_count_empty_collection(mock_store):
    """空 collection → count = 0。"""
    await mock_store.create_collection("test", dimension=4)
    n = await mock_store.count("test")
    assert n == 0


@pytest.mark.asyncio
async def test_count_after_insert(mock_store):
    """插入 N 条后 → count = N。"""
    await mock_store.create_collection("test", dimension=4)
    await mock_store.insert(
        "test",
        [
            VectorRecord(id=f"r{i}", vector=[0.1] * 4, text=f"t{i}",
                         metadata={"doc_id": f"d{i % 2}"})
            for i in range(10)
        ],
    )
    n = await mock_store.count("test")
    assert n == 10


@pytest.mark.asyncio
async def test_count_nonexistent_collection(mock_store):
    """不存在的 collection → count = 0（不是抛错）。"""
    n = await mock_store.count("nonexistent")
    assert n == 0


@pytest.mark.asyncio
async def test_count_after_delete(mock_store):
    """删除部分记录后 → count 正确。"""
    await mock_store.create_collection("test", dimension=4)
    await mock_store.insert(
        "test",
        [
            VectorRecord(id=f"r{i}", vector=[0.1] * 4, text=f"t{i}", metadata={})
            for i in range(5)
        ],
    )
    await mock_store.delete("test", ["r0", "r1"])
    n = await mock_store.count("test")
    assert n == 3
