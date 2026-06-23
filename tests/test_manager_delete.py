"""Manager.delete_document() 测试：删除文档要真删 Milvus chunks。

Phase 6.7: 修复删除文档 500 Internal Server Error。
3 层 bug 叠加：
  1. Milvus 主键是 Int64，不是 String
  2. MilvusVectorStore.delete 假设主键是 VARCHAR（用 `id in ["001"]`），Int64 cast 失败
  3. Manager.delete_document 传的是 doc_id，不是 chunk 真实 ID

正确做法：用 filter `metadata["doc_id"] == "001"` 删所有 doc_id 匹配的 chunks。
"""
from __future__ import annotations

import pytest

from study_rag.capabilities.vector_store import (
    VectorRecord,
    VectorStoreConfig,
    create_vector_store,
)
from study_rag.knowledge_bases.manager import KnowledgeBaseManager
from study_rag.knowledge_bases.models import (
    DocumentMeta,
    KnowledgeBaseConfig,
)
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
async def test_delete_document_removes_chunks_from_milvus(setup):
    """删除文档时，vector store 里该 doc 的所有 chunks 都要删掉。"""
    manager, vs = setup
    # 模拟 3 个 chunks：2 个属于 doc_id='d1'，1 个属于 'd2'
    await vs.insert(
        "kb_test",
        [
            VectorRecord(id="42", vector=[0.1] * 4, text="chunk0",
                         metadata={"doc_id": "d1", "chunk_index": 0}),
            VectorRecord(id="43", vector=[0.1] * 4, text="chunk1",
                         metadata={"doc_id": "d1", "chunk_index": 1}),
            VectorRecord(id="44", vector=[0.1] * 4, text="chunk2",
                         metadata={"doc_id": "d2", "chunk_index": 0}),
        ],
    )
    # 手动设置 _docs（绕过 add_document）
    async with manager._lock:
        manager._docs.setdefault("test_kb", {})["d1"] = DocumentMeta(
            doc_id="d1",
            kb_id="test_kb",
            title="Doc 1",
            source=None,
            content="text",
            metadata={},
        )
    assert await vs.count("kb_test") == 3

    # 删除 d1
    ok = await manager.delete_document("test_kb", "d1")
    assert ok is True
    # 内存 _docs 里 d1 没了
    assert manager.get_document("test_kb", "d1") is None
    # vector store 里 d1 的 chunks 没了（只剩 d2 的 1 个）
    assert await vs.count("kb_test") == 1
    # 验证剩的是 d2 的 chunk
    remaining = await vs.query("kb_test", filter_expr={"doc_id": "d2"})
    assert len(remaining) == 1


@pytest.mark.asyncio
async def test_delete_document_milvus_error_does_not_500(setup):
    """vector store 抛错时，delete_document 仍然要返回 True（内存删成功）。"""
    manager, vs = setup
    async with manager._lock:
        manager._docs.setdefault("test_kb", {})["d1"] = DocumentMeta(
            doc_id="d1",
            kb_id="test_kb",
            title="Doc 1",
            source=None,
            content="text",
            metadata={},
        )

    # 让 vs.delete 抛错
    async def broken_delete(*args, **kwargs):
        raise RuntimeError("Simulated Milvus failure")

    vs.delete = broken_delete  # type: ignore[method-assign]

    # 不应该 500；内存删成功 → 返回 True
    ok = await manager.delete_document("test_kb", "d1")
    assert ok is True
    assert manager.get_document("test_kb", "d1") is None


@pytest.mark.asyncio
async def test_delete_document_nonexistent_returns_false(setup):
    """删除不存在的 doc → 返回 False。"""
    manager, _ = setup
    ok = await manager.delete_document("test_kb", "nonexistent")
    assert ok is False


@pytest.mark.asyncio
async def test_delete_document_unknown_kb_returns_false(setup):
    """未知 KB → 返回 False。"""
    manager, _ = setup
    ok = await manager.delete_document("nonexistent_kb", "d1")
    assert ok is False
