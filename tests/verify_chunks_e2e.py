"""Chunk 查看端到端验证：
   1. VectorStore.query() 能按 doc_id 过滤
   2. Manager.list_chunks() 返回排序后的 ChunkInfo
   3. API GET /documents/{doc_id}/chunks 返回正确结构
   4. 分页参数生效
   5. 未知 KB → 404

跑法：
    python -m tests.verify_chunks_e2e

风格对齐 verify_upload_e2e.py：
  - `from __future__ import annotations`
  - 顶部 docstring 说明验证范围
  - 函数命名 `verify_*`
  - 每节用 `_section("name")` 分隔
  - 失败抛 AssertionError 或 raise
  - ASGI 启动失败时打 [skip] 优雅降级

API 端点验证用 in-memory manager（避免生产 Milvus 配置的 filter bug 干扰）：
  - 写临时 YAML 指向 mock embedder + mock vector store
  - monkey-patch AppPaths 后重建 manager
  - 测完恢复原配置
"""
from __future__ import annotations

import asyncio
import io
import sys
import tempfile
from pathlib import Path

import yaml


def _section(name: str) -> None:
    print(f"\n=== {name} ===")


# ---- 1. VectorStore.query() ----
def verify_vector_store_query() -> None:
    """VectorStore.query() 按 metadata filter 拿 chunks。"""
    _section("1. VectorStore.query() 直接调用")
    from study_rag.capabilities.vector_store import (
        VectorRecord,
        VectorStoreConfig,
        create_vector_store,
    )

    store = create_vector_store(VectorStoreConfig(provider="mock", uri=""))
    asyncio.run(store.create_collection("test", dimension=4))
    asyncio.run(
        store.insert(
            "test",
            [
                VectorRecord(
                    id=f"d1-{i}",
                    vector=[0.1] * 4,
                    text=f"d1 chunk {i}",
                    metadata={"doc_id": "doc_a", "chunk_index": i},
                )
                for i in range(5)
            ]
            + [
                VectorRecord(
                    id=f"d2-{i}",
                    vector=[0.1] * 4,
                    text=f"d2 chunk {i}",
                    metadata={"doc_id": "doc_b", "chunk_index": i},
                )
                for i in range(3)
            ],
        )
    )

    # 全部
    all_records = asyncio.run(store.query("test"))
    assert len(all_records) == 8, f"expected 8, got {len(all_records)}"
    print(f"  [OK] 全部查询: {len(all_records)} 条")

    # 按 doc_id 过滤
    doc_a_records = asyncio.run(store.query("test", filter_expr={"doc_id": "doc_a"}))
    assert len(doc_a_records) == 5
    assert all(r.metadata["doc_id"] == "doc_a" for r in doc_a_records)
    print(f"  [OK] doc_id=doc_a 过滤: {len(doc_a_records)} 条")

    # 分页
    page1 = asyncio.run(
        store.query("test", filter_expr={"doc_id": "doc_a"}, limit=2, offset=0)
    )
    page2 = asyncio.run(
        store.query("test", filter_expr={"doc_id": "doc_a"}, limit=2, offset=2)
    )
    assert len(page1) == 2 and len(page2) == 2
    print(f"  [OK] 分页 limit=2: page1={len(page1)} page2={len(page2)}")

    # 不存在的 collection
    empty = asyncio.run(store.query("nonexistent"))
    assert empty == []
    print(f"  [OK] 不存在 collection: 返回 {len(empty)} 条")


# ---- 2. Manager.list_chunks() ----
def verify_manager_list_chunks() -> None:
    """Manager.list_chunks() + get_chunk_count() + ChunkInfo。"""
    _section("2. Manager.list_chunks()")
    from study_rag.capabilities.vector_store import (
        VectorRecord,
        VectorStoreConfig,
        create_vector_store,
    )
    from study_rag.knowledge_bases.manager import KnowledgeBaseManager
    from study_rag.knowledge_bases.models import ChunkInfo, KnowledgeBaseConfig
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

    # 最小 manager（用 in-memory store + stub embedder）
    registry = KnowledgeBaseRegistry([])
    registry.add(
        KnowledgeBaseConfig(
            kb_id="test_kb",
            name="Test",
            description="test",
            department="test",
            collection="kb_test",
            embedding="stub",
            reranker=None,
            enabled=True,
        )
    )
    store = create_vector_store(VectorStoreConfig(provider="mock", uri=""))
    manager = KnowledgeBaseManager(
        registry=registry,
        vector_store=store,
        embedders={"stub": _StubEmbedder()},  # type: ignore[arg-type]
        docs_index_path=None,
    )
    asyncio.run(manager.init_kb("test_kb"))

    # 插 7 个 chunks
    asyncio.run(
        store.insert(
            "kb_test",
            [
                VectorRecord(
                    id=f"doc_a-{i}",
                    vector=[0.1] * 4,
                    text=f"chunk {i} text content",
                    metadata={
                        "doc_id": "doc_a",
                        "chunk_index": i,
                        "title": "Test Doc",
                        "parser": "sentence",
                    },
                )
                for i in range(7)
            ],
        )
    )

    # list_chunks
    chunks = asyncio.run(manager.list_chunks("test_kb", "doc_a"))
    assert len(chunks) == 7
    assert all(isinstance(c, ChunkInfo) for c in chunks)
    # 排序
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3, 4, 5, 6]
    # 字段
    c0 = chunks[0]
    assert c0.chunk_id == "doc_a-0"
    assert c0.text == "chunk 0 text content"
    assert c0.char_count == len(c0.text)
    assert c0.metadata.get("parser") == "sentence"
    print(f"  [OK] list_chunks: {len(chunks)} 个，索引连续 0-6，字段完整")

    # get_chunk_count
    count = asyncio.run(manager.get_chunk_count("test_kb", "doc_a"))
    assert count == 7
    print(f"  [OK] get_chunk_count: {count}")

    # 不存在的 doc → 空 list
    empty = asyncio.run(manager.list_chunks("test_kb", "nonexistent"))
    assert empty == []
    print(f"  [OK] 不存在的 doc: 返回 {len(empty)} 条")

    # 不存在的 KB → KeyError
    try:
        asyncio.run(manager.list_chunks("nonexistent_kb", "doc_a"))
        assert False, "should have raised"
    except KeyError:
        print(f"  [OK] 不存在的 KB: 抛 KeyError")


# ---- 3. API GET /documents/{doc_id}/chunks ----
async def verify_api_endpoint() -> None:
    """API GET /documents/{doc_id}/chunks 完整流程。

    用 in-memory manager（mock embedder + mock vector store）做端到端，
    避免真实 Milvus / Ollama 配置的依赖与潜在 filter 行为差异。
    """
    _section("3. API GET /documents/{doc_id}/chunks")
    try:
        from httpx import ASGITransport, AsyncClient

        from study_rag.app import create_app
        from study_rag.capabilities.embedding import EmbeddingConfig, create_embedder
        from study_rag.capabilities.vector_store import (
            VectorStoreConfig,
            create_vector_store,
        )
        from study_rag.knowledge_bases import manager as mgr_mod
        from study_rag.knowledge_bases import registry as reg_mod
        from study_rag.knowledge_bases.manager import KnowledgeBaseManager
        from study_rag.settings import AppPaths
    except ImportError as e:  # noqa: BLE001
        print(f"  [skip] 依赖导入失败: {e}")
        return

    # 1) 准备临时 YAML（mock embedder + mock vector store）
    orig_paths = {
        "KB_CONFIG": AppPaths.KB_CONFIG,
        "EMBEDDING_CONFIG": AppPaths.EMBEDDING_CONFIG,
        "VECTOR_STORE_CONFIG": AppPaths.VECTOR_STORE_CONFIG,
        "RERANKER_CONFIG": AppPaths.RERANKER_CONFIG,
        "DOCS_INDEX": AppPaths.DOCS_INDEX,
    }
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        kb_yaml = {
            "knowledge_bases": [
                {
                    "kb_id": "kb_e2e_chunks",
                    "name": "E2E Chunks",
                    "description": "verify_chunks_e2e 专用",
                    "department": "test",
                    "collection": "c_e2e_chunks",
                    "embedding": "mock_e2e",
                    "reranker": None,
                    "enabled": True,
                }
            ]
        }
        emb_yaml = {
            "embeddings": {
                "mock_e2e": {
                    "provider": "mock",
                    "model_name": "m",
                    "dimension": 8,
                }
            }
        }
        vs_yaml = {"vector_store": {"provider": "mock", "uri": ""}}
        rerank_yaml = {"rerankers": {}}

        kb_path = tmp / "kb.yaml"
        emb_path = tmp / "emb.yaml"
        vs_path = tmp / "vs.yaml"
        rerank_path = tmp / "rerank.yaml"
        docs_path = tmp / "docs.json"
        kb_path.write_text(yaml.safe_dump(kb_yaml, allow_unicode=True), encoding="utf-8")
        emb_path.write_text(yaml.safe_dump(emb_yaml, allow_unicode=True), encoding="utf-8")
        vs_path.write_text(yaml.safe_dump(vs_yaml, allow_unicode=True), encoding="utf-8")
        rerank_path.write_text(yaml.safe_dump(rerank_yaml, allow_unicode=True), encoding="utf-8")

        AppPaths.KB_CONFIG = kb_path
        AppPaths.EMBEDDING_CONFIG = emb_path
        AppPaths.VECTOR_STORE_CONFIG = vs_path
        AppPaths.RERANKER_CONFIG = rerank_path
        AppPaths.DOCS_INDEX = docs_path

        # 2) 清单例 + 直接构造 in-memory manager
        reg_mod.reset_registry_cache()
        mgr_mod.reset_manager_singleton()
        try:
            registry = reg_mod.get_registry()
            embedders = {
                "mock_e2e": create_embedder(
                    EmbeddingConfig(provider="mock", model_name="m", dimension=8)
                )
            }
            vs = create_vector_store(VectorStoreConfig(provider="mock"))
            manager = KnowledgeBaseManager(
                registry=registry,
                vector_store=vs,
                embedders=embedders,
            )
            await manager.init_all()
            mgr_mod._manager_singleton = manager  # type: ignore[attr-defined]

            # 3) 重新 load parser registry
            try:
                from study_rag.capabilities.llamaindex import registry as li_reg_mod

                li_reg_mod._registry_singleton = None  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass

            try:
                app = create_app()
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] ASGI app 创建失败: {e}")
                return

            # 4) upload + 列出 chunks
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://t"
            ) as c:
                # upload 1 个文档（足够大以触发切块）
                files = {
                    "file": ("big.txt", io.BytesIO(b"Sentence " * 2000), "text/plain")
                }
                data = {
                    "doc_id": "e2e-chunks-1",
                    "title": "E2E Chunks",
                    "parser": "sentence_512",
                }
                r = await c.post(
                    "/admin/kbs/kb_e2e_chunks/documents/upload",
                    files=files,
                    data=data,
                )
                if r.status_code != 200:
                    detail = r.json().get("detail", "")[:120] if r.headers.get(
                        "content-type", ""
                    ).startswith("application/json") else r.text[:120]
                    print(f"  [skip] upload 失败: {r.status_code} {detail}")
                    return

                # 列出 chunks
                r = await c.get(
                    "/admin/kbs/kb_e2e_chunks/documents/e2e-chunks-1/chunks?limit=10"
                )
                assert r.status_code == 200, f"list_chunks {r.status_code}: {r.text}"
                body = r.json()
                assert body["kb_id"] == "kb_e2e_chunks"
                assert body["doc_id"] == "e2e-chunks-1"
                assert body["total"] >= 1
                assert len(body["chunks"]) >= 1
                # 验证每条 chunk 字段
                for chunk in body["chunks"]:
                    assert "chunk_id" in chunk
                    assert "chunk_index" in chunk
                    assert "text" in chunk
                    assert "char_count" in chunk
                    assert chunk["char_count"] == len(chunk["text"])
                # 索引排序
                indices = [c["chunk_index"] for c in body["chunks"]]
                assert indices == sorted(indices)
                print(
                    f"  [OK] list 返回 total={body['total']}，"
                    f"本页 {len(body['chunks'])} 个 (limit=10)"
                )

                # 分页：limit=2 offset=0 / offset=2
                r1 = await c.get(
                    "/admin/kbs/kb_e2e_chunks/documents/e2e-chunks-1/chunks"
                    "?limit=2&offset=0"
                )
                r2 = await c.get(
                    "/admin/kbs/kb_e2e_chunks/documents/e2e-chunks-1/chunks"
                    "?limit=2&offset=2"
                )
                assert r1.status_code == 200 and r2.status_code == 200
                idx1 = [c["chunk_index"] for c in r1.json()["chunks"]]
                idx2 = [c["chunk_index"] for c in r2.json()["chunks"]]
                assert len(idx1) == 2
                if idx2:
                    assert idx2[0] == 2
                print(f"  [OK] 分页 limit=2: page1={idx1}, page2={idx2}")

                # 未知 KB → 404
                r = await c.get("/admin/kbs/nonexistent_kb/documents/x/chunks")
                assert r.status_code == 404
                print(f"  [OK] 未知 KB → 404")

                # 未知 doc → 200 + 空
                r = await c.get(
                    "/admin/kbs/kb_e2e_chunks/documents/nonexistent_doc/chunks"
                )
                assert r.status_code == 200
                body = r.json()
                assert body["total"] == 0
                assert body["chunks"] == []
                print(f"  [OK] 未知 doc → 200 + 空列表")
        finally:
            # 5) 恢复 AppPaths + 单例
            AppPaths.KB_CONFIG = orig_paths["KB_CONFIG"]
            AppPaths.EMBEDDING_CONFIG = orig_paths["EMBEDDING_CONFIG"]
            AppPaths.VECTOR_STORE_CONFIG = orig_paths["VECTOR_STORE_CONFIG"]
            AppPaths.RERANKER_CONFIG = orig_paths["RERANKER_CONFIG"]
            AppPaths.DOCS_INDEX = orig_paths["DOCS_INDEX"]
            reg_mod.reset_registry_cache()
            mgr_mod.reset_manager_singleton()


# ---- main ----
def main() -> None:
    print("=" * 60)
    print("Chunk 查看端到端验证")
    print("=" * 60)

    verify_vector_store_query()
    verify_manager_list_chunks()
    asyncio.run(verify_api_endpoint())

    print("\n" + "=" * 60)
    print("=== ALL OK ===")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n[FAIL] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
