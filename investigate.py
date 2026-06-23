"""Investigate: Milvus delete 行为 + 真实 chunk 状态。"""
import asyncio
import time
from httpx import ASGITransport, AsyncClient
from study_rag.app import create_app


async def main():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # 1. summary 当前状态
        r = await c.get("/admin/kbs/rd_test")
        s = r.json()
        print(f"当前: docs={s['document_count']} chunks={s['chunk_count']}")

        # 2. 模拟一个新 doc_id='test_x'，插入 3 个 chunks
        # 直接走 vs，避免 add_document
        from study_rag.capabilities.vector_store import VectorRecord
        vs = app.state.manager._vector_store
        await vs.insert(
            "kb_rd_test",
            [
                VectorRecord(id="tx-0", vector=[0.1]*1024, text="a", metadata={"doc_id": "test_x", "title": "TX"}),
                VectorRecord(id="tx-1", vector=[0.1]*1024, text="b", metadata={"doc_id": "test_x", "title": "TX"}),
                VectorRecord(id="tx-2", vector=[0.1]*1024, text="c", metadata={"doc_id": "test_x", "title": "TX"}),
            ],
        )
        # flush + 重新查 count
        time.sleep(1)
        r = await c.get("/admin/kbs/rd_test")
        s = r.json()
        print(f"插入 test_x (3 chunks) 后: chunks={s['chunk_count']}")

        # 3. 查 test_x 的 chunks（看 metadata 结构）
        r = await c.get("/admin/kbs/rd_test/documents/test_x/chunks?limit=10")
        if r.status_code == 200:
            data = r.json()
            print(f"\ntest_x chunks (从 list_chunks):")
            for ch in data.get("chunks", [])[:3]:
                print(f"  id={ch.get('id')!r} chunk_index={ch.get('chunk_index')}")
                # 这里没有 metadata，需要直接 query

        # 4. 直接 query 看 metadata 真实结构
        from study_rag.capabilities.vector_store.filters import matches_filter
        all_chunks = await vs.query("kb_rd_test", filter_expr={"doc_id": "test_x"}, limit=10)
        print(f"\nvs.query(filter={{doc_id: 'test_x'}}): {len(all_chunks)} 条")
        for r_ in all_chunks[:3]:
            print(f"  id={r_.id!r} metadata={r_.metadata}")

        # 5. 删除 test_x（用 Manager 接口）
        # 但要先把它加到 _docs
        from study_rag.knowledge_bases.models import DocumentMeta
        async with app.state.manager._lock:
            app.state.manager._docs.setdefault("test_kb", {})["test_x"] = DocumentMeta(
                doc_id="test_x", kb_id="test_kb", title="TX",
                source=None, content="text", metadata={},
            )
        ok = await app.state.manager.delete_document("test_kb", "test_x")
        print(f"\ndelete_document('test_x'): ok={ok}")

        # 6. 立即查 + 3 秒后查
        time.sleep(0.5)
        s1 = (await c.get("/admin/kbs/rd_test")).json()
        print(f"删除 0.5s 后: chunks={s1['chunk_count']}")
        time.sleep(3)
        s2 = (await c.get("/admin/kbs/rd_test")).json()
        print(f"删除 3s 后:   chunks={s2['chunk_count']}")
        time.sleep(5)
        s3 = (await c.get("/admin/kbs/rd_test")).json()
        print(f"删除 8s 后:   chunks={s3['chunk_count']}")


asyncio.run(main())
