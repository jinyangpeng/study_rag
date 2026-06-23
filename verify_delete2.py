"""Verify: 删带 chunks 的文档（002 有 6 chunks），Milvus 真删。"""
import asyncio
from httpx import ASGITransport, AsyncClient
from study_rag.app import create_app


async def main():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # 用 summary 接口（API: GET /admin/kbs/{kb_id}）拿 chunk_count
        r = await c.get("/admin/kbs/rd_test")
        print(f"GET /admin/kbs/rd_test: {r.status_code}")
        s = r.json()
        print(f"  doc_count={s['document_count']}  chunk_count={s['chunk_count']}")

        # 删 002（带 6 chunks）
        r = await c.delete("/admin/kbs/rd_test/documents/002")
        print(f"\nDELETE 002: {r.status_code} body={r.text[:200]}")

        # 验证 summary 数字下降
        r = await c.get("/admin/kbs/rd_test")
        s = r.json()
        print(f"  after delete: doc_count={s['document_count']}  chunk_count={s['chunk_count']}")
        assert s['document_count'] == 0, f"FAIL: doc_count={s['document_count']}"
        assert s['chunk_count'] == 0, f"FAIL: chunk_count={s['chunk_count']} (应该=0,Milvus 删干净了)"
        print("  ✓ chunk_count 从 6 → 0,Milvus 真删了")


asyncio.run(main())
