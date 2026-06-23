"""Verify delete fix: 之前 500，现在应该 200 + chunks 真删了。"""
import asyncio
from httpx import ASGITransport, AsyncClient
from study_rag.app import create_app


async def main():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # 1. 看 rd_test 当前状态
        r = await c.get("/admin/kbs/rd_test/documents")
        print(f"GET docs: {r.status_code}, {len(r.json())} docs")
        for d in r.json():
            print(f"  doc_id={d['doc_id']!r} title={d.get('title')!r}")
        # 看 002 的 chunks 数
        r = await c.get("/admin/kbs/rd_test/documents/002/chunks?limit=5")
        before_002 = r.json().get("total", 0)
        print(f"  002 chunks before delete: {before_002}")

        # 2. 删除 001（之前 500）
        r = await c.delete("/admin/kbs/rd_test/documents/001")
        print(f"\nDELETE /admin/kbs/rd_test/documents/001: {r.status_code}")
        print(f"  body: {r.text[:200]}")

        # 3. 验证删成功了
        r = await c.get("/admin/kbs/rd_test/documents")
        docs_after = [d["doc_id"] for d in r.json()]
        print(f"\nGET docs after: {docs_after}")
        assert "001" not in docs_after, "FAIL: 001 还在"
        print("  ✓ 001 已从内存 _docs 移除")

        r = await c.get("/admin/kbs/rd_test/documents/001/chunks?limit=5")
        # 注意：删完之后 endpoint 应该返回 total=0（没有 chunks）or 404
        print(f"  001 chunks after: status={r.status_code} body_total={r.json().get('total')}")


asyncio.run(main())
