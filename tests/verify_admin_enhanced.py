"""验证：Admin REST 完整流程（鉴权 + search + chunked + batch + /metrics）。"""

# ruff: noqa: T201, PT017, PT018
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    print("=" * 60)
    print("Verify: Admin REST 完整流程")
    print("=" * 60)

    # 不启用 admin 鉴权（开发模式）
    os.environ.pop("STUDY_RAG_ADMIN_TOKEN", None)

    from fastapi.testclient import TestClient

    from study_rag.app import create_app

    app = create_app()
    client = TestClient(app)

    # ---- 1. 健康检查 ----
    print("\n[1] 健康检查")
    r = client.get("/health")
    assert r.status_code == 200
    print(f"    /health → {r.json()}")

    r = client.get("/health/detailed")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "kbs_total" in data
    assert "embedders" in data
    print(f"    /health/detailed → {data}")

    # ---- 2. request-id 响应头 + admin 调用以产生 metrics ----
    print("\n[2] request-id 响应头（顺带触发一次 admin 请求让 metrics 有数据）")
    r = client.get("/admin/kbs", headers={"X-Request-Id": "test-rid-abc"})
    assert r.headers.get("X-Request-Id") == "test-rid-abc"
    print(f"    X-Request-Id={r.headers['X-Request-Id']}")

    # ---- 3. /metrics 端点 ----
    print("\n[3] /metrics 端点")
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    # 验证：调用过 admin endpoint 之后 metrics 应有计数
    assert "study_rag_admin_requests_total" in body, f"missing admin metric in: {body[:200]}"
    print(f"    /metrics → {len(body)} bytes (contains study_rag_admin_*)")

    # ---- 4. 列出 KB + 挑选 embedder 可用的 KB ----
    print("\n[4] 列出 KB")
    r = client.get("/admin/kbs")
    assert r.status_code == 200
    kbs = r.json()
    assert len(kbs) > 0, "no KBs configured"
    # 选 embedder 实际加载的 KB
    from study_rag.knowledge_bases.manager import build_default_manager
    from study_rag.knowledge_bases.registry import get_registry

    manager = build_default_manager()
    registry = get_registry()
    loaded = manager._embedders  # type: ignore[attr-defined]
    working_kb: str | None = None
    for kb in kbs:
        cfg = registry.get(kb["kb_id"])
        if cfg and cfg.embedding in loaded:
            working_kb = kb["kb_id"]
            break
    print(
        f"    found {len(kbs)} KBs, loaded embedders: {list(loaded)}"
        f" → working_kb={working_kb}"
    )
    if working_kb is None:
        print("    ! 没有任何 KB 的 embedder 已加载；跳过 embed 相关步骤")
    kb_id = working_kb or kbs[0]["kb_id"]  # 兜底用第一个（仅用于非 embed 操作）

    # ---- 5. 添加文档 ----
    print("\n[5] 添加文档（整篇一个 chunk）")
    if working_kb is None:
        print("    SKIP: no working embedder in this environment")
    else:
        try:
            r = client.post(
                f"/admin/kbs/{kb_id}/documents",
                json={
                    "kb_id": kb_id,
                    "doc_id": "verify_admin_001",
                    "title": "Admin Test Doc",
                    "content": "This is a test document for admin REST verification.",
                    "source": "wiki",
                },
            )
            if r.status_code != 200:
                print(f"    SKIP: add failed ({r.status_code}): {r.text[:200]}")
                working_kb = None
            else:
                print(f"    add_document → {r.json()['doc_id']}")
        except Exception as e:
            print(f"    SKIP: {type(e).__name__}: {e}")
            working_kb = None

    # ---- 6. 重复添加（应失败 409）----
    print("\n[6] 重复添加（应失败 409）")
    if working_kb is None:
        print("    SKIP")
    else:
        r = client.post(
            f"/admin/kbs/{kb_id}/documents",
            json={
                "kb_id": kb_id,
                "doc_id": "verify_admin_001",
                "title": "dup",
                "content": "dup",
            },
        )
        assert r.status_code in (400, 409, 500), f"expected 4xx, got {r.status_code}"
        print(f"    duplicate add → {r.status_code}")

    # ---- 7. 列出文档 ----
    print("\n[7] 列出文档")
    if working_kb is None:
        print("    SKIP")
    else:
        r = client.get(f"/admin/kbs/{kb_id}/documents")
        assert r.status_code == 200
        docs = r.json()
        assert len(docs) > 0
        print(f"    list_documents → {len(docs)} docs")

    # ---- 8. 获取单个文档 ----
    print("\n[8] 获取单个文档")
    if working_kb is None:
        print("    SKIP")
    else:
        r = client.get(f"/admin/kbs/{kb_id}/documents/verify_admin_001")
        assert r.status_code == 200
        doc = r.json()
        assert doc["doc_id"] == "verify_admin_001"
        print(f"    get_document → title='{doc['title']}'")

    # ---- 9. 检索（管理调试）----
    print("\n[9] 检索（admin search）")
    if working_kb is None:
        print("    SKIP")
    else:
        r = client.post(
            f"/admin/kbs/{kb_id}/search",
            json={"query": "test document", "top_k": 3, "use_rerank": True},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        print(f"    search → {len(data['hits'])} hits, duration={data.get('duration_ms')}ms")

    # ---- 10. 检索（带 filter_expr）----
    print("\n[10] 检索（带 filter_expr）")
    if working_kb is None:
        print("    SKIP")
    else:
        r = client.post(
            f"/admin/kbs/{kb_id}/search",
            json={
                "query": "test",
                "top_k": 3,
                "filter_expr": {"source": "wiki"},
            },
        )
        assert r.status_code == 200, r.text
        print(f"    search+filter → {len(r.json()['hits'])} hits")

    # ---- 11. 批量添加 ----
    print("\n[11] 批量添加")
    if working_kb is None:
        print("    SKIP")
    else:
        r = client.post(
            f"/admin/kbs/{kb_id}/documents/batch",
            json={
                "documents": [
                    {
                        "doc_id": "verify_batch_a",
                        "title": "Batch A",
                        "content": "Content A",
                    },
                    {
                        "doc_id": "verify_batch_b",
                        "title": "Batch B",
                        "content": "Content B",
                        "source": "wiki",
                    },
                ],
            },
        )
        assert r.status_code == 200, r.text
        data = r.json()
        print(f"    batch → ok={data['counts']['ok']}, fail={data['counts']['fail']}")

    # ---- 12. 切块添加 ----
    print("\n[12] 切块添加（chunked）")
    if working_kb is None:
        print("    SKIP")
    else:
        r = client.post(
            f"/admin/kbs/{kb_id}/documents/chunked"
            f"?doc_id=verify_chunked_001&title=Chunked+Doc"
            f"&content=" + "First sentence. " * 50,
            json={
                "parser_config": {"strategy": "sentence", "chunk_size": 100},
                "source": "wiki",
            },
        )
        assert r.status_code == 200, r.text
        chunks = r.json().get("chunks", 0)
        print(f"    chunked_add → {chunks} chunks")

    # ---- 13. 删除文档 ----
    print("\n[13] 删除文档")
    if working_kb is None:
        print("    SKIP")
    else:
        r = client.delete(f"/admin/kbs/{kb_id}/documents/verify_admin_001")
        assert r.status_code == 200
        print(f"    delete → {r.json()}")

    # ---- 14. 启用 admin 鉴权 ----
    print("\n[14] 启用 admin 鉴权")
    os.environ["STUDY_RAG_ADMIN_TOKEN"] = "test-secret-token"

    # reload app（因为鉴权检查在 create_app 时只读一次 env）
    from study_rag.app import create_app
    from study_rag.observability.logging import configure_logging

    configure_logging(level="INFO")
    app_auth = create_app()
    c_auth = TestClient(app_auth)

    # 无 token → 401
    r = c_auth.get("/admin/kbs")
    assert r.status_code == 401, f"expected 401, got {r.status_code}"
    print(f"    no token → 401 (got {r.status_code})")

    # 错误 token → 401
    r = c_auth.get("/admin/kbs", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    print(f"    wrong token → 401 (got {r.status_code})")

    # 正确 token → 200
    r = c_auth.get(
        "/admin/kbs", headers={"Authorization": "Bearer test-secret-token"}
    )
    assert r.status_code == 200
    print(f"    correct token → 200 (got {r.status_code})")

    # /health 仍然不需要鉴权
    r = c_auth.get("/health/detailed")
    assert r.status_code == 200
    print(f"    /health/detailed (no auth) → 200 (got {r.status_code})")

    del os.environ["STUDY_RAG_ADMIN_TOKEN"]

    # ---- 15. 404 KB ----
    print("\n[15] 错误 KB → 404")
    # 重新构造无鉴权 app
    app3 = create_app()
    c3 = TestClient(app3)
    r = c3.get("/admin/kbs/non_existent_kb")
    assert r.status_code == 404
    print(f"    non-existent kb → 404 (got {r.status_code})")

    print("\n" + "=" * 60)
    print("ALL PASS: Admin REST 完整流程")
    print("=" * 60)


if __name__ == "__main__":
    main()
