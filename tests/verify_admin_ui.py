"""FastAPI 集成：管理 UI 挂载点 + 路由行为。

覆盖：
  1. /admin/ui → 重定向到 /admin/ui/
  2. /admin/ui/ → 返回 SPA 入口（未构建时是 fallback HTML）
  3. SPA fallback：/admin/ui/kbs 也返回入口
  4. /admin/ui/assets/* 在未构建时不会 500
  5. 不影响 OpenAPI spec（路由不在 schema 里）
  6. 鉴权不影响 UI 路由（/admin/ui/* 不需要 token）
"""

# ruff: noqa: T201, PT017
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    print("=" * 60)
    print("Verify: admin UI mount")
    print("=" * 60)

    import os

    os.environ.pop("STUDY_RAG_ADMIN_TOKEN", None)

    from fastapi.testclient import TestClient

    from study_rag.app import create_app

    app = create_app()
    client = TestClient(app)

    # ---- 1. /admin/ui/ → 200 HTML ----
    print("\n[1] /admin/ui/ → 200 HTML")
    r = client.get("/admin/ui/")
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:200]}"
    assert "text/html" in r.headers.get("content-type", "")
    body = r.text
    assert "study_rag" in body or "study-rag" in body.lower()
    print(f"    PASS: returned {len(body)} bytes HTML")

    # ---- 1b. /admin/ui (无尾斜杠) → 200 HTML（被规范化） ----
    print("\n[1b] /admin/ui (no trailing slash) → 200 HTML")
    r = client.get("/admin/ui")
    assert r.status_code == 200, f"expected 200, got {r.status_code}"
    assert "text/html" in r.headers.get("content-type", "")
    print("    PASS")

    # ---- 3. SPA fallback：/admin/ui/kbs 也返回入口 ----
    print("\n[3] SPA fallback: /admin/ui/kbs → index.html")
    r = client.get("/admin/ui/kbs")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    print("    PASS")

    # /admin/ui/dashboard
    r = client.get("/admin/ui/dashboard")
    assert r.status_code == 200
    print("    PASS: /admin/ui/dashboard also serves HTML")

    # ---- 4. 路径穿越不 500 ----
    print("\n[4] path traversal: /admin/ui/../etc/passwd")
    # TestClient 会规范化路径，所以直接 404 而不是 500
    r = client.get("/admin/ui/../etc/passwd")
    assert r.status_code in (200, 400, 404), f"got {r.status_code}"
    print(f"    PASS: status={r.status_code} (no 500)")

    # ---- 5. 不影响 OpenAPI spec ----
    print("\n[5] /admin/ui/* not in OpenAPI spec")
    spec = client.get("/openapi.json").json()
    for path in spec["paths"]:
        assert not path.startswith("/admin/ui"), f"{path} leaked into OpenAPI"
    print("    PASS: no /admin/ui/* paths in OpenAPI")

    # ---- 6. 不需要鉴权 ----
    print("\n[6] /admin/ui/ works without auth token")
    # 不带 Authorization 头也应该能访问（避免 UI 跟 token 死锁）
    r = client.get("/admin/ui/")
    assert r.status_code == 200
    print("    PASS: no Authorization required for UI mount")

    # ---- 7. /docs 仍然能访问 ----
    print("\n[7] /docs (Swagger UI) still accessible")
    r = client.get("/docs")
    assert r.status_code == 200
    print("    PASS")

    # ---- 8. /admin/kbs 仍然能访问（API 不受影响） ----
    print("\n[8] /admin/kbs API still works")
    r = client.get("/admin/kbs")
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:200]}"
    assert isinstance(r.json(), list)
    print(f"    PASS: returned {len(r.json())} KBs")

    # ---- 9. 静态资源：/admin/ui/assets/index-*.js ----
    print("\n[9] /admin/ui/assets/*.js (build artifacts)")
    r = client.get("/admin/ui/")
    assert r.status_code == 200
    html = r.text
    # 从 index.html 里找到第一个 <script src="..."> 或 <link href="...">
    import re

    assets = re.findall(r'(?:src|href)="(/admin/ui/assets/[^"]+)"', html)
    if not assets:
        print("    SKIP: no built assets found (frontend not built?)")
    else:
        for asset in assets[:3]:
            r = client.get(asset)
            assert r.status_code == 200, f"{asset}: {r.status_code}"
            size = len(r.content)
            print(f"    PASS: {asset} → {size} bytes")
        # 至少一个 JS chunk + 一个 CSS chunk
        assert any(".js" in a for a in assets), "no JS asset found"
        assert any(".css" in a for a in assets), "no CSS asset found"
        print("    PASS: JS + CSS assets present")

    print("\n" + "=" * 60)
    print("ALL PASS: admin UI mount")
    print("=" * 60)


if __name__ == "__main__":
    main()
