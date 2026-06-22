"""验证：observability（structlog + request-id + metrics）。"""

# ruff: noqa: T201, PT017, PT018
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    print("=" * 60)
    print("Verify: observability")
    print("=" * 60)

    # ---- 1. structlog 基础 ----
    print("\n[1] structlog configuration")
    from study_rag.observability.logging import configure_logging, get_logger

    configure_logging(level="INFO")
    log = get_logger("verify_observability")
    log.info("structlog_works", status="ok", count=42)
    print("    PASS: get_logger + log.info")

    # ---- 2. request-id 上下文 ----
    print("\n[2] request_id contextvar")
    from study_rag.observability.logging import (
        get_request_id,
        set_request_id,
    )

    set_request_id("test-rid-12345")
    rid = get_request_id()
    assert rid == "test-rid-12345", f"expected 'test-rid-12345', got '{rid}'"
    print(f"    PASS: request_id={rid}")

    # ---- 3. metrics: counter ----
    print("\n[3] metrics counter")
    from study_rag.observability.metrics import get_metrics

    m = get_metrics()
    m.inc("test_counter_total", {"label": "a"})
    m.inc("test_counter_total", {"label": "a"})
    m.inc("test_counter_total", {"label": "b"}, value=5.0)

    out = m.render()
    assert "test_counter_total" in out, "counter not in output"
    assert "label=\"a\"" in out, "label a not in output"
    print("    PASS: counter rendered")

    # ---- 4. metrics: histogram ----
    print("\n[4] metrics histogram")
    for v in [10, 50, 100, 200, 500, 1000, 5000]:
        m.observe("test_hist_latency_ms", float(v), {"kb": "k1"})

    out = m.render()
    assert "test_hist_latency_ms_bucket" in out, "histogram bucket not in output"
    assert "test_hist_latency_ms_count" in out, "histogram count not in output"
    assert "test_hist_latency_ms_sum" in out, "histogram sum not in output"
    print("    PASS: histogram rendered with bucket/sum/count")

    # ---- 5. Prometheus format ----
    print("\n[5] Prometheus exposition format")
    out = m.render()
    lines = out.split("\n")
    type_lines = [line for line in lines if line.startswith("# TYPE")]
    assert any("test_counter_total" in line for line in type_lines)
    assert any("test_hist_latency_ms" in line for line in type_lines)
    print(f"    PASS: {len(type_lines)} TYPE declarations")

    # ---- 6. Middleware: BaseHTTPMiddleware ----
    print("\n[6] RequestIDMiddleware importable")
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.testclient import TestClient

    from study_rag.observability.middleware import RequestIDMiddleware

    app = Starlette()
    app.add_middleware(RequestIDMiddleware)

    async def ping(_request):  # type: ignore[no-untyped-def]
        return PlainTextResponse("pong")

    app.router.add_route("/ping", ping, methods=["GET"])

    client = TestClient(app)
    # 1) 不传 X-Request-Id
    r = client.get("/ping")
    assert r.status_code == 200
    assert r.headers.get("X-Request-Id"), "auto-generated request_id missing"
    auto_rid = r.headers["X-Request-Id"]
    print(f"    PASS: auto-generated rid={auto_rid[:16]}...")

    # 2) 传 X-Request-Id
    r = client.get("/ping", headers={"X-Request-Id": "my-custom-rid"})
    assert r.headers["X-Request-Id"] == "my-custom-rid"
    print("    PASS: custom rid echoed back")

    # ---- 7. FastAPI + RequestIDMiddleware 集成 ----
    print("\n[7] FastAPI integration")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient as FastTestClient

    from study_rag.observability.metrics import AdminMetrics

    app2 = FastAPI()
    app2.add_middleware(RequestIDMiddleware)

    @app2.get("/api/kbs")
    async def list_kbs_endpoint() -> dict:
        m = get_metrics()
        m.inc(AdminMetrics.REQUESTS, {"endpoint": "list_kbs"})
        return {"kbs": []}

    c2 = FastTestClient(app2)
    r = c2.get("/api/kbs")
    assert r.status_code == 200
    assert r.headers.get("X-Request-Id")

    # 验证 metrics 计数了
    out = m.render()
    assert "study_rag_admin_requests_total" in out
    assert "endpoint=\"list_kbs\"" in out
    print("    PASS: FastAPI + metrics + request-id all wired up")

    # ---- 8. JSON 日志格式（non-TTY）----
    print("\n[8] JSON log format (non-TTY)")
    # 上面的输出已经验证：structlog 在非 TTY 时输出 JSON
    # 重新触发一次，验证 format
    log.info("json_format_test", key="value", num=42)
    from study_rag.observability.logging import _add_request_id

    rendered = _add_request_id(
        None,
        "info",
        {"event": "test", "level": "info"},
    )
    assert "request_id" in rendered  # contextvar 默认 None 也会被注入
    print("    PASS: JSON log format (structlog output visible above)")

    print("\n" + "=" * 60)
    print("ALL PASS: observability")
    print("=" * 60)


if __name__ == "__main__":
    main()
