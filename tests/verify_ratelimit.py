"""验证：限流（Token Bucket）。

覆盖：
  1. TokenBucketLimiter 基础行为（容量、refill、cost）
  2. retry_after 计算
  3. reset 行为
  4. 并发安全
  5. 跟 FastAPI 集成：admin_ratelimit_dep 触发 429 + Retry-After
  6. /health/detailed 暴露限流器状态
  7. metrics：ratelimit_total{outcome=allowed/rejected}
  8. 不同 key 互不干扰
"""

# ruff: noqa: T201, PT017
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    print("=" * 60)
    print("Verify: ratelimit (Token Bucket)")
    print("=" * 60)

    # ---- 1. 基础：消费到 0 后被拒绝 ----
    print("\n[1] burst capacity = 5, rate = 1/s")
    from study_rag.observability.ratelimit import TokenBucketLimiter

    rl = TokenBucketLimiter(capacity=5, refill_rate=1.0, name="test1")
    rl.reset()

    # 立即消费 5 个
    for i in range(5):
        assert rl.allow("k1"), f"call {i + 1}/5 should be allowed"
    # 第 6 个应当被拒
    assert not rl.allow("k1"), "6th call should be rejected"
    print("    PASS: first 5 allowed, 6th rejected")

    # ---- 2. retry_after：缺 1 token / 1 token per second = ~1s ----
    print("\n[2] retry_after estimation")
    ra = rl.retry_after("k1")
    assert 0.5 < ra < 1.5, f"retry_after out of range: {ra}"
    print(f"    PASS: retry_after={ra:.3f}s")

    # ---- 3. 等一会儿后又能消费 ----
    print("\n[3] refill after sleep")
    time.sleep(1.2)
    assert rl.allow("k1"), "after sleep, should be allowed again"
    print("    PASS: refilled after 1.2s sleep")

    # ---- 4. 不同 key 互不干扰 ----
    print("\n[4] per-key isolation")
    rl.reset()
    for _ in range(5):
        assert rl.allow("alice")
    # alice 满了，bob 不受影响
    assert not rl.allow("alice")
    assert rl.allow("bob")
    print("    PASS: alice exhausted, bob still has tokens")

    # ---- 5. cost > 1（一次消费多个 token） ----
    print("\n[5] cost=3")
    rl.reset()
    assert rl.allow("k1", cost=3)
    # 桶里剩 2，再要 3 → 拒
    assert not rl.allow("k1", cost=3)
    # 1 个可以
    assert rl.allow("k1", cost=1)
    print("    PASS: cost=3 consumes 3 tokens")

    # ---- 6. reset ----
    print("\n[6] reset()")
    rl.allow("k1")  # 消耗
    rl.reset("k1")
    # 重置后桶满
    for _ in range(5):
        assert rl.allow("k1")
    print("    PASS: reset restores full bucket")

    # ---- 7. stats ----
    print("\n[7] stats()")
    rl.reset()
    rl.allow("ip:1.2.3.4")
    rl.allow("ip:1.2.3.4")
    rl.allow("ip:5.6.7.8")
    s = rl.stats()
    assert s["capacity"] == 5
    assert s["refill_rate"] == 1.0
    assert s["tracked_keys"] == 2
    print(f"    PASS: {s}")

    # ---- 8. metrics: ratelimit_total ----
    print("\n[8] metrics counter")
    from study_rag.observability.metrics import get_metrics

    out = get_metrics().render()
    assert "study_rag_ratelimit_total" in out, "ratelimit metric missing"
    assert 'name="test1"' in out
    assert 'outcome="allowed"' in out
    assert 'outcome="rejected"' in out
    print("    PASS: study_rag_ratelimit_total{..., outcome=allowed/rejected}")

    # ---- 9. FastAPI 集成：429 + Retry-After ----
    print("\n[9] FastAPI admin_ratelimit_dep 触发 429")
    import os

    os.environ.pop("STUDY_RAG_ADMIN_TOKEN", None)

    from fastapi import Depends, FastAPI
    from fastapi.testclient import TestClient

    from study_rag.api.admin import admin_auth_dep, admin_ratelimit_dep
    from study_rag.observability.ratelimit import reset_limiters

    # 强制重置 + 小容量
    reset_limiters()
    import study_rag.observability.ratelimit as rl_mod

    rl_mod._admin_limiter = TokenBucketLimiter(capacity=3, refill_rate=0.5, name="admin")

    app = FastAPI()

    @app.get("/probe")
    async def probe(
        _: str = Depends(admin_auth_dep),
        __: str = Depends(admin_ratelimit_dep),
    ) -> dict[str, str]:
        return {"ok": "yes"}

    client = TestClient(app)

    # 头 3 个 200
    for i in range(3):
        r = client.get("/probe", headers={"X-Forwarded-For": "9.9.9.9"})
        assert r.status_code == 200, f"call {i + 1}: {r.status_code}"
    # 第 4 个 429
    r = client.get("/probe", headers={"X-Forwarded-For": "9.9.9.9"})
    assert r.status_code == 429, f"expected 429, got {r.status_code}: {r.text}"
    assert "Retry-After" in r.headers
    assert int(r.headers["Retry-After"]) >= 1
    print(f"    PASS: 4th call → 429, Retry-After={r.headers['Retry-After']}s")

    # 不同 IP 不受影响
    r = client.get("/probe", headers={"X-Forwarded-For": "8.8.8.8"})
    assert r.status_code == 200, f"different IP should pass: {r.status_code}"
    print("    PASS: different X-Forwarded-For is independent")

    # ---- 10. /health/detailed 暴露 limiter stats ----
    print("\n[10] 限流器 stats() 输出符合 schema")
    from study_rag.observability.ratelimit import get_admin_limiter, get_search_limiter

    reset_limiters()
    admin = get_admin_limiter()
    search = get_search_limiter()
    admin.allow("9.9.9.9")
    admin.allow("9.9.9.9")
    search.allow("kb:rd_frontend")
    a = admin.stats()
    s = search.stats()
    for required in ("name", "capacity", "refill_rate", "tracked_keys"):
        assert required in a, f"missing {required} in admin stats: {a}"
        assert required in s, f"missing {required} in search stats: {s}"
    assert a["name"] == "admin"
    assert s["name"] == "search"
    assert a["tracked_keys"] >= 1
    print(f"    PASS: admin.stats={a}")
    print(f"    PASS: search.stats={s}")

    # ---- 11. 并发安全（1000 个请求同时打） ----
    print("\n[11] concurrent safety (1000 concurrent calls)")
    rl3 = TokenBucketLimiter(capacity=10, refill_rate=0.0, name="concurrent")
    rl3.reset()

    async def one_call() -> bool:
        return rl3.allow("k1")

    async def run_many() -> tuple[int, int]:
        results = await asyncio.gather(*[one_call() for _ in range(1000)])
        ok = sum(1 for r in results if r)
        rej = sum(1 for r in results if not r)
        return ok, rej

    ok, rej = asyncio.run(run_many())
    # refill_rate=0 → 不会补 token；只允许 capacity=10
    assert ok == 10, f"allowed {ok}, expected exactly 10 (capacity)"
    assert rej == 990
    print(f"    PASS: {ok} allowed, {rej} rejected (capacity=10, refill=0)")

    print("\n" + "=" * 60)
    print("ALL PASS: ratelimit")
    print("=" * 60)


if __name__ == "__main__":
    main()
