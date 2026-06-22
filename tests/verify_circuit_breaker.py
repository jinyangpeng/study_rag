"""验证：熔断器（Circuit Breaker）。

覆盖：
  1. CLOSED 状态正常放行
  2. 连续失败达到 threshold → OPEN
  3. OPEN 状态直接拒绝（抛 CircuitOpenError）
  4. open_timeout 后进入 HALF_OPEN
  5. HALF_OPEN 探测成功 → CLOSED
  6. HALF_OPEN 探测失败 → 重新 OPEN
  7. metrics：circuit_breaker_state / calls_total / open_total
  8. OpenAI embedder 走熔断器（mock OpenAI client 失败）
  9. /health/detailed 暴露熔断器状态
"""

# ruff: noqa: T201, PT017
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _flush() -> None:
    sys.stdout.flush()


async def _succeed() -> str:
    return "ok"


async def _fail() -> None:
    raise RuntimeError("downstream failed")


def main() -> None:
    print("=" * 60)
    print("Verify: circuit breaker")
    print("=" * 60)
    _flush()

    # ---- 1. CLOSED → 正常调用 ----
    print("\n[1] CLOSED state: normal calls pass through")
    _flush()
    from study_rag.observability.circuit_breaker import (
        CircuitBreaker,
        CircuitOpenError,
        CircuitState,
    )

    cb = CircuitBreaker(
        name="test1",
        failure_threshold=3,
        open_timeout_s=1.0,
    )
    cb.reset()

    async def scenario() -> str:
        return await cb.call(_succeed)

    result = asyncio.run(scenario())
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED
    print("    PASS: 1 success, state=CLOSED")
    _flush()

    # ---- 2. 连续失败达到 threshold → OPEN ----
    print("\n[2] consecutive failures → OPEN")
    _flush()
    cb.reset()

    async def fail_n() -> None:
        for _ in range(3):
            try:
                await cb.call(_fail)
            except RuntimeError:
                pass

    asyncio.run(fail_n())
    assert cb.state == CircuitState.OPEN, f"expected OPEN, got {cb.state}"
    print("    PASS: 3 consecutive failures → OPEN")
    _flush()

    # ---- 3. OPEN 状态下直接拒绝 ----
    print("\n[3] OPEN state: requests rejected immediately")
    _flush()
    rejected = 0

    async def should_reject() -> None:
        nonlocal rejected
        try:
            await cb.call(_succeed)
        except CircuitOpenError:
            rejected += 1

    asyncio.run(should_reject())
    assert rejected == 1, f"expected 1 reject, got {rejected}"
    print("    PASS: success call rejected when OPEN")
    _flush()

    # ---- 4. retry_after（OPEN 持续时间倒计时） ----
    print("\n[4] retry_after() during OPEN")
    _flush()
    ra = cb.retry_after()
    assert 0 < ra <= 1.0, f"retry_after out of range: {ra}"
    print(f"    PASS: retry_after={ra:.3f}s (open_timeout=1s)")
    _flush()

    # ---- 5. 等 open_timeout → HALF_OPEN → 探测成功 → CLOSED ----
    print("\n[5] HALF_OPEN transition + success → CLOSED")
    _flush()
    time.sleep(1.1)  # 等过 open_timeout

    async def probe_success() -> str:
        return await cb.call(_succeed)

    result = asyncio.run(probe_success())
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED, f"expected CLOSED, got {cb.state}"
    print("    PASS: timeout → HALF_OPEN → success → CLOSED")
    _flush()

    # ---- 6. HALF_OPEN 探测失败 → 重新 OPEN ----
    print("\n[6] HALF_OPEN probe failure → re-OPEN")
    _flush()
    cb.reset()

    async def fail_then_probe_fail() -> None:
        # 先触发 OPEN
        for _ in range(3):
            try:
                await cb.call(_fail)
            except RuntimeError:
                pass
        # 等到 HALF_OPEN
        await asyncio.sleep(1.1)
        # 探测失败
        try:
            await cb.call(_fail)
        except RuntimeError:
            pass

    asyncio.run(fail_then_probe_fail())
    assert cb.state == CircuitState.OPEN, f"expected OPEN after probe fail, got {cb.state}"
    print("    PASS: probe failure re-OPENs the breaker")
    _flush()

    # ---- 7. metrics ----
    print("\n[7] metrics output")
    from study_rag.observability.metrics import get_metrics

    out = get_metrics().render()
    assert "study_rag_circuit_breaker_calls_total" in out
    assert "study_rag_circuit_breaker_open_total" in out
    assert 'name="test1"' in out
    assert 'outcome="rejected"' in out or 'outcome="error"' in out
    print("    PASS: circuit_breaker_calls_total + open_total present")

    # ---- 8. stats ----
    print("\n[8] stats()")
    cb.reset()
    s = cb.stats()
    assert s["name"] == "test1"
    assert s["state"] == "closed"
    assert "failure_count" in s
    assert "retry_after_s" in s
    print(f"    PASS: {s}")

    # ---- 9. OpenAI embedder 走熔断器（mock client） ----
    print("\n[9] OpenAIEmbedder._embed_batch uses breaker")
    from unittest.mock import AsyncMock, MagicMock

    from study_rag.capabilities.embedding.base import EmbeddingConfig
    from study_rag.capabilities.embedding.impls_openai import OpenAIEmbedder
    from study_rag.observability.circuit_breaker import reset_breakers

    reset_breakers()
    cb_openai = None

    # 构造 embedder 不走真实 OpenAI，注入 mock client
    cfg = EmbeddingConfig(
        provider="openai",
        model_name="text-embedding-3-small",
        dimension=4,
    )
    embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
    embedder.dimension = 4
    embedder._config = cfg
    embedder._model = "text-embedding-3-small"
    embedder._batch_size = 10
    embedder._encoding_format = "float"
    embedder._client = MagicMock()

    # mock response
    mock_item = MagicMock()
    mock_item.embedding = [0.1, 0.2, 0.3, 0.4]
    mock_response = MagicMock()
    mock_response.data = [mock_item]
    embedder._client.embeddings.create = AsyncMock(return_value=mock_response)

    async def call_openai() -> list[list[float]]:
        return await embedder._embed_batch(["hello"])

    vectors = asyncio.run(call_openai())
    assert vectors == [[0.1, 0.2, 0.3, 0.4]]
    # 拿熔断器单例，state 应该是 CLOSED（成功调用后）
    from study_rag.observability.circuit_breaker import get_openai_breaker

    cb_openai = get_openai_breaker()
    assert cb_openai.state == CircuitState.CLOSED
    print("    PASS: OpenAIEmbedder._embed_batch goes through breaker")

    # ---- 10. embedder 失败 → breaker 累计失败 ----
    print("\n[10] repeated failures trip the breaker")
    reset_breakers()
    # 强制把 threshold 改小
    from study_rag.observability import circuit_breaker as cb_mod

    cb_mod._openai_cb = CircuitBreaker(
        name="openai_embed",
        failure_threshold=2,
        open_timeout_s=10.0,
    )
    embedder._client.embeddings.create = AsyncMock(side_effect=RuntimeError("API down"))

    async def call_fail() -> None:
        try:
            await embedder._embed_batch(["x"])
        except (RuntimeError, CircuitOpenError):
            pass

    async def fail_twice() -> None:
        for _ in range(2):
            await call_fail()

    asyncio.run(fail_twice())
    # 第 3 次应当被熔断（rejected）而不是 RuntimeError
    rejected2 = 0

    async def call_should_reject() -> None:
        nonlocal rejected2
        try:
            await embedder._embed_batch(["x"])
        except CircuitOpenError:
            rejected2 += 1
        except RuntimeError:
            pass

    asyncio.run(call_should_reject())
    assert rejected2 == 1, f"expected 1 CircuitOpenError, got {rejected2}"
    assert get_openai_breaker().state == CircuitState.OPEN
    print("    PASS: 2 failures → breaker OPEN, 3rd call rejected with CircuitOpenError")

    # ---- 11. /health/detailed 暴露熔断器状态（只验证 stats schema） ----
    print("\n[11] circuit_breaker.stats() schema 完整")
    from study_rag.observability.circuit_breaker import (
        get_openai_breaker,
        get_search_breaker,
    )

    reset_breakers()
    openai_cb = get_openai_breaker()
    search_cb = get_search_breaker()
    s1 = openai_cb.stats()
    s2 = search_cb.stats()
    for required in ("name", "state", "failure_count", "retry_after_s"):
        assert required in s1, f"missing {required} in openai stats: {s1}"
        assert required in s2, f"missing {required} in search stats: {s2}"
    assert s1["name"] == "openai_embed"
    assert s2["name"] == "search"
    assert s1["state"] == "closed"
    assert s2["state"] == "closed"
    print(f"    PASS: openai={s1}")
    print(f"    PASS: search={s2}")

    print("\n" + "=" * 60)
    print("ALL PASS: circuit_breaker")
    print("=" * 60)


if __name__ == "__main__":
    main()
