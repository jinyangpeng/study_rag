"""HttpReranker（TEI / Jina / Cohere / OpenAI 兼容协议）单元测试。

用 ``httpx.MockTransport`` 拦截请求，无需真实 reranker 服务即可验证：
  - 协议请求体构造与响应解析
  - top_k 截断与原索引映射
  - 重试（5xx / 网络错误）与不重试（4xx）
  - 健康检查
  - 配置校验（base_url 缺失 / 非法 protocol）
"""

from __future__ import annotations

import httpx
import pytest

from study_rag.capabilities.reranker.base import RerankerConfig
from study_rag.capabilities.reranker.impls_http import HttpReranker
from study_rag.capabilities.vector_store.base import SearchResult


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #


def _results(n: int = 3) -> list[SearchResult]:
    """构造 n 个测试检索结果（分数倒序无意义，reranker 会重排）。"""
    return [
        SearchResult(id=f"doc-{i}", text=f"文本片段 {i}", score=0.1 * i)
        for i in range(n)
    ]


def _make_reranker(
    handler,
    *,
    protocol: str = "tei",
    base_url: str = "http://test:8080",
    model_name: str = "BAAI/bge-reranker-v2-m3",
    top_k: int = 5,
    max_retries: int = 3,
    retry_backoff: float = 0.0,  # 测试中不真实等待
    extra: dict | None = None,
) -> HttpReranker:
    """构造一个注入了 MockTransport 的 HttpReranker。"""
    ex = {"base_url": base_url, "max_retries": max_retries, "retry_backoff": retry_backoff}
    if extra:
        ex.update(extra)
    cfg = RerankerConfig(
        provider="http",
        protocol=protocol,
        model_name=model_name,
        top_k=top_k,
        extra=ex,
    )
    r = HttpReranker(cfg)
    # 注入 mock transport，拦截所有出站请求
    transport = httpx.MockTransport(handler)
    r._client = httpx.AsyncClient(transport=transport)
    return r


# --------------------------------------------------------------------------- #
# 配置校验
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_protocol_is_top_level_field(self):
        """protocol 应作为顶层字段被 Pydantic 接收（修复历史 bug）。"""
        cfg = RerankerConfig(provider="http", protocol="tei")
        assert cfg.protocol == "tei"

    def test_protocol_defaults_empty(self):
        cfg = RerankerConfig(provider="http")
        assert cfg.protocol == ""

    def test_missing_base_url_raises(self):
        cfg = RerankerConfig(provider="http", protocol="tei", extra={})
        with pytest.raises(ValueError, match="base_url"):
            HttpReranker(cfg)

    def test_invalid_protocol_raises(self):
        cfg = RerankerConfig(
            provider="http",
            protocol="unknown",
            extra={"base_url": "http://test:8080"},
        )
        with pytest.raises(ValueError, match="unsupported protocol"):
            HttpReranker(cfg)

    def test_extra_protocol_backward_compat(self):
        """protocol 仍可写在 extra（向后兼容）。"""
        cfg = RerankerConfig(
            provider="http",
            protocol="",  # 顶层为空
            extra={"base_url": "http://test:8080", "protocol": "jina"},
        )
        r = HttpReranker(cfg)
        assert r._protocol == "jina"


# --------------------------------------------------------------------------- #
# TEI 协议：请求构造 / 响应解析 / top_k
# --------------------------------------------------------------------------- #


class TestTEIRerank:
    async def test_tei_payload_and_response(self):
        """TEI 协议：请求体含 query/texts/truncate_input_tokens，响应按 score 倒序。"""
        captured: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["url"] = str(req.url)
            captured["body"] = req.read()
            # TEI 返回 [{index, score}, ...]，故意打乱顺序验证排序
            return httpx.Response(
                200,
                json=[
                    {"index": 2, "score": 0.91},
                    {"index": 0, "score": 0.42},
                    {"index": 1, "score": 0.77},
                ],
            )

        r = _make_reranker(handler, top_k=5)
        out = await r.rerank("BGE 原理", _results(3))

        # 端点路径
        assert captured["url"] == "http://test:8080/rerank"
        # 请求体字段（TEI 规范）
        import json

        payload = json.loads(captured["body"])
        assert payload["query"] == "BGE 原理"
        assert payload["texts"] == ["文本片段 0", "文本片段 1", "文本片段 2"]
        assert payload["truncate_input_tokens"] == 512
        assert payload["return_documents"] is False
        assert payload["model"] == "BAAI/bge-reranker-v2-m3"

        # 响应按 score 倒序：doc-2(0.91) > doc-1(0.77) > doc-0(0.42)
        assert [o.id for o in out] == ["doc-2", "doc-1", "doc-0"]
        assert [o.score for o in out] == [0.91, 0.77, 0.42]
        # 元数据保留
        assert out[0].text == "文本片段 2"

    async def test_top_k_truncation(self):
        """top_k 截断：返回数量不超过 top_k。"""

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=[
                    {"index": 0, "score": 0.9},
                    {"index": 1, "score": 0.8},
                    {"index": 2, "score": 0.7},
                ],
            )

        r = _make_reranker(handler, top_k=2)
        out = await r.rerank("q", _results(3))
        assert len(out) == 2
        assert out[0].id == "doc-0"

    async def test_top_k_override(self):
        """运行时 top_k 参数覆盖配置。"""

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=[
                    {"index": 0, "score": 0.9},
                    {"index": 1, "score": 0.8},
                    {"index": 2, "score": 0.7},
                ],
            )

        r = _make_reranker(handler, top_k=5)
        out = await r.rerank("q", _results(3), top_k=1)
        assert len(out) == 1

    async def test_empty_results(self):
        """空结果直接返回，不发起请求。"""
        called = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            called["n"] += 1
            return httpx.Response(200, json=[])

        r = _make_reranker(handler)
        out = await r.rerank("q", [])
        assert out == []
        assert called["n"] == 0

    async def test_out_of_range_index_ignored(self):
        """越界 index 被忽略（防御服务返回脏数据）。"""

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=[
                    {"index": 0, "score": 0.9},
                    {"index": 99, "score": 0.95},  # 越界
                    {"index": 1, "score": 0.8},
                ],
            )

        r = _make_reranker(handler, top_k=5)
        out = await r.rerank("q", _results(2))
        assert {o.id for o in out} == {"doc-0", "doc-1"}


# --------------------------------------------------------------------------- #
# 重试机制
# --------------------------------------------------------------------------- #


class TestRetry:
    async def test_retry_on_5xx_then_success(self):
        """5xx 重试，最终成功。"""
        calls = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(503)  # 服务端临时不可用
            return httpx.Response(
                200, json=[{"index": 0, "score": 0.9}]
            )

        r = _make_reranker(handler, max_retries=3, retry_backoff=0.0)
        out = await r.rerank("q", _results(1))
        assert len(out) == 1
        assert calls["n"] == 3  # 2 次失败 + 1 次成功

    async def test_retry_exhausted_raises(self):
        """重试耗尽后抛出 HTTPStatusError。"""
        calls = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(500)

        r = _make_reranker(handler, max_retries=2, retry_backoff=0.0)
        with pytest.raises(httpx.HTTPError):
            await r.rerank("q", _results(1))
        # 1 次初试 + 2 次重试 = 3 次
        assert calls["n"] == 3

    async def test_no_retry_on_4xx(self):
        """4xx（非 429）客户端错误不重试，立即抛出。"""
        calls = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(400, json={"error": "bad request"})

        r = _make_reranker(handler, max_retries=3, retry_backoff=0.0)
        with pytest.raises(httpx.HTTPError):
            await r.rerank("q", _results(1))
        assert calls["n"] == 1

    async def test_retry_on_429(self):
        """429 限流可重试。"""
        calls = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429)
            return httpx.Response(200, json=[{"index": 0, "score": 0.5}])

        r = _make_reranker(handler, max_retries=2, retry_backoff=0.0)
        out = await r.rerank("q", _results(1))
        assert len(out) == 1
        assert calls["n"] == 2


# --------------------------------------------------------------------------- #
# 健康检查
# --------------------------------------------------------------------------- #


class TestHealthCheck:
    async def test_health_check_ok(self):
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.url.path == "/health"
            return httpx.Response(200)

        r = _make_reranker(handler)
        assert await r.health_check() is True

    async def test_health_check_unhealthy(self):
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(503)

        r = _make_reranker(handler)
        assert await r.health_check() is False

    async def test_health_check_network_error(self):
        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=req)

        r = _make_reranker(handler)
        # 网络异常不抛出，返回 False
        assert await r.health_check() is False


# --------------------------------------------------------------------------- #
# 其它协议（回归保障）
# --------------------------------------------------------------------------- #


class TestOtherProtocols:
    async def test_jina_protocol(self):
        captured: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["url"] = str(req.url)
            return httpx.Response(
                200, json=[{"index": 0, "score": 0.88}]
            )

        r = _make_reranker(handler, protocol="jina")
        out = await r.rerank("q", _results(1))
        assert captured["url"] == "http://test:8080/api/v1/rerank"
        assert len(out) == 1

    async def test_cohere_compat_protocol(self):
        captured: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["body"] = req.read()
            return httpx.Response(
                200, json={"results": [{"index": 0, "relevance_score": 0.93}]}
            )

        r = _make_reranker(handler, protocol="cohere_compat", model_name="rerank-multilingual-v3.0")
        out = await r.rerank("q", _results(1))
        import json

        payload = json.loads(captured["body"])
        assert payload["model"] == "rerank-multilingual-v3.0"
        assert payload["documents"] == ["文本片段 0"]
        assert out[0].score == 0.93

    async def test_openai_protocol(self):
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"data": [{"index": 0, "score": 0.66}]}
            )

        r = _make_reranker(handler, protocol="openai")
        out = await r.rerank("q", _results(1))
        assert len(out) == 1
        assert out[0].score == 0.66


# --------------------------------------------------------------------------- #
# 生命周期
# --------------------------------------------------------------------------- #


class TestLifecycle:
    async def test_aclose_releases_client(self):
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[])

        r = _make_reranker(handler)
        assert r._client is not None
        await r.aclose()
        assert r._client is None
