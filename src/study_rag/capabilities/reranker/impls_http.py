"""通用 HTTP Reranker：自托管的 OpenAI/Cohere/Jina/TEI 兼容服务都能用。

支持的协议（通过 ``protocol`` 字段切换，写在配置顶层）：

  - **tei** （默认）：Hugging Face Text Embeddings Inference 的
    ``POST /rerank`` 端点。Docker 一行启动::

        docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:1.5 \\
            --model-id BAAI/bge-reranker-v2-m3

    协议规范：https://github.com/huggingface/text-embeddings-inference

  - **jina**：Jina AI 的 ``POST /api/v1/rerank``（私有部署时同样协议）

  - **cohere_compat**：Cohere 兼容协议（自建代理可以用）

  - **openai**：OpenAI 风格 ``POST /v1/rerank``（部分网关支持，如 LiteLLM）

企业级特性：
  - 连接池复用（实例级 ``httpx.AsyncClient``，避免每次请求重建 TLS）
  - 指数退避重试（仅对网络错误 / 5xx 重试，4xx 直接失败）
  - 健康检查 ``health_check()``（探活 reranker 服务）
  - 优雅关闭 ``aclose()``

依赖：httpx>=0.25
安装：pip install httpx   （项目默认已装）

无外部重型依赖（不像 BGE 需要 torch/FlagEmbedding），不依赖任何 provider SDK。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

from ..vector_store.base import SearchResult
from .base import RerankerConfig, register_reranker

logger = logging.getLogger(__name__)

# 可重试的 HTTP 状态码（5xx 服务端错误 / 429 限流）
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_VALID_PROTOCOLS = {"tei", "jina", "cohere_compat", "openai"}


@register_reranker("http")
class HttpReranker:
    """自托管 HTTP Reranker。

    适合：
      - 不想装 torch/FlagEmbedding
      - 已有 BGE/Cohere 重排服务在公司内网
      - 想 GPU 资源池化（多个 study_rag 实例共享一个 rerank 服务）

    协议示例（tei）::

        POST /rerank
        Content-Type: application/json
        {
          "query": "BGE 的原理",
          "texts": ["BGE 是...", "其他文本..."],
          "truncate_input_tokens": 512,
          "return_documents": false
        }

        Response:
        [
          {"index": 0, "score": 0.94},
          {"index": 3, "score": 0.72}
        ]
    """

    DEFAULT_PROTOCOL = "tei"
    DEFAULT_TIMEOUT = 30.0
    DEFAULT_TRUNCATE_TOKENS = 512
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_BACKOFF = 0.5

    def __init__(self, config: RerankerConfig):
        self._config = config
        self._top_k = config.top_k
        self._model_name = config.model_name  # 有些协议需要

        extra = config.extra or {}

        # protocol：优先顶层字段，其次 extra（向后兼容），最后默认 tei
        self._protocol = config.protocol or extra.get("protocol", self.DEFAULT_PROTOCOL)
        if self._protocol not in _VALID_PROTOCOLS:
            raise ValueError(
                f"HttpReranker unsupported protocol: {self._protocol}. "
                f"Choose from: {' | '.join(sorted(_VALID_PROTOCOLS))}"
            )

        # base_url：服务根地址，例如 http://127.0.0.1:8080
        self._base_url = (extra.get("base_url") or "").rstrip("/")
        if not self._base_url:
            # 兜底：环境变量
            self._base_url = os.environ.get("RERANK_HTTP_BASE_URL", "").rstrip("/")
        if not self._base_url:
            raise ValueError(
                "HttpReranker requires 'base_url' in config.extra "
                "(e.g. http://127.0.0.1:8080) or RERANK_HTTP_BASE_URL env var"
            )

        # 鉴权（私有部署时常带 Bearer）
        self._api_key = extra.get("api_key") or os.environ.get(
            "RERANK_HTTP_API_KEY", ""
        )

        # 其它
        self._timeout = float(extra.get("timeout", self.DEFAULT_TIMEOUT))
        self._truncate_tokens = int(
            extra.get("truncate_input_tokens", self.DEFAULT_TRUNCATE_TOKENS)
        )
        self._rerank_path = extra.get("path", self._default_path())
        self._max_retries = int(extra.get("max_retries", self.DEFAULT_MAX_RETRIES))
        self._retry_backoff = float(extra.get("retry_backoff", self.DEFAULT_RETRY_BACKOFF))

        # 实例级连接池（lazy 初始化，事件循环就绪后再创建）
        self._client: httpx.AsyncClient | None = None

        logger.info(
            "HttpReranker initialized: protocol=%s, base_url=%s, path=%s, model=%s, "
            "max_retries=%d",
            self._protocol,
            self._base_url,
            self._rerank_path,
            self._model_name or "(none)",
            self._max_retries,
        )

    # ------------------------------------------------------------------ #
    # 协议相关
    # ------------------------------------------------------------------ #

    def _default_path(self) -> str:
        if self._protocol == "tei":
            return "/rerank"
        if self._protocol == "jina":
            return "/api/v1/rerank"
        if self._protocol == "openai":
            return "/v1/rerank"
        return "/rerank"  # cohere_compat

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _build_payload(self, query: str, texts: list[str]) -> dict[str, Any]:
        """按协议构造 request body。"""
        if self._protocol in {"tei", "jina"}:
            payload: dict[str, Any] = {
                "query": query,
                "texts": texts,
                "truncate_input_tokens": self._truncate_tokens,
                "return_documents": False,
            }
            if self._model_name:
                payload["model"] = self._model_name
            return payload
        if self._protocol == "cohere_compat":
            return {
                "model": self._model_name or "rerank-multilingual-v3.0",
                "query": query,
                "documents": texts,
                "top_n": len(texts),
                "return_documents": False,
            }
        # openai
        return {
            "model": self._model_name or "rerank",
            "query": query,
            "documents": texts,
            "top_n": len(texts),
        }

    def _parse_response(
        self, body: Any, n_results: int
    ) -> list[tuple[int, float]]:
        """把服务返回的 body 解析成 [(orig_index, score), ...] 倒序。"""
        if self._protocol in {"tei", "jina"}:
            # TEI/Jina: [{ "index": int, "score": float }, ...]
            items = body if isinstance(body, list) else body.get("results", [])
            parsed = [(int(it["index"]), float(it["score"])) for it in items]
        elif self._protocol == "cohere_compat":
            # Cohere 协议: { "results": [ { "index": int, "relevance_score": float } ] }
            items = body.get("results", [])
            parsed = [(int(it["index"]), float(it["relevance_score"])) for it in items]
        else:  # openai
            items = body.get("data", body.get("results", []))
            parsed = [(int(it["index"]), float(it["score"])) for it in items]
        # 按 score 倒序
        parsed.sort(key=lambda x: x[1], reverse=True)
        return parsed

    # ------------------------------------------------------------------ #
    # 连接池 / 生命周期
    # ------------------------------------------------------------------ #

    def _get_client(self) -> httpx.AsyncClient:
        """复用实例级 AsyncClient（连接池），避免每次请求重建 TLS。"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers=self._headers(),
                limits=httpx.Limits(max_keepalive_connections=10),
            )
        return self._client

    async def health_check(self) -> bool:
        """探活 reranker 服务。

        TEI / Infinity 暴露 ``GET /health``，其它协议回退到对根路径发 HEAD。
        不抛异常，返回布尔。
        """
        path = "/health" if self._protocol == "tei" else "/"
        url = f"{self._base_url}{path}"
        try:
            client = self._get_client()
            resp = await client.get(url, timeout=min(self._timeout, 5.0))
            ok = resp.status_code < 500
            if not ok:
                logger.warning(
                    "HttpReranker health_check unhealthy: url=%s status=%s",
                    url,
                    resp.status_code,
                )
            return ok
        except Exception as e:  # noqa: BLE001
            logger.warning("HttpReranker health_check failed: url=%s err=%s", url, e)
            return False

    async def aclose(self) -> None:
        """关闭复用的连接池。应用退出时调用。"""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    # ------------------------------------------------------------------ #
    # 核心：rerank
    # ------------------------------------------------------------------ #

    async def _post_with_retry(
        self, url: str, payload: dict[str, Any]
    ) -> Any:
        """带指数退避的重试 POST。

        - 网络错误（ConnectError/ReadTimeout 等）：重试
        - 5xx / 429：重试
        - 4xx（非 429）：直接抛出（客户端错误，重试无意义）
        """
        last_exc: Exception | None = None
        client = self._get_client()
        for attempt in range(self._max_retries + 1):
            try:
                resp = await client.post(url, json=payload, headers=self._headers())
                if resp.status_code < 400:
                    return resp.json()
                if resp.status_code in _RETRYABLE_STATUS and attempt < self._max_retries:
                    logger.warning(
                        "HttpReranker retryable status: url=%s status=%s attempt=%d/%d",
                        url,
                        resp.status_code,
                        attempt + 1,
                        self._max_retries,
                    )
                    last_exc = httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}", request=resp.request, response=resp
                    )
                else:
                    # 4xx 非 429，或重试耗尽
                    resp.raise_for_status()
                    return resp.json()
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                last_exc = e
                if attempt >= self._max_retries:
                    break
                logger.warning(
                    "HttpReranker network error: url=%s err=%s attempt=%d/%d",
                    url,
                    e,
                    attempt + 1,
                    self._max_retries,
                )
            # 指数退避：0.5s, 1.0s, 2.0s, ...
            await asyncio.sleep(self._retry_backoff * (2**attempt))
        # 重试耗尽
        logger.error(
            "HttpReranker request failed after %d retries: url=%s err=%s",
            self._max_retries,
            url,
            last_exc,
        )
        raise last_exc if last_exc else httpx.HTTPError(f"request failed: {url}")

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """调用 HTTP rerank 服务重排。"""
        if not results:
            return []
        k = top_k if top_k is not None else self._top_k

        texts = [r.text for r in results]
        payload = self._build_payload(query, texts)

        url = f"{self._base_url}{self._rerank_path}"
        try:
            body: Any = await self._post_with_retry(url, payload)
        except httpx.HTTPError as e:
            logger.error("HttpReranker request failed: url=%s err=%s", url, e)
            raise

        pairs = self._parse_response(body, n_results=len(results))

        out: list[SearchResult] = []
        for orig_idx, score in pairs[:k]:
            if orig_idx < 0 or orig_idx >= len(results):
                continue
            original = results[orig_idx]
            out.append(
                SearchResult(
                    id=original.id,
                    text=original.text,
                    score=score,
                    metadata=original.metadata,
                )
            )
        logger.debug(
            "HttpReranker query=%r protocol=%s -> %d/%d",
            query[:30],
            self._protocol,
            len(out),
            len(results),
        )
        return out
