"""通用 HTTP Reranker：自托管的 OpenAI/Cohere/Jina/TEI 兼容服务都能用。

支持的协议（通过 ``protocol`` 字段切换）：

  - **tei** （默认）：Hugging Face Text Embeddings Inference 的
    ``POST /rerank`` 端点。Docker 一行启动::

        docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:1.5 \\
            --model-id BAAI/bge-reranker-v2-m3

    协议规范：https://github.com/huggingface/text-embeddings-inference

  - **jina**：Jina AI 的 ``POST /api/v1/rerank``（私有部署时同样协议）

  - **cohere_compat**：Cohere 兼容协议（自建代理可以用）

  - **openai**：OpenAI 风格 ``POST /v1/rerank``（部分网关支持，如 LiteLLM）

依赖：httpx>=0.25
安装：pip install httpx   （项目默认已装）

无外部重型依赖（不像 BGE 需要 torch/FlagEmbedding），不依赖任何 provider SDK。
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ..vector_store.base import SearchResult
from .base import RerankerConfig, register_reranker

logger = logging.getLogger(__name__)


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

    def __init__(self, config: RerankerConfig):
        self._config = config
        self._top_k = config.top_k
        self._model_name = config.model_name  # 有些协议需要

        try:
            import httpx  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "HttpReranker requires 'httpx'. Install with: pip install httpx"
            ) from e

        extra = config.extra or {}

        # base_url：服务根地址，例如 http://10.0.0.5:8080
        self._base_url = (extra.get("base_url") or "").rstrip("/")
        if not self._base_url:
            # 兜底：环境变量
            self._base_url = os.environ.get("RERANK_HTTP_BASE_URL", "").rstrip("/")
        if not self._base_url:
            raise ValueError(
                "HttpReranker requires 'base_url' in config.extra "
                "(e.g. http://10.0.0.5:8080) or RERANK_HTTP_BASE_URL env var"
            )

        # 协议
        self._protocol = extra.get("protocol", self.DEFAULT_PROTOCOL)
        if self._protocol not in {"tei", "jina", "cohere_compat", "openai"}:
            raise ValueError(
                f"HttpReranker unsupported protocol: {self._protocol}. "
                "Choose from: tei | jina | cohere_compat | openai"
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

        logger.info(
            "HttpReranker initialized: protocol=%s, base_url=%s, path=%s, model=%s",
            self._protocol,
            self._base_url,
            self._rerank_path,
            self._model_name or "(none)",
        )

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

        import httpx

        url = f"{self._base_url}{self._rerank_path}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    url, json=payload, headers=self._headers()
                )
                resp.raise_for_status()
                body: Any = resp.json()
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

    async def aclose(self) -> None:
        """HTTP 客户端是无状态短连接，无需关闭。"""
        return None
