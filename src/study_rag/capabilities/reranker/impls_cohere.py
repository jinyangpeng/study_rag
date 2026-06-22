"""Cohere Reranker：通过 Cohere API 重排。

支持的模型（截至 2024）：
  - rerank-v3.5         (多语言, 推荐, 100+ 语言)
  - rerank-english-v3.0 (英文, 速度更快)
  - rerank-multilingual-v3.0

依赖：cohere>=5.0.0
安装：pip install study-rag[reranker-cohere]

API Key:
  - 控制台: https://dashboard.cohere.com/api-keys
  - 配置: extra.api_key: ${COHERE_API_KEY}  或环境变量 COHERE_API_KEY

注：
  - Cohere 的 relevance_score 已经是 0~1，越大越相关，可直接用。
  - 返回结果已按相关性倒序，top_n 控制返回数量。
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ..vector_store.base import SearchResult
from .base import RerankerConfig, register_reranker

logger = logging.getLogger(__name__)


def _get_cohere_async_client():
    """懒加载 cohere.AsyncClient。"""
    try:
        from cohere import AsyncClient  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "CohereReranker requires 'cohere'. "
            "Install with: pip install study-rag[reranker-cohere]"
        ) from e
    return AsyncClient


@register_reranker("cohere")
class CohereReranker:
    """Cohere Reranker（基于 AsyncClient）。

    适合：
      - 多语言重排（v3.5 支持 100+ 语言）
      - 不想本地部署大模型
      - 中小规模（API 限流：1000 req/min on default tier）
    """

    DEFAULT_MODEL = "rerank-v3.5"

    def __init__(self, config: RerankerConfig):
        self._config = config
        self._top_k = config.top_k

        AsyncClient = _get_cohere_async_client()  # noqa: N806

        extra = config.extra or {}
        api_key = extra.get("api_key")
        if not api_key:
            api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "CohereReranker requires 'api_key' in config.extra "
                "or COHERE_API_KEY env var"
            )

        timeout = extra.get("timeout", 60.0)
        max_retries = extra.get("max_retries", 3)
        base_url = extra.get("base_url")  # 私有部署 / 代理

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = AsyncClient(**client_kwargs)
        self._model = config.model_name or self.DEFAULT_MODEL

        logger.info(
            "CohereReranker initialized: model=%s, top_k=%d",
            self._model,
            self._top_k,
        )

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """调用 Cohere rerank API 重排。

        Cohere 接口直接返回排序后的结果 + relevance_score。
        """
        if not results:
            return []

        k = top_k if top_k is not None else self._top_k

        # 准备 documents 列表（保持与 results 的索引对应）
        documents = [r.text for r in results]

        try:
            response = await self._client.rerank(
                model=self._model,
                query=query,
                documents=documents,
                top_n=k,
                return_documents=False,  # 我们已经有 text 了，省点流量
            )
        except Exception as e:
            logger.error("Cohere rerank failed: %s", e)
            raise

        # response.results: list[RerankResult]，按相关性倒序
        # 每个含 index (原始 documents 中的索引) 和 relevance_score
        out: list[SearchResult] = []
        for r in response.results:
            original = results[r.index]
            out.append(
                SearchResult(
                    id=original.id,
                    text=original.text,
                    score=float(r.relevance_score),
                    metadata=original.metadata,
                )
            )

        logger.debug(
            "Cohere rerank query=%r model=%s top_n=%d -> %d results",
            query[:30],
            self._model,
            k,
            len(out),
        )
        return out

    async def aclose(self) -> None:
        """关闭 HTTP 连接池。"""
        try:
            await self._client.close()
        except Exception as e:
            logger.warning("Error closing Cohere client: %s", e)
