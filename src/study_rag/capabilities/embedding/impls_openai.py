"""OpenAI Embedder：通过 OpenAI 兼容 API 生成 embedding。

支持：
  - OpenAI 官方（text-embedding-3-small / 3-large / ada-002）
  - 任何 OpenAI 兼容 API（通过 base_url 配置）：
      - Azure OpenAI
      - 通义千问 dashscope
      - 智谱 AI
      - LocalAI / vLLM / Ollama（开启 OpenAI 兼容）

依赖：openai>=1.40.0
安装：pip install study-rag[embedding-openai]

熔断:
  - 每次 OpenAI API 调用都通过 get_openai_breaker() 包一层
  - 连续失败达到 threshold → OPEN（直接抛 CircuitOpenError 给上层）
  - 防止 OpenAI 接口挂掉时把我们的请求队列塞满
"""

from __future__ import annotations

import logging
from typing import Any

from ...observability.circuit_breaker import get_openai_breaker
from .base import EmbeddingConfig, register_embedder

logger = logging.getLogger(__name__)


@register_embedder("openai")
class OpenAIEmbedder:
    """OpenAI Embedder。"""

    def __init__(self, config: EmbeddingConfig):
        self.dimension = config.dimension
        self._config = config

        # 懒加载 openai
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAIEmbedder requires the 'openai' package. "
                "Install with: pip install study-rag[embedding-openai]"
            ) from e

        # 从 extra 取可选配置
        extra = config.extra or {}
        api_key = extra.get("api_key")
        base_url = extra.get("base_url")
        timeout = extra.get("timeout", 60.0)
        max_retries = extra.get("max_retries", 3)

        if not api_key:
            # 尝试从环境变量读取
            import os

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAIEmbedder requires 'api_key' in config.extra "
                    "or OPENAI_API_KEY env var"
                )

        client_kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": max_retries}
        if base_url:
            client_kwargs["base_url"] = base_url
        if timeout:
            client_kwargs["timeout"] = timeout

        self._client = AsyncOpenAI(**client_kwargs)
        self._model = config.model_name
        self._batch_size = config.batch_size
        self._encoding_format = extra.get("encoding_format", "float")

        logger.info(
            "OpenAIEmbedder initialized: model=%s, dim=%d, base_url=%s",
            self._model,
            self.dimension,
            base_url or "default",
        )

    async def embed_query(self, text: str) -> list[float]:
        """编码单个查询。"""
        result = await self._embed_batch([text])
        return result[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量编码文档，按 batch_size 切片处理。"""
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            vectors = await self._embed_batch(batch)
            all_vectors.extend(vectors)
        return all_vectors

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """调用 OpenAI API 编码一批文本（走熔断器）。"""
        kwargs: dict[str, Any] = {
            "input": texts,
            "model": self._model,
        }
        if self._encoding_format and self._encoding_format != "float":
            kwargs["encoding_format"] = self._encoding_format
        # text-embedding-3-* 支持自定义 dimension
        if self._model.startswith("text-embedding-3") and self.dimension:
            kwargs["dimensions"] = self.dimension

        # 走熔断器：保护 OpenAI 不被连续失败打爆
        async def _call() -> Any:
            return await self._client.embeddings.create(**kwargs)

        response = await get_openai_breaker().call(_call)

        vectors = [item.embedding for item in response.data]

        # 维度校验（防止配置错误）
        if vectors and len(vectors[0]) != self.dimension:
            actual = len(vectors[0])
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"config={self.dimension}, actual={actual}. "
                f"Check model_name and dimension in config."
            )

        return vectors


@register_embedder("azure_openai")
class AzureOpenAIEmbedder(OpenAIEmbedder):
    """Azure OpenAI Embedder（继承自 OpenAI，配置 base_url 即可）。

    典型的 base_url: https://{resource}.openai.azure.com/openai/deployments/{deployment}
    Azure 不支持 dimensions 参数，会用部署时设定的维度。
    """

    def __init__(self, config: EmbeddingConfig):
        # Azure 强制使用 api_version
        extra = config.extra or {}
        if "api_version" not in extra:
            extra["api_version"] = "2024-02-01"
        config.extra = extra
        # Azure 不支持自定义 dimensions
        if config.model_name and "/" not in config.model_name:
            # model_name 应该是 deployment 名
            pass
        super().__init__(config)
