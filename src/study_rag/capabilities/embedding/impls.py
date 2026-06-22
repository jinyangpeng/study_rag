"""Mock Embedder：用于本地开发和测试，不依赖任何外部服务。

无需任何第三方包。
"""

from __future__ import annotations

import hashlib
import math

from .base import EmbeddingConfig, register_embedder


@register_embedder("mock")
class MockEmbedder:
    """基于 hash 的伪 Embedder。

    通过对文本 hash 后归一化生成固定维度的向量。
    语义上不真实（不同文本可能 hash 后相似），但接口完全一致。
    """

    def __init__(self, config: EmbeddingConfig):
        self.dimension = config.dimension
        self._config = config

    async def embed_query(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def _hash_to_vector(self, text: str) -> list[float]:
        """通过多次 hash 拼出一个固定维度的向量并归一化。"""
        vec: list[float] = []
        seed = text.encode("utf-8")
        # 用多次 sha256 拼接
        for i in range(math.ceil(self.dimension / 32)):
            h = hashlib.sha256(seed + i.to_bytes(2, "big")).digest()
            for byte in h:
                vec.append((byte / 255.0) - 0.5)  # 归一到 [-0.5, 0.5]
        vec = vec[: self.dimension]
        # L2 归一化
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]
