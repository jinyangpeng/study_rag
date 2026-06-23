"""Mock Vector Store：基于内存，用于本地开发和测试。"""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict

from .base import SearchResult, VectorRecord, VectorStoreConfig, register_vector_store
from .filters import matches_filter


@register_vector_store("mock")
class InMemoryVectorStore:
    """内存版向量库，支持 cosine similarity + metadata filter。"""

    def __init__(self, config: VectorStoreConfig):
        self._config = config
        # collection_name -> list[VectorRecord]
        self._collections: dict[str, list[VectorRecord]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def create_collection(self, name: str, dimension: int) -> None:
        async with self._lock:
            if name not in self._collections:
                self._collections[name] = []

    async def drop_collection(self, name: str) -> None:
        async with self._lock:
            self._collections.pop(name, None)

    async def has_collection(self, name: str) -> bool:
        return name in self._collections

    async def insert(self, collection: str, records: list[VectorRecord]) -> None:
        async with self._lock:
            self._collections[collection].extend(records)

    async def delete(self, collection: str, ids: list[str]) -> None:
        async with self._lock:
            id_set = set(ids)
            self._collections[collection] = [
                r for r in self._collections[collection] if r.id not in id_set
            ]

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 5,
        filter_expr: dict | None = None,
    ) -> list[SearchResult]:
        records = self._collections.get(collection, [])
        if not records:
            return []

        scored: list[tuple[float, VectorRecord]] = []
        for r in records:
            # 先按 metadata filter 过滤（不命中直接跳过，省得算相似度）
            if not matches_filter(r.metadata, filter_expr):
                continue
            score = self._cosine(query_vector, r.vector)
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]
        return [
            SearchResult(id=r.id, text=r.text, score=score, metadata=r.metadata)
            for score, r in top
        ]

    async def query(
        self,
        collection: str,
        filter_expr: dict | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[VectorRecord]:
        """按 metadata 过滤取所有 chunks（不查相似度）。

        Args:
            collection: collection 名
            filter_expr: metadata 过滤表达式（复用 matches_filter）
            limit: 返回上限
            offset: 跳过条数

        Returns:
            匹配的 VectorRecord 列表（不含 score，因为没有相似度计算）
        """
        records = self._collections.get(collection, [])
        if not records:
            return []
        matched = [r for r in records if matches_filter(r.metadata, filter_expr)]
        return matched[offset:offset + limit]

    async def count(self, collection: str) -> int:
        """O(1) mock 实现：直接读 in-memory dict 的长度。

        collection 不存在 → 返回 0（不抛错）。
        """
        records = self._collections.get(collection, [])
        return len(records)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (na * nb)
