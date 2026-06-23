"""Vector Store 能力抽象。

定义向量库的接口 + 工厂方法。
具体实现（milvus、qdrant、chroma 等）由 impl 子模块提供。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class VectorStoreConfig(BaseModel):
    """向量库配置（YAML schema）。"""

    provider: str = Field(..., description="实现 provider：mock / milvus / qdrant")
    uri: str = Field(default="", description="连接地址")
    extra: dict = Field(default_factory=dict, description="扩展参数")


class VectorRecord(BaseModel):
    """向量记录。"""

    id: str
    vector: list[float]
    text: str
    metadata: dict = Field(default_factory=dict)


class SearchResult(BaseModel):
    """检索结果。"""

    id: str
    text: str
    score: float
    metadata: dict = Field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    """向量库接口。

    每个 collection 对应一个知识库。
    """

    async def create_collection(self, name: str, dimension: int) -> None: ...
    async def drop_collection(self, name: str) -> None: ...
    async def has_collection(self, name: str) -> bool: ...

    async def insert(self, collection: str, records: list[VectorRecord]) -> None: ...
    async def delete(
        self,
        collection: str,
        ids: list[str] | None = None,
        filter_expr: dict | None = None,
    ) -> int:
        """删除 collection 中匹配的记录。

        - ids: 主键列表（与 filter_expr 互斥；同时传行为由实现决定）
        - filter_expr: metadata 过滤字典
        - 返回删除的记录数
        - collection 不存在 / 空 args → 返回 0，不抛错
        """
        ...
    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 5,
        filter_expr: dict | None = None,
    ) -> list[SearchResult]: ...
    async def query(
        self,
        collection: str,
        filter_expr: dict | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[VectorRecord]: ...
    async def count(self, collection: str) -> int: ...


_VECTOR_STORE_REGISTRY: dict[str, type[VectorStore]] = {}


def register_vector_store(provider: str):
    """装饰器：注册 VectorStore 实现。"""

    def decorator(cls: type[VectorStore]) -> type[VectorStore]:
        _VECTOR_STORE_REGISTRY[provider] = cls
        return cls

    return decorator


def create_vector_store(config: VectorStoreConfig) -> VectorStore:
    """根据配置创建 VectorStore 实例。"""
    impl_cls = _VECTOR_STORE_REGISTRY.get(config.provider)
    if impl_cls is None:
        raise ValueError(
            f"Unknown vector store provider: {config.provider}. "
            f"Available: {list(_VECTOR_STORE_REGISTRY.keys())}"
        )
    return impl_cls(config)  # type: ignore[abstract, call-arg]


def list_vector_store_providers() -> list[str]:
    """列出已注册的 vector store provider。"""
    return list(_VECTOR_STORE_REGISTRY.keys())
