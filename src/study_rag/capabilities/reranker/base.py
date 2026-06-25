"""Reranker 能力抽象。

对初检结果重排序，提升 top-k 准确率。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..vector_store.base import SearchResult


class RerankerConfig(BaseModel):
    """Reranker 配置（YAML schema）。"""

    provider: str = Field(..., description="实现 provider：mock / bge / cohere / http")
    # 仅 provider=http 时使用，显式声明以便 Pydantic 校验，避免写在 extra 被忽略
    protocol: str = Field(
        default="",
        description="HTTP reranker 协议：tei / jina / cohere_compat / openai（仅 provider=http 生效）",
    )
    model_name: str = Field(default="", description="模型名称")
    top_k: int = Field(default=5, description="重排后保留数量")
    extra: dict = Field(default_factory=dict, description="扩展参数")


@runtime_checkable
class Reranker(Protocol):
    """Reranker 接口。"""

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]: ...


_RERANKER_REGISTRY: dict[str, type[Reranker]] = {}


def register_reranker(provider: str):
    def decorator(cls: type[Reranker]) -> type[Reranker]:
        _RERANKER_REGISTRY[provider] = cls
        return cls

    return decorator


def create_reranker(config: RerankerConfig) -> Reranker:
    impl_cls = _RERANKER_REGISTRY.get(config.provider)
    if impl_cls is None:
        raise ValueError(
            f"Unknown reranker provider: {config.provider}. "
            f"Available: {list(_RERANKER_REGISTRY.keys())}"
        )
    return impl_cls(config)  # type: ignore[abstract, call-arg]


def list_reranker_providers() -> list[str]:
    return list(_RERANKER_REGISTRY.keys())
