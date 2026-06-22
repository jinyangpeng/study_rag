"""Embedding 能力抽象。

定义 Embedder 接口 + 工厂方法。
具体实现（bge、openai 等）由 impl 子模块提供。
"""

from __future__ import annotations

import os
import re
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

# ${VAR_NAME} 形式的占位符
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _resolve_env(value: Any) -> Any:
    """递归解析 dict/list/str 中的 ${ENV_VAR} 占位符。"""
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


class EmbeddingConfig(BaseModel):
    """Embedding 模型配置（YAML schema）。

    支持额外字段（如 description、query_instruction 等），运行时通过 `extra` 访问。
    """

    model_config = ConfigDict(extra="allow")

    provider: str = Field(..., description="实现 provider：mock / openai / bge 等")
    model_name: str = Field(..., description="模型名称")
    dimension: int = Field(..., description="向量维度")
    batch_size: int = Field(default=32, description="批处理大小")
    extra: dict[str, Any] = Field(default_factory=dict, description="扩展参数")

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> EmbeddingConfig:
        """从原始 dict 构造，自动处理 ${ENV_VAR} 替换。

        顶层自定义字段（如 description）会归入 extra。
        """
        resolved = _resolve_env(raw)
        known = {"provider", "model_name", "dimension", "batch_size", "extra"}
        # 把未知顶层字段 + extra 字段合并到 extra
        extra_data: dict[str, Any] = {}
        for k, v in resolved.items():
            if k not in known:
                extra_data[k] = v
        if "extra" in resolved and isinstance(resolved["extra"], dict):
            extra_data.update(resolved["extra"])
        merged = {**resolved, "extra": extra_data}
        return cls(**merged)


@runtime_checkable
class Embedder(Protocol):
    """Embedding 接口。

    任何实现都需要：
      - embed_query: 编码单个查询
      - embed_documents: 编码多个文档
    """

    dimension: int
    _config: EmbeddingConfig

    async def embed_query(self, text: str) -> list[float]: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


# Registry of embedder implementations
_EMBEDDER_REGISTRY: dict[str, type[Embedder]] = {}


def register_embedder(provider: str):
    """装饰器：注册 Embedder 实现。

    Usage:
        @register_embedder("bge")
        class BGEEmbedder:
            ...
    """

    def decorator(cls: type[Embedder]) -> type[Embedder]:
        _EMBEDDER_REGISTRY[provider] = cls
        return cls

    return decorator


def create_embedder(config: EmbeddingConfig) -> Embedder:
    """根据配置创建 Embedder 实例。"""
    impl_cls = _EMBEDDER_REGISTRY.get(config.provider)
    if impl_cls is None:
        raise ValueError(
            f"Unknown embedder provider: {config.provider}. "
            f"Available: {list(_EMBEDDER_REGISTRY.keys())}"
        )
    return impl_cls(config)  # type: ignore[abstract, call-arg]


def list_embedder_providers() -> list[str]:
    """列出已注册的 embedder provider。"""
    return list(_EMBEDDER_REGISTRY.keys())
