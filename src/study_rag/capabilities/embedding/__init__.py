"""Embedding 能力包。

注册机制：
  - impls.py（mock）：无依赖，始终可用
  - impls_openai.py（openai / azure_openai）：需要 openai 包
  - impls_bge.py（bge / bge_zh）：需要 FlagEmbedding + torch

缺失依赖时仅该 provider 不可用，其他 provider 正常工作。
"""

# Mock: 始终可用
from . import impls  # noqa: F401, E402
from .base import (
    Embedder,
    EmbeddingConfig,
    create_embedder,
    list_embedder_providers,
    register_embedder,
)

# OpenAI: 缺失依赖时跳过
try:
    from . import impls_openai  # noqa: F401, E402
except ImportError as e:
    import logging

    logging.getLogger(__name__).debug("OpenAI embedder not available: %s", e)

# BGE: 缺失依赖时跳过
try:
    from . import impls_bge  # noqa: F401, E402
except ImportError as e:
    import logging

    logging.getLogger(__name__).debug("BGE embedder not available: %s", e)


__all__ = [
    "Embedder",
    "EmbeddingConfig",
    "create_embedder",
    "list_embedder_providers",
    "register_embedder",
]
