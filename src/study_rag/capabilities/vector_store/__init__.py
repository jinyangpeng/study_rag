"""Vector Store 能力包。

注册机制：
  - impls.py（mock）：无依赖，始终可用
  - impls_milvus.py（milvus）：需要 pymilvus 包

缺失依赖时仅该 provider 不可用，其他 provider 正常工作。
"""

# Mock: 始终可用
from . import impls  # noqa: F401, E402
from .base import (
    SearchResult,
    VectorRecord,
    VectorStore,
    VectorStoreConfig,
    create_vector_store,
    list_vector_store_providers,
    register_vector_store,
)

# Milvus: 缺失依赖时跳过
try:
    from . import impls_milvus  # noqa: F401, E402
except ImportError as e:
    import logging

    logging.getLogger(__name__).debug("Milvus vector store not available: %s", e)

__all__ = [
    "SearchResult",
    "VectorRecord",
    "VectorStore",
    "VectorStoreConfig",
    "create_vector_store",
    "list_vector_store_providers",
    "register_vector_store",
]
