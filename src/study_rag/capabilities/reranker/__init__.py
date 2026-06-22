"""Reranker 能力包。

注册机制：
  - impls.py（mock / none）：无依赖，始终可用
  - impls_bge.py（bge / bge_m3）：需要 FlagEmbedding + torch
  - impls_cohere.py（cohere）：需要 cohere SDK
  - impls_http.py（http）：需要 httpx（项目默认依赖），对接自托管 TEI / Jina / Cohere 兼容服务

缺失依赖时仅该 provider 不可用，其他 provider 正常工作。
"""

# Mock / None: 始终可用
from . import impls  # noqa: F401, E402
from .base import (
    Reranker,
    RerankerConfig,
    create_reranker,
    list_reranker_providers,
    register_reranker,
)

# HTTP (TEI / Jina / Cohere 自托管): httpx 缺失时跳过
try:
    from . import impls_http  # noqa: F401, E402
except ImportError as e:
    import logging

    logging.getLogger(__name__).debug("HTTP reranker not available: %s", e)

# BGE: 缺失依赖时跳过
try:
    from . import impls_bge  # noqa: F401, E402
except ImportError as e:
    import logging

    logging.getLogger(__name__).debug("BGE reranker not available: %s", e)

# Cohere: 缺失依赖时跳过
try:
    from . import impls_cohere  # noqa: F401, E402
except ImportError as e:
    import logging

    logging.getLogger(__name__).debug("Cohere reranker not available: %s", e)

__all__ = [
    "Reranker",
    "RerankerConfig",
    "create_reranker",
    "list_reranker_providers",
    "register_reranker",
]
