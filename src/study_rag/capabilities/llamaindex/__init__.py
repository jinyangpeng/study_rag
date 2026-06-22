"""LlamaIndex 整合包：把 LlamaIndex 组件封装到本项目的能力体系内。

提供：
  - parser.py            NodeParser 包装（Sentence / Semantic / whole）
  - embedding_adapter.py 把我们 Embedder 适配为 LlamaIndex BaseEmbedding
  - vector_store_adapter.py 把我们 VectorStore 适配为 LI BasePydanticVectorStore
  - reranker_adapter.py  把我们 Reranker 适配为 LI BaseNodePostprocessor
  - retrieval_engine.py  LlamaIndexRetrievalEngine（用 VectorStoreIndex + retriever）
  - node_mapper.py       我们 SearchResult <-> LI NodeWithScore 互转

设计原则：
  1. 全部懒加载：缺 llama-index 时该包不应让项目崩溃
  2. 不替代现有能力：在 capabilities/ 之上的"可选层"
  3. 两条检索路径并存：DefaultRetrievalEngine vs LlamaIndexRetrievalEngine

安装：
  pip install llama-index-core
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

__all__ = [
    "DocumentReader",
    "LIEmbeddingAdapter",
    "LIRerankerPostprocessor",
    "LIVectorStoreAdapter",
    "LlamaIndexRetrievalEngine",
    "NodeMapper",
    "NodeParserConfig",
    "NodeParserFactory",
    "ParserRegistry",
    "ParserSpec",
    "UnsupportedFormatError",
    "get_parser_registry",
    "is_llama_index_available",
    "read_document",
]


def is_llama_index_available() -> bool:
    """检测 LlamaIndex 是否安装。"""
    try:
        import llama_index.core  # noqa: F401

        return True
    except ImportError:
        return False


# 懒加载子模块（缺 llama-index-core 时不导入实际类）
_submodule_names = (
    "parser",
    "embedding_adapter",
    "vector_store_adapter",
    "reranker_adapter",
    "retrieval_engine",
    "node_mapper",
    "registry",
    "reader",
)

# 用 __getattr__ 实现 PEP 562 懒加载（type checker 友好）
_submodule_cache: dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    """按需导入子模块的属性。"""
    if name in __all__ and name != "is_llama_index_available":
        # 找到定义 name 的子模块
        for mod_name in _submodule_names:
            mod = importlib.import_module(f".{mod_name}", __name__)
            if name in getattr(mod, "__all__", []) or hasattr(mod, name):
                _submodule_cache[mod_name] = mod
                return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# 预导入子模块（让符号可被发现，但不导入 llama_index 相关）
# 用 TYPE_CHECKING 提供静态分析
if TYPE_CHECKING:
    from .embedding_adapter import LIEmbeddingAdapter
    from .node_mapper import NodeMapper
    from .parser import NodeParserConfig, NodeParserFactory
    from .reader import DocumentReader, UnsupportedFormatError, read_document
    from .registry import ParserRegistry, ParserSpec, get_parser_registry
    from .reranker_adapter import LIRerankerPostprocessor
    from .retrieval_engine import LlamaIndexRetrievalEngine
    from .vector_store_adapter import LIVectorStoreAdapter
