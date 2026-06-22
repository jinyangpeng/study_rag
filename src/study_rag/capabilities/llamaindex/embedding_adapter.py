"""Embedding 适配器：把我们的 Embedder 包装为 LlamaIndex 的 BaseEmbedding。

这样 LlamaIndex 的 VectorStoreIndex / Retriever 就能直接用我们的 embedder。
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging

from ..embedding.base import Embedder

logger = logging.getLogger(__name__)

__all__ = ["LIEmbeddingAdapter"]


def _get_li_base_embedding():
    try:
        from llama_index.core.embeddings import BaseEmbedding
    except ImportError as e:
        raise ImportError(
            "需要 llama-index-core。安装: pip install llama-index-core"
        ) from e
    return BaseEmbedding


# LlamaIndex 的 SemanticSplitter 等会调 BaseEmbedding 的 sync 入口（_get_text_embeddings），
# 而我们的 Embedder 是 async 的；FastAPI handler 又是 async def 跑在事件循环里，
# 直接 asyncio.run() 会抛 "cannot be called from a running event loop"。
# 解决办法：把 async 协程丢到独立线程里跑（独立事件循环），sync / async 上下文都安全。
_sync_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="li-emb-sync"
)


def _run_async(coro):
    """在独立线程的事件循环里跑 async 协程。"""
    return _sync_executor.submit(asyncio.run, coro).result()


class LIEmbeddingAdapter(_get_li_base_embedding()):  # type: ignore[misc]
    """把我们的 Embedder 适配为 LlamaIndex 的 BaseEmbedding。

    用法：
        adapter = LIEmbeddingAdapter(our_embedder)
        Settings.embed_model = adapter  # 让 LI 全局使用我们的 embedder
    """

    def __init__(self, embedder: Embedder):
        # BaseEmbedding 父类初始化需要一些参数；LI 0.14.x 的实现较宽松
        try:
            super().__init__(model_name=embedder._config.model_name)
        except Exception:
            # 兜底：传 kwargs
            try:
                super().__init__()
            except Exception:
                pass
        self._embedder = embedder
        self._dim = embedder.dimension
        self._model_name = embedder._config.model_name
        self._batch_size = embedder._config.batch_size

    # ---- LI BaseEmbedding 要求的属性 ----

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embed_batch_size(self) -> int:
        return self._batch_size

    @property
    def dimension(self) -> int:
        return self._dim

    # ---- LI 要求的接口 ----

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await self._embedder.embed_query(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return await self._embedder.embed_query(text)  # 单文本

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        # 默认逐条调用；某些 embedder 有批量方法
        if hasattr(self._embedder, "embed_documents"):
            return await self._embedder.embed_documents(texts)
        # fallback
        results: list[list[float]] = []
        for t in texts:
            results.append(await self._embedder.embed_query(t))
        return results

    # 同步版本（LI 某些场景会调用）
    def _get_query_embedding(self, query: str) -> list[float]:
        return _run_async(self._aget_query_embedding(query))

    def _get_text_embedding(self, text: str) -> list[float]:
        return _run_async(self._aget_text_embedding(text))

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return _run_async(self._aget_text_embeddings(texts))
