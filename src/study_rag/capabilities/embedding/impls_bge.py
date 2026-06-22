"""BGE Embedder：通过 FlagEmbedding 在本地加载 BGE 系列模型。

支持的模型：
  - BAAI/bge-m3              (1024 维, 多语言, 推荐)
  - BAAI/bge-large-zh-v1.5   (1024 维, 中文)
  - BAAI/bge-base-zh-v1.5    (768 维, 中文)
  - BAAI/bge-small-zh-v1.5   (512 维, 中文)

依赖：FlagEmbedding>=1.2.0 + torch>=2.0.0
安装：pip install study-rag[embedding-bge]

注：首次运行会从 HuggingFace 下载模型（约 2-3GB），需要联网。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .base import EmbeddingConfig, register_embedder

logger = logging.getLogger(__name__)


def _get_flag_model_class():
    """懒加载 FlagEmbedding.FlagModel。"""
    try:
        from FlagEmbedding import FlagModel  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "BGEEmbedder requires 'FlagEmbedding' and 'torch'. "
            "Install with: pip install study-rag[embedding-bge]"
        ) from e
    return FlagModel


@register_embedder("bge")
class BGEEmbedder:
    """BGE Embedder（基于 FlagModel）。

    适合：
      - 中文/多语言知识库
      - 数据隐私要求高（本地推理）
      - 不希望走外部 API

    性能提示：
      - 首次加载模型较慢（约 10-30s）
      - CPU 推理约 50-200 docs/s
      - GPU 推理约 1000+ docs/s
    """

    # BGE 推荐的 query 前缀
    BGE_QUERY_PREFIX_ZH = "为这个句子生成表示以用于检索相关文章："
    BGE_QUERY_PREFIX_EN = "Represent this sentence for searching relevant passages: "

    def __init__(self, config: EmbeddingConfig):
        self.dimension = config.dimension
        self._config = config

        FlagModel = _get_flag_model_class()  # noqa: N806

        extra = config.extra or {}
        query_instruction = extra.get("query_instruction")
        use_fp16 = extra.get("use_fp16", True)
        device = extra.get("device")  # None = auto
        cache_dir = extra.get("cache_dir")

        kwargs: dict[str, Any] = {
            "use_fp16": use_fp16,
        }
        if query_instruction is not None:
            kwargs["query_instruction"] = query_instruction
        if device:
            kwargs["devices"] = device
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        # BGE-M3 走 BGEM3FlagModel，其他走 FlagModel
        # 通过 model_name 后缀判断
        model_name = config.model_name
        if "m3" in model_name.lower():
            # FlagEmbedding 会根据模型类型自动选择
            # BGE-M3 输出 dense(1024) + sparse + colbert，这里只用 dense
            kwargs["normalize_embeddings"] = True
        else:
            kwargs["normalize_embeddings"] = True

        logger.info("Loading BGE model: %s (dim=%d)", model_name, self.dimension)
        self._model = FlagModel(model_name, **kwargs)
        self._batch_size = config.batch_size
        logger.info("BGE model loaded: %s", model_name)

    async def embed_query(self, text: str) -> list[float]:
        """编码单个查询（带 query 前缀）。"""
        vectors = await asyncio.to_thread(self._encode_queries, [text])
        return vectors[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量编码文档（不带 query 前缀）。"""
        return await asyncio.to_thread(self._encode_documents, texts)

    def _encode_queries(self, texts: list[str]) -> list[list[float]]:
        """同步：编码查询（FlagModel 会自动加 query 前缀）。"""
        result = self._model.encode(
            texts,
            batch_size=self._batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return self._extract_dense(result)

    def _encode_documents(self, texts: list[str]) -> list[list[float]]:
        """同步：编码文档。"""
        result = self._model.encode(
            texts,
            batch_size=self._batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return self._extract_dense(result)

    @staticmethod
    def _extract_dense(result: Any) -> list[list[float]]:
        """从 FlagModel 输出中提取 dense vectors。"""
        import numpy as np

        if isinstance(result, dict):
            dense = result.get("dense_vecs")
        else:
            # BGE-M3 返回 dict，其他模型直接返回 numpy
            dense = result

        if dense is None:
            raise RuntimeError("FlagModel returned no dense vectors")

        if isinstance(dense, list):
            return [v.tolist() if hasattr(v, "tolist") else list(v) for v in dense]

        # numpy array
        if isinstance(dense, np.ndarray):
            return dense.tolist()

        raise RuntimeError(f"Unexpected dense vector type: {type(dense)}")


@register_embedder("bge_zh")
class BGEZhEmbedder(BGEEmbedder):
    """BGE 中文模型快捷 provider，自动应用中文 query 前缀。"""

    def __init__(self, config: EmbeddingConfig):
        # 自动注入中文 query 前缀
        extra = dict(config.extra or {})
        if "query_instruction" not in extra:
            extra["query_instruction"] = self.BGE_QUERY_PREFIX_ZH
        config.extra = extra
        super().__init__(config)
