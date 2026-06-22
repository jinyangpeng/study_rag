"""BGE Reranker：基于 FlagEmbedding 的本地重排。

支持的模型：
  - BAAI/bge-reranker-v2-m3       (多语言, 推荐)
  - BAAI/bge-reranker-large       (英文)
  - BAAI/bge-reranker-base        (英文, 更快)
  - BAAI/bge-reranker-v2-gemma    (Gemma-based, 质量更高)

依赖：FlagEmbedding>=1.2.0 + torch>=2.0.0
安装：pip install study-rag[reranker-bge]

注：
  - 首次运行会从 HuggingFace 下载模型（约 1-2GB），需要联网。
  - BGE reranker 内部已经做了 query/passage 拼接，无需手动加前缀。
  - 推理建议 batch，避免长 context 时 OOM。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..vector_store.base import SearchResult
from .base import RerankerConfig, register_reranker

logger = logging.getLogger(__name__)


def _get_flag_reranker_class():
    """懒加载 FlagEmbedding 的 Reranker 类。"""
    try:
        from FlagEmbedding import FlagReranker  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "BGEReranker requires 'FlagEmbedding' and 'torch'. "
            "Install with: pip install study-rag[reranker-bge]"
        ) from e
    return FlagReranker


@register_reranker("bge")
class BGEReranker:
    """BGE Reranker（基于 FlagEmbedding.FlagReranker）。

    适合：
      - 中文 / 多语言重排
      - 数据隐私要求高（本地推理）
      - 不想走外部 API

    性能：
      - 首次加载模型较慢（约 10-30s）
      - CPU 推理约 50-200 docs/s
      - GPU 推理约 1000+ docs/s
    """

    def __init__(self, config: RerankerConfig):
        self._config = config
        self._top_k = config.top_k
        self._batch_size = int(config.extra.get("batch_size", 32)) if config.extra else 32

        FlagReranker = _get_flag_reranker_class()  # noqa: N806

        extra = config.extra or {}
        use_fp16 = extra.get("use_fp16", True)
        device = extra.get("device")  # None = auto
        cache_dir = extra.get("cache_dir")

        kwargs: dict[str, Any] = {"use_fp16": use_fp16}
        if device:
            kwargs["devices"] = device
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        model_name = config.model_name or "BAAI/bge-reranker-v2-m3"
        logger.info("Loading BGE reranker: %s (batch_size=%d)", model_name, self._batch_size)
        self._model = FlagReranker(model_name, **kwargs)
        logger.info("BGE reranker loaded: %s", model_name)

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """对候选结果重排，按相关性倒序返回 top_k。"""
        if not results:
            return []

        # FlagReranker.compute_score([[q, doc], ...]) -> list[float]
        pairs = [[query, r.text] for r in results]

        def _score() -> list[float]:
            scores: list[float] = []
            # 按 batch_size 切片
            for i in range(0, len(pairs), self._batch_size):
                batch = pairs[i : i + self._batch_size]
                s = self._model.compute_score(batch, normalize=True)
                if isinstance(s, float):
                    s = [s]
                scores.extend(s)
            return scores

        scores = await asyncio.to_thread(_score)

        # 拼回 SearchResult
        scored = list(zip(results, scores, strict=False))
        # normalize=True 时 score ∈ [0, 1]，越大越相关
        scored.sort(key=lambda x: x[1], reverse=True)

        k = top_k if top_k is not None else self._top_k
        out: list[SearchResult] = []
        for r, s in scored[:k]:
            # 复制一份，更新 score（不修改原对象）
            out.append(
                SearchResult(
                    id=r.id,
                    text=r.text,
                    score=float(s),
                    metadata=r.metadata,
                )
            )
        logger.debug(
            "BGE rerank query=%r top_k=%d -> %d results",
            query[:30],
            k,
            len(out),
        )
        return out


# 便捷别名
@register_reranker("bge_m3")
class BGERerankerM3(BGEReranker):
    """BGE Reranker M3 的便捷别名。"""

    def __init__(self, config: RerankerConfig):
        if not config.model_name:
            config.model_name = "BAAI/bge-reranker-v2-m3"
        super().__init__(config)
