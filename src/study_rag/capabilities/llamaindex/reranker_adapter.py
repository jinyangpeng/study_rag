"""Reranker 适配器：把我们的 Reranker 包装为 LlamaIndex 的 BaseNodePostprocessor。

这样我们的 reranker 可以作为 LI QueryEngine 的 postprocessor 使用。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..reranker.base import Reranker
from ..vector_store.base import SearchResult

logger = logging.getLogger(__name__)

__all__ = ["LIRerankerPostprocessor"]


class LIRerankerPostprocessor:
    """LlamaIndex NodePostProcessor 适配器：把我们的 Reranker 接入 LI。"""

    def __init__(self, reranker: Reranker, top_n: int = 5):
        self._reranker = reranker
        self._top_n = top_n

    # ---- LI BaseNodePostprocessor 接口 ----

    async def _postprocess_nodes(
        self,
        nodes: list[Any],
        query: Any = None,
    ) -> list[Any]:
        """LI 异步调用入口。"""
        if not nodes:
            return []

        # query 可能是 QueryBundle
        q_text = getattr(query, "query_str", str(query)) if query else ""

        # 我们的 Reranker 接受 SearchResult 列表
        from ..vector_store.base import SearchResult

        results: list[SearchResult] = []
        for n in nodes:
            text = n.node.get_content() if hasattr(n.node, "get_content") else str(n.node)
            results.append(
                SearchResult(
                    id=n.node.node_id,
                    text=text,
                    score=float(n.score) if n.score is not None else 0.0,
                    metadata=dict(n.node.metadata or {}),
                )
            )

        reranked: list[SearchResult] = await self._reranker.rerank(
            query=q_text, results=results, top_k=self._top_n,
        )

        # 转回 NodeWithScore
        from llama_index.core.schema import NodeWithScore, TextNode

        out: list[NodeWithScore] = []
        for r in reranked:
            node = TextNode(
                id_=r.id,
                text=r.text,
                metadata=r.metadata or {},
            )
            out.append(NodeWithScore(node=node, score=r.score))
        return out

    def postprocess_nodes(
        self,
        nodes: list[Any],
        query: Any = None,
    ) -> list[Any]:
        """同步版本。"""
        return asyncio.run(self._postprocess_nodes(nodes, query))
