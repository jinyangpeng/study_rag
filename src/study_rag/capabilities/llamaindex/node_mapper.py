"""Node 互转工具：SearchResult <-> LlamaIndex Node / NodeWithScore。"""

from __future__ import annotations

from typing import Any

from ..vector_store.base import SearchResult

__all__ = ["NodeMapper"]


class NodeMapper:
    """我们在 SearchResult 和 LI Node / NodeWithScore 之间互转。"""

    @staticmethod
    def search_result_to_node(r: SearchResult) -> Any:
        """SearchResult -> LI TextNode。"""
        from llama_index.core.schema import TextNode

        return TextNode(
            id_=r.id,
            text=r.text,
            metadata=r.metadata or {},
        )

    @staticmethod
    def search_result_to_node_with_score(r: SearchResult) -> Any:
        """SearchResult -> LI NodeWithScore。"""
        from llama_index.core.schema import NodeWithScore

        node = NodeMapper.search_result_to_node(r)
        return NodeWithScore(node=node, score=r.score)

    @staticmethod
    def node_with_score_to_search_result(n: Any) -> SearchResult:
        """LI NodeWithScore -> SearchResult。"""
        from llama_index.core.schema import NodeWithScore

        if isinstance(n, NodeWithScore):
            node = n.node
            score = float(n.score) if n.score is not None else 0.0
        else:
            node = n
            score = 0.0
        text = node.get_content() if hasattr(node, "get_content") else str(node)
        return SearchResult(
            id=node.node_id,
            text=text,
            score=score,
            metadata=dict(node.metadata or {}),
        )

    @staticmethod
    def nodes_to_search_results(nodes: list[Any]) -> list[SearchResult]:
        """LI NodeWithScore 列表 -> SearchResult 列表。"""
        return [NodeMapper.node_with_score_to_search_result(n) for n in nodes]

    @staticmethod
    def search_results_to_nodes_with_score(
        results: list[SearchResult],
    ) -> list[Any]:
        """SearchResult 列表 -> LI NodeWithScore 列表。"""
        return [NodeMapper.search_result_to_node_with_score(r) for r in results]
