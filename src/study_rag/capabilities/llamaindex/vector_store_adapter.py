"""VectorStore 适配器：把我们的 VectorStore 适配为 LlamaIndex 的 BasePydanticVectorStore。

这是关键的桥接器：
    our VectorStore  ──┐
                      ├──>  LlamaIndex VectorStoreIndex.from_vector_store()
    our Embedder     ──┘

适配后即可使用 LlamaIndex 的所有上层 API：
  - VectorStoreIndex
  - Retriever
  - QueryEngine
  - ResponseSynthesizer
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, ClassVar, cast

from ..vector_store.base import SearchResult, VectorRecord, VectorStore

logger = logging.getLogger(__name__)

__all__ = ["LIVectorStoreAdapter"]


def _get_li_types():
    """懒加载 LI 类型。"""
    try:
        from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
        from llama_index.core.vector_stores.types import (
            BasePydanticVectorStore,
            VectorStoreQuery,
            VectorStoreQueryResult,
        )
    except ImportError as e:
        raise ImportError(
            "需要 llama-index-core。安装: pip install llama-index-core"
        ) from e
    return {
        "BasePydanticVectorStore": BasePydanticVectorStore,
        "VectorStoreQuery": VectorStoreQuery,
        "VectorStoreQueryResult": VectorStoreQueryResult,
        "TextNode": TextNode,
        "NodeRelationship": NodeRelationship,
        "RelatedNodeInfo": RelatedNodeInfo,
    }


def _convert_filter_to_li(filter_expr: Any) -> Any:
    """把我们的 filter dict 转成 LI 的 MetadataFilters。

    注：LI 用 MetadataFilters 表达嵌套条件。简单起见，我们用 and 合并。
    """
    if not filter_expr:
        return None
    try:
        from llama_index.core.vector_stores.types import (
            FilterCondition,
            FilterOperator,
            MetadataFilter,
            MetadataFilters,
        )
    except ImportError:
        return None

    items: list[MetadataFilter] = []
    for key, value in filter_expr.items():
        if "__" in key:
            field, op = key.rsplit("__", 1)
        else:
            field, op = key, "eq"
        op_map = {
            "eq": FilterOperator.EQ,
            "ne": FilterOperator.NE,
            "in": FilterOperator.IN,
            "gt": FilterOperator.GT,
            "gte": FilterOperator.GTE,
            "lt": FilterOperator.LT,
            "lte": FilterOperator.LTE,
        }
        if op not in op_map:
            continue
        items.append(MetadataFilter(key=field, value=value, operator=op_map[op]))

    if not items:
        return None
    if len(items) == 1:
        return items[0]
    # LI 的 MetadataFilters.filters 接受 Sequence[MetadataFilter | MetadataFilters]
    # 用 cast 绕过 invariant list 检查（runtime 正确）
    return MetadataFilters(
        filters=cast("list[MetadataFilter | Any]", items),
        condition=FilterCondition.AND,
    )


class LIVectorStoreAdapter(_get_li_types()["BasePydanticVectorStore"]):  # type: ignore[misc]
    """LlamaIndex VectorStore 适配器。

    用法：
        adapter = LIVectorStoreAdapter(
            store=our_vector_store,
            collection="kb_rd_frontend",
        )
        index = VectorStoreIndex.from_vector_store(adapter)
    """

    # Pydantic 字段（BasePydanticVectorStore 要求的）
    stores_text: ClassVar[bool] = True
    is_embedding_query: ClassVar[bool] = True

    # 我们的依赖（Pydantic 不让直接存非字段实例，放到 model_config 之外的属性）
    _store: Any = None
    _collection: str = ""

    def __init__(self, store: VectorStore, collection: str, **kwargs: Any):
        # Pydantic model 初始化
        try:
            super().__init__(stores_text=True, is_embedding_query=True, **kwargs)
        except Exception:
            # 兜底
            try:
                super().__init__(**kwargs)
            except Exception:
                pass
        # 直接设值（绕过 Pydantic 字段校验，因为我们不在 ORM 模型里）
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_collection", collection)

    @property
    def client(self) -> Any:
        return self._store

    @property
    def collection_name(self) -> str:
        return self._collection

    # ---- LI 接口实现 ----

    async def aadd(self, nodes: list[Any], **kwargs: Any) -> list[str]:
        """LI 调用：把 nodes 写入我们的 vector store。"""
        li_types = _get_li_types()
        TextNode = li_types["TextNode"]  # noqa: N806

        records: list[VectorRecord] = []
        ids: list[str] = []
        for n in nodes:
            if isinstance(n, dict):
                node = TextNode(**n)
            else:
                node = n

            embedding = getattr(node, "embedding", None)
            if embedding is None:
                logger.warning("Node %s has no embedding, skipping", node.node_id)
                continue

            text = node.get_content() if hasattr(node, "get_content") else node.text
            metadata = dict(node.metadata or {})
            metadata.setdefault("node_id", node.node_id)
            metadata.setdefault("ref_doc_id", getattr(node, "ref_doc_id", ""))
            if not metadata.get("title"):
                metadata["title"] = ""

            records.append(VectorRecord(
                id=node.node_id,
                vector=embedding,
                text=text,
                metadata=metadata,
            ))
            ids.append(node.node_id)

        if records:
            await self._store.insert(self._collection, records)
        return ids

    def add(self, nodes: list[Any], **kwargs: Any) -> list[str]:
        return asyncio.run(self.aadd(nodes, **kwargs))

    async def adelete(self, ref_doc_id: str, **kwargs: Any) -> None:
        await self._store.delete(self._collection, [ref_doc_id])

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        asyncio.run(self.adelete(ref_doc_id, **kwargs))

    async def aquery(
        self,
        query: Any,
        custom_embedding: list[float] | None = None,
        **kwargs: Any,
    ) -> Any:
        li_types = _get_li_types()
        VectorStoreQueryResult = li_types["VectorStoreQueryResult"]  # noqa: N806

        from llama_index.core.schema import NodeWithScore

        if custom_embedding is not None:
            qvec = custom_embedding
        elif query.query_embedding is not None:
            qvec = list(query.query_embedding)
        else:
            raise ValueError(
                "query.query_embedding is None and no custom_embedding provided"
            )

        filter_dict = _li_filters_to_dict(query.filters)

        results: list[SearchResult] = await self._store.search(
            collection=self._collection,
            query_vector=qvec,
            top_k=query.similarity_top_k or 5,
            filter_expr=filter_dict,
        )

        nodes: list = []
        similarities: list[float] = []
        ids: list[str] = []
        for r in results:
            node = NodeWithScore(
                node=_text_node_from_search_result(r),
                score=r.score,
            )
            nodes.append(node)
            similarities.append(r.score)
            ids.append(r.id)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    def query(self, query: Any, **kwargs: Any) -> Any:
        return asyncio.run(self.aquery(query, **kwargs))


# ---- 辅助函数 ----

def _text_node_from_search_result(r: SearchResult) -> Any:
    from llama_index.core.schema import TextNode

    return TextNode(
        id_=r.id,
        text=r.text,
        metadata=r.metadata or {},
    )


def _li_filters_to_dict(filters: Any) -> dict | None:
    """LI MetadataFilters -> 我们 dict。"""
    if filters is None:
        return None
    try:
        from llama_index.core.vector_stores.types import MetadataFilter
    except ImportError:
        return None

    if isinstance(filters, MetadataFilter):
        return {_filter_key(filters): filters.value}

    if hasattr(filters, "filters"):
        out: dict = {}
        for f in filters.filters:
            if isinstance(f, MetadataFilter):
                out[_filter_key(f)] = f.value
            else:
                logger.warning("Unknown nested filter type: %s", type(f))
        return out

    return None


def _filter_key(f: Any) -> str:
    """把 LI MetadataFilter 还原为我们的 key（'field__op'）。"""
    op = str(f.operator).split(".")[-1].lower() if f.operator else "eq"
    return f"{f.key}__{op}"
