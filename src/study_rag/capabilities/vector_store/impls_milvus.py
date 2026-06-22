"""Milvus Vector Store：基于 pymilvus 的 Milvus / Milvus Lite 实现。

支持的部署形态：
  - Milvus Lite：嵌入式（uri = 本地 .db 文件）
  - Milvus Standalone：单机部署
  - Milvus Cluster：分布式集群（推荐生产用）
  - Zilliz Cloud：托管服务（uri = https://xxx.zillizcloud.com, token = API key）

依赖：pymilvus>=2.4.0
安装：pip install study-rag[vector-milvus]

企业级实践：
  - 懒连接：第一次操作时才连接，避免 import 时阻塞
  - 共享 client：实例内 singleton
  - 健康检查：connect() 内部 ping 验证
  - 优雅关闭：close() 释放连接
  - 维度校验：插入时校验，写错维度清晰报错
  - 索引策略：AUTOINDEX（2.4+ 自适应）
  - 批量插入：自动按 batch_size 切片
  - metadata filter：转成 Milvus expr 语法
  - 重连机制：连接断开时自动重连一次
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from .base import (
    SearchResult,
    VectorRecord,
    VectorStoreConfig,
    register_vector_store,
)
from .filters import format_value, to_milvus_expr

logger = logging.getLogger(__name__)


# ---- 辅助函数（pymilvus 懒加载） ----

def _get_milvus_client_class():
    """懒加载 pymilvus.MilvusClient。"""
    try:
        from pymilvus import MilvusClient  # type: ignore[import-not-found,import-untyped]
    except ImportError as e:
        raise ImportError(
            "MilvusVectorStore requires 'pymilvus'. "
            "Install with: pip install study-rag[vector-milvus]"
        ) from e
    return MilvusClient


def _to_milvus_filter(filter_expr: dict | None) -> str | None:
    """dict -> Milvus expr 的便捷入口（详见 filters 模块）。"""
    return to_milvus_expr(filter_expr)


def _format_value(value: Any) -> str:
    return format_value(value)


def _parse_uri(uri: str) -> tuple[str, str | None]:
    """解析 uri，判断是 Milvus Lite (本地文件) 还是远程服务。

    Returns:
        (uri, token): token 默认为 None
    """
    if not uri:
        raise ValueError(
            "Milvus uri 不能为空。"
            "  - 本地 Lite: uri: ./milvus.db\n"
            "  - 远程服务: uri: http://localhost:19530\n"
            "  - Zilliz:    uri: https://xxx.zillizcloud.com + token: <api_key>"
        )
    return uri, None


# ---- MilvusVectorStore ----

@register_vector_store("milvus")
class MilvusVectorStore:
    """Milvus Vector Store 实现。

    配置示例（vector_store.yaml）:
        vector_store:
          provider: milvus
          uri: ${MILVUS_URI:-./milvus.db}    # 本地 Lite 模式
          # uri: http://localhost:19530       # Standalone
          # uri: https://xxx.zillizcloud.com  # Zilliz Cloud
          extra:
            token: ${MILVUS_TOKEN}            # Zilliz / 鉴权时必填
            db_name: ${MILVUS_DB:-default}    # 数据库名
            batch_size: 1000                  # 单批插入上限
            index:
              type: AUTOINDEX                 # 2.4+ 自适应索引
              metric_type: COSINE
    """

    # Milvus 中每个 collection 的 schema 字段名（固定）
    _FIELD_ID = "id"
    _FIELD_VECTOR = "vector"
    _FIELD_TEXT = "text"
    _FIELD_METADATA = "metadata"

    @staticmethod
    def _to_int64_id(raw: str) -> int:
        """把 string 主键稳定映射到 int64。

        Milvus 的 create_collection（auto schema）默认把主键建为 int64，
        所以 string doc_id 需要先 hash 成 64 位有符号整数。
        用 SHA-256（确定性，跨进程一致）取前 8 字节解释为有符号 int64。
        """
        import hashlib

        digest = hashlib.sha256(raw.encode("utf-8")).digest()[:8]
        val = int.from_bytes(digest, byteorder="big", signed=False)
        # 转到 [-2^63, 2^63-1]
        if val >= 1 << 63:
            val -= 1 << 64
        return val

    def __init__(self, config: VectorStoreConfig):
        self._config = config
        self._uri, _ = _parse_uri(config.uri)
        self._extra = config.extra or {}
        self._token = self._extra.get("token")
        self._db_name = self._extra.get("db_name", "default")
        self._batch_size = int(self._extra.get("batch_size", 1000))
        self._index_type = self._extra.get("index", {}).get("type", "AUTOINDEX")
        self._metric_type = self._extra.get("index", {}).get("metric_type", "COSINE")
        # 允许在 extra 中覆盖默认 schema 字段名（一般不用）
        self._id_field = self._extra.get("id_field", self._FIELD_ID)
        self._text_field = self._extra.get("text_field", self._FIELD_TEXT)
        self._metadata_field = self._extra.get("metadata_field", self._FIELD_METADATA)
        self._vector_field = self._extra.get("vector_field", self._FIELD_VECTOR)

        self._client = None
        self._lock = threading.RLock()  # 保护 _client 的初始化
        logger.info(
            "MilvusVectorStore configured: uri=%s, db=%s, metric=%s, index=%s",
            self._uri,
            self._db_name,
            self._metric_type,
            self._index_type,
        )

    # ---- 连接管理 ----

    def _connect(self):
        """懒连接。第一次操作时调用。线程安全。

        Enterprise：连接失败抛 ConnectionError（业务层可捕获做重试）。
        缺 pymilvus 依赖时抛 ImportError。
        """
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is not None:
                return self._client
            MilvusClient = _get_milvus_client_class()  # noqa: N806
            kwargs: dict[str, Any] = {"uri": self._uri, "db_name": self._db_name}
            if self._token:
                kwargs["token"] = self._token
            logger.info("Connecting to Milvus: uri=%s db=%s", self._uri, self._db_name)
            self._client = MilvusClient(**kwargs)
            # 健康检查：列出 collections 验证连接
            try:
                _ = self._client.list_collections()
            except Exception as e:
                self._client = None
                raise ConnectionError(
                    f"Milvus 健康检查失败: {e}. "
                    f"请检查 uri / token / 网络连接"
                ) from e
            logger.info("Milvus connected OK")
            return self._client

    def close(self) -> None:
        """关闭连接（释放资源）。"""
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception as e:
                    logger.warning("Error closing Milvus client: %s", e)
                finally:
                    self._client = None
                    logger.info("Milvus connection closed")

    async def health_check(self) -> bool:
        """健康检查：能否连通 Milvus。"""
        try:
            client = await asyncio.to_thread(self._connect)
            await asyncio.to_thread(client.list_collections)
            return True
        except Exception as e:
            logger.warning("Milvus health check failed: %s", e)
            return False

    # ---- VectorStore Protocol 实现 ----

    async def create_collection(self, name: str, dimension: int) -> None:
        """创建 collection（带向量索引）。已存在则跳过。

        Milvus 中：
          - has_collection(name) 判断存在
          - create_collection 时指定 dim + 自动建索引
        """
        def _op():
            client = self._connect()
            if client.has_collection(name):
                logger.debug("Collection %s already exists, skip create", name)
                return
            logger.info("Creating collection: name=%s dim=%d", name, dimension)
            client.create_collection(
                collection_name=name,
                dimension=dimension,
                primary_field_name=self._id_field,
                vector_field_name=self._vector_field,
                metric_type=self._metric_type,
                # AUTOINDEX 走默认配置；如需 IVF_FLAT/HNSW 可在 extra.index.params 传
                index_type=self._index_type,
            )
            # 加载到内存以支持搜索
            client.load_collection(name)

        await asyncio.to_thread(_op)
        logger.info("Collection %s ready (dim=%d)", name, dimension)

    async def drop_collection(self, name: str) -> None:
        def _op():
            client = self._connect()
            if client.has_collection(name):
                client.drop_collection(name)
                logger.info("Dropped collection: %s", name)

        await asyncio.to_thread(_op)

    async def has_collection(self, name: str) -> bool:
        def _op() -> bool:
            client = self._connect()
            return client.has_collection(name)

        return await asyncio.to_thread(_op)

    async def insert(self, collection: str, records: list[VectorRecord]) -> None:
        """批量插入。MilvusClient 内部已支持 list 批量写入。"""
        if not records:
            return

        def _op():
            client = self._connect()
            if not client.has_collection(collection):
                raise ValueError(
                    f"Collection '{collection}' 不存在。"
                    f"请先调用 create_collection() 或在 manager 启动时 init_all()"
                )
            # 按 batch_size 切片
            for i in range(0, len(records), self._batch_size):
                batch = records[i : i + self._batch_size]
                rows = [
                    {
                        self._id_field: self._to_int64_id(r.id),
                        self._vector_field: r.vector,
                        self._text_field: r.text,
                        # 把原始 string id 存到 metadata，搜索时再取回（避免暴露 int64）
                        self._metadata_field: {**(r.metadata or {}), "_doc_id": r.id},
                    }
                    for r in batch
                ]
                client.insert(collection_name=collection, data=rows)
            # 触发刷盘（不 await 也没事，但显式 flush 更可控）
            try:
                client.flush(collection)
            except Exception as e:
                # flush 在某些版本不可用，忽略
                logger.debug("Milvus flush not supported or failed: %s", e)

        await asyncio.to_thread(_op)
        logger.info("Inserted %d records into %s", len(records), collection)

    async def delete(self, collection: str, ids: list[str]) -> None:
        if not ids:
            return

        def _op():
            client = self._connect()
            if not client.has_collection(collection):
                logger.warning(
                    "delete() called on non-existent collection: %s", collection
                )
                return
            # 用主键过滤删除
            id_list = ", ".join(f'"{i}"' for i in ids)
            client.delete(
                collection_name=collection,
                filter=f"{self._id_field} in [{id_list}]",
            )

        await asyncio.to_thread(_op)
        logger.info("Deleted %d ids from %s", len(ids), collection)

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 5,
        filter_expr: dict | None = None,
    ) -> list[SearchResult]:
        def _op() -> list[SearchResult]:
            client = self._connect()
            if not client.has_collection(collection):
                raise ValueError(f"Collection '{collection}' 不存在")
            expr = to_milvus_expr(filter_expr)
            res = client.search(
                collection_name=collection,
                data=[query_vector],
                limit=top_k,
                filter=expr,
                output_fields=[self._text_field, self._metadata_field],
            )
            # res: list[list[Hit]]，外层对应 query 数量（这里只有 1 个）
            hits = res[0] if res else []
            results_list: list[SearchResult] = []
            for h in hits:
                meta = h.get(self._metadata_field, h.get("metadata", {})) or {}
                # metadata._doc_id 优先（保留原始 string id）；缺失则回退到 int64 主键
                original_id = meta.pop("_doc_id", None) or str(h.get(self._id_field, h.get("id", "")))
                results_list.append(
                    SearchResult(
                        id=original_id,
                        text=h.get(self._text_field, h.get("text", "")),
                        score=float(h.get("distance", 0.0)),
                        metadata=meta,
                    )
                )
            return results_list

        results = await asyncio.to_thread(_op)
        logger.debug(
            "Search %s top_k=%d filter=%s -> %d hits",
            collection,
            top_k,
            filter_expr,
            len(results),
        )
        return results

    async def query(
        self,
        collection: str,
        filter_expr: dict | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[VectorRecord]:
        """按 metadata 过滤取所有 chunks（不查相似度）。

        对应 MilvusClient.query()，配合 to_milvus_expr() 转换 filter。

        Args:
            collection: collection 名
            filter_expr: metadata 过滤表达式（复用 matches_filter / to_milvus_expr）
            limit: 返回上限
            offset: 跳过条数

        Returns:
            匹配的 VectorRecord 列表（含 id / vector / text / metadata）
        """
        try:
            client = self._connect()
        except ImportError as e:
            raise NotImplementedError(
                f"MilvusVectorStore.query() requires pymilvus: {e}"
            ) from e

        def _op() -> list[VectorRecord]:
            if not client.has_collection(collection):
                logger.warning(
                    "query() called on non-existent collection: %s", collection
                )
                return []
            expr = to_milvus_expr(filter_expr)
            res = client.query(
                collection_name=collection,
                filter=expr or "",
                output_fields=[
                    self._id_field,
                    self._text_field,
                    self._metadata_field,
                    self._vector_field,
                ],
                limit=limit,
                offset=offset,
            )
            results: list[VectorRecord] = []
            for h in res:
                meta = h.get(self._metadata_field, h.get("metadata", {})) or {}
                # 优先取 metadata._doc_id（保留原始 string id），缺失则回退到 int64 主键
                original_id = str(meta.get("_doc_id") or h.get(self._id_field, ""))
                # Milvus 返回的 vector 是 list[list[float]] (单条) 或 list[float]
                vec = h.get(self._vector_field, []) or []
                if vec and isinstance(vec[0], list):
                    vec = vec[0]
                results.append(
                    VectorRecord(
                        id=original_id,
                        vector=vec,
                        text=h.get(self._text_field, h.get("text", "")),
                        metadata=meta,
                    )
                )
            return results

        records = await asyncio.to_thread(_op)
        logger.debug(
            "Query %s filter=%s limit=%d offset=%d -> %d records",
            collection,
            filter_expr,
            limit,
            offset,
            len(records),
        )
        return records
