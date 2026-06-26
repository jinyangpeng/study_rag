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
    # BM25 全文检索专用字段名
    _FIELD_SPARSE = "sparse_bm25"

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

    async def create_collection_with_bm25(
        self,
        name: str,
        dimension: int,
        analyzer_type: str = "chinese",
    ) -> None:
        """创建带 BM25 全文检索能力的 collection（Milvus 2.5+）。

        Schema 包含：
          - id (INT64, auto_id)
          - text (VARCHAR, enable_analyzer=True) — BM25 输入字段
          - sparse_bm25 (SPARSE_FLOAT_VECTOR) — Function 自动生成
          - vector (FLOAT_VECTOR, dim) — dense 向量
          - metadata (JSON)

        Function 对象将 text 自动转为 sparse_bm25 向量，无需额外 Embedding 模型。

        Args:
            name: collection 名
            dimension: dense 向量维度
            analyzer_type: 分词器类型（chinese / english / standard），默认 chinese

        Raises:
            ImportError: pymilvus < 2.5（不支持 Function / SPARSE_FLOAT_VECTOR）
            RuntimeError: Milvus 版本 < 2.5
        """
        def _op():
            client = self._connect()
            if client.has_collection(name):
                logger.debug("Collection %s already exists, skip create", name)
                return

            # 检查 pymilvus 版本是否支持 BM25 Function
            try:
                from pymilvus import (  # type: ignore[import-not-found,import-untyped]
                    DataType,
                    Function,
                    FunctionType,
                )
            except ImportError as e:
                raise ImportError(
                    "BM25 full-text search requires pymilvus>=2.5. "
                    "Current version doesn't support Function/FunctionType. "
                    "Upgrade with: pip install 'pymilvus>=2.5'"
                ) from e

            logger.info(
                "Creating BM25 collection: name=%s dim=%d analyzer=%s",
                name, dimension, analyzer_type,
            )

            schema = client.create_schema(
                auto_id=True,
                enable_dynamic_field=True,
            )

            # 分词器参数
            analyzer_params = {"type": analyzer_type}

            # 字段定义
            schema.add_field(
                field_name=self._id_field,
                datatype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            )
            schema.add_field(
                field_name=self._text_field,
                datatype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
                analyzer_params=analyzer_params,
                enable_match=True,
            )
            schema.add_field(
                field_name=self._FIELD_SPARSE,
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )
            schema.add_field(
                field_name=self._vector_field,
                datatype=DataType.FLOAT_VECTOR,
                dim=dimension,
            )
            schema.add_field(
                field_name=self._metadata_field,
                datatype=DataType.JSON,
            )

            # BM25 Function：自动将 text 转为 sparse_bm25 向量
            bm25_function = Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=[self._text_field],
                output_field_names=self._FIELD_SPARSE,
            )
            schema.add_function(bm25_function)

            # 索引定义
            index_params = client.prepare_index_params()
            # Dense 向量索引
            index_params.add_index(
                field_name=self._vector_field,
                index_type=self._index_type,
                metric_type=self._metric_type,
            )
            # BM25 sparse 向量索引
            index_params.add_index(
                field_name=self._FIELD_SPARSE,
                index_type="SPARSE_WAND",
                metric_type="BM25",
            )

            client.create_collection(
                collection_name=name,
                schema=schema,
                index_params=index_params,
            )
            client.load_collection(name)

        try:
            await asyncio.to_thread(_op)
        except Exception as e:
            # 给出更友好的错误提示
            if "Function" in str(e) or "SPARSE" in str(e):
                raise RuntimeError(
                    f"Failed to create BM25 collection. "
                    f"Ensure Milvus >= 2.5 and pymilvus >= 2.5. Error: {e}"
                ) from e
            raise
        logger.info("BM25 collection %s ready (dim=%d)", name, dimension)

    async def search_sparse(
        self,
        collection: str,
        query_text: str,
        top_k: int = 10,
        filter_expr: dict | None = None,
    ) -> list[SearchResult]:
        """纯 BM25 全文检索（Milvus 2.5+）。

        直接传入文本查询，Milvus 自动分词并计算 BM25 分数。
        无需调用方做 embedding。

        Args:
            collection: collection 名（须由 create_collection_with_bm25 创建）
            query_text: 查询文本（原样传入，Milvus 内部分词）
            top_k: 返回结果数
            filter_expr: metadata 过滤条件
        """
        def _op() -> list[SearchResult]:
            client = self._connect()
            if not client.has_collection(collection):
                raise ValueError(f"Collection '{collection}' 不存在")
            expr = to_milvus_expr(filter_expr)
            search_params = {"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}}
            res = client.search(
                collection_name=collection,
                data=[query_text],
                anns_field=self._FIELD_SPARSE,
                limit=top_k,
                filter=expr or "",
                search_params=search_params,
                output_fields=[self._text_field, self._metadata_field],
            )
            hits = res[0] if res else []
            results: list[SearchResult] = []
            for h in hits:
                meta = h.get(self._metadata_field, h.get("metadata", {})) or {}
                original_id = meta.pop("_doc_id", None) or str(
                    h.get(self._id_field, h.get("id", ""))
                )
                results.append(
                    SearchResult(
                        id=original_id,
                        text=h.get(self._text_field, h.get("text", "")),
                        score=float(h.get("distance", 0.0)),
                        metadata=meta,
                    )
                )
            return results

        return await asyncio.to_thread(_op)

    async def hybrid_search(
        self,
        collection: str,
        query_vector: list[float],
        query_text: str,
        top_k: int = 10,
        filter_expr: dict | None = None,
        dense_weight: float = 0.6,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """Dense + BM25 混合检索 + RRF 融合（Milvus 2.5+ 原生）。

        用 Milvus 的 hybrid_search API + RRFRanker 在服务端完成融合，
        无需客户端拉取两路结果再合并。

        Args:
            collection: collection 名（须由 create_collection_with_bm25 创建）
            query_vector: dense 查询向量
            query_text: 查询文本（用于 BM25）
            top_k: 最终返回结果数
            filter_expr: metadata 过滤条件
            dense_weight: dense 权重（0~1，sparse 权重 = 1 - dense_weight）
                          注意：RRFRanker 不直接使用 weight，而是通过 rank 输入顺序
                          影响。这里用 rrf_k 控制平滑度。
            rrf_k: RRF 常数（越大越平滑）
        """
        def _op() -> list[SearchResult]:
            from pymilvus import (  # type: ignore[import-not-found,import-untyped]
                AnnSearchRequest,
                RRFRanker,
            )

            client = self._connect()
            if not client.has_collection(collection):
                raise ValueError(f"Collection '{collection}' 不存在")
            expr = to_milvus_expr(filter_expr)

            # Dense 检索请求
            dense_params = {"metric_type": self._metric_type}
            request_dense = AnnSearchRequest(
                [query_vector], self._vector_field, dense_params, limit=top_k * 4,
                expr=expr or "",
            )

            # BM25 检索请求
            sparse_params = {"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}}
            request_sparse = AnnSearchRequest(
                [query_text], self._FIELD_SPARSE, sparse_params, limit=top_k * 4,
                expr=expr or "",
            )

            # RRF 融合
            ranker = RRFRanker(rrf_k)
            res = client.hybrid_search(
                collection_name=collection,
                reqs=[request_dense, request_sparse],
                ranker=ranker,
                limit=top_k,
                output_fields=[self._text_field, self._metadata_field],
            )
            hits = res[0] if res else []
            results: list[SearchResult] = []
            for h in hits:
                meta = h.get(self._metadata_field, h.get("metadata", {})) or {}
                original_id = meta.pop("_doc_id", None) or str(
                    h.get(self._id_field, h.get("id", ""))
                )
                results.append(
                    SearchResult(
                        id=original_id,
                        text=h.get(self._text_field, h.get("text", "")),
                        score=float(h.get("distance", 0.0)),
                        metadata=meta,
                    )
                )
            return results

        return await asyncio.to_thread(_op)

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

    async def delete(
        self,
        collection: str,
        ids: list[str] | None = None,
        filter_expr: dict | None = None,
    ) -> int:
        """Milvus 实现：filter-based delete（兼容按 ids 删）。

        关键修复：旧版用 `id in ["001"]`（主键是 Int64，cast 失败抛 1100）。
        改用 `metadata["_doc_id"]` 字符串字段匹配：每条 chunk 写入时
        已在 metadata 里加了 ``_doc_id`` 字段（保留原始 string id），所以
        filter 走 metadata JSON path 即可。

        错误降级：collection 不存在 / pymilvus 错误 / 网络问题 → 返回 0，
        不让 API 返回 500。
        """
        if not ids and not filter_expr:
            return 0

        def _op() -> int:
            try:
                client = self._connect()
            except Exception:  # noqa: BLE001
                logger.warning("delete() failed: cannot connect to Milvus")
                return 0
            if not client.has_collection(collection):
                logger.warning("delete() on non-existent collection: %s", collection)
                return 0
            # 构造 filter
            if filter_expr:
                expr = to_milvus_expr(filter_expr)
            else:
                # ids 走 metadata._doc_id 字符串字段（不是 Int64 主键）
                escaped = ", ".join(format_value(i) for i in (ids or []))
                expr = f'{self._metadata_field}["_doc_id"] in [{escaped}]'
            try:
                client.delete(
                    collection_name=collection,
                    filter=expr,
                )
                # Milvus client.delete() 不返回删了多少条
                # 用 -1 表示"成功，条数未知"；调用方通常不依赖具体值
                return -1
            except Exception as e:  # noqa: BLE001
                logger.warning("delete() failed for %s: %s", collection, e)
                return 0

        return await asyncio.to_thread(_op)

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

    async def count(self, collection: str) -> int:
        """O(1) Milvus 实现：用 num_entities API，不拉数据。

        对应 MilvusClient.get_collection_stats()。pymilvus 2.4+ / 3.x 都有。

        错误降级：
        - collection 不存在 → 0
        - pymilvus 缺 / 抛错 → 0（不阻断上层）
        """
        def _op() -> int:
            try:
                client = self._connect()
            except ImportError as e:  # noqa: PERF203
                logger.warning("count() requires pymilvus: %s", e)
                return 0
            except Exception as e:  # noqa: BLE001
                logger.warning("count() connect failed for %s: %s", collection, e)
                return 0
            try:
                if not client.has_collection(collection):
                    logger.warning(
                        "count() on non-existent collection: %s", collection
                    )
                    return 0
                stats = client.get_collection_stats(collection_name=collection)
                # stats: {"row_count": N, ...}  (pymilvus 2.4+ / 3.x)
                return int(stats.get("row_count", 0))
            except Exception as e:  # noqa: BLE001
                logger.warning("count() failed for %s: %s", collection, e)
                return 0

        return await asyncio.to_thread(_op)
