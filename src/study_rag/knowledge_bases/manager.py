"""知识库管理：文档增删改查 + 关联向量库。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..capabilities.embedding import Embedder, EmbeddingConfig, create_embedder
from ..capabilities.reranker import Reranker, RerankerConfig, create_reranker
from ..capabilities.retrieval import (
    RetrievalConfig,
    RetrievalEngine,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalStrategy,
    create_retrieval_engine,
)
from ..capabilities.vector_store import (
    VectorRecord,
    VectorStore,
    VectorStoreConfig,
    create_vector_store,
)
from ..observability.logging import get_logger
from ..settings import AppPaths
from .models import (
    ChunkInfo,
    DocumentCreate,
    DocumentMeta,
    KnowledgeBaseSummary,
)
from .registry import KnowledgeBaseRegistry, get_registry

if TYPE_CHECKING:
    from ..capabilities.llamaindex import LlamaIndexRetrievalEngine

logger = get_logger(__name__)


class ComponentUnavailableError(Exception):
    """组件（embedder / reranker / vector store）未加载或不可用。

    区别于 KBNotFoundError / InvalidParameterError：
    - KBNotFoundError: KB 标识不存在或 disabled
    - ComponentUnavailableError: KB 存在但其依赖的 embedder/reranker 因
      依赖缺失或配置问题未加载
    - InvalidParameterError: 参数非法（query 空、top_k 越界等）

    抛出位置：manager.get_embedder / get_reranker 等。
    上层（API / MCP Tool）应捕获并返回 4xx 而不是 5xx。
    """

    def __init__(
        self,
        component: str,
        name: str,
        kb_id: str | None = None,
        hint: str = "",
    ) -> None:
        self.component = component
        self.name = name
        self.kb_id = kb_id
        msg = f"{component} '{name}' is unavailable"
        if kb_id:
            msg += f" (referenced by KB '{kb_id}')"
        if hint:
            msg += f". {hint}"
        super().__init__(msg)


class KnowledgeBaseManager:
    """知识库管理器：负责 KB 的创建、文档的增删改查。

    设计：
      - 内存存储 DocumentMeta（便于 get_document）
      - 向量库存储 Document Vector（用于检索）
      - 单实例多 KB 共享一个 VectorStore
      - 多个 embedding 实例（按名字复用，相同 name 共享一个实例）
      - 多个 reranker 实例（按名字复用，相同 name 共享一个实例）
    """

    def __init__(
        self,
        registry: KnowledgeBaseRegistry,
        vector_store: VectorStore,
        embedders: dict[str, Embedder],
        rerankers: dict[str, Reranker] | None = None,
        docs_index_path: Path | None = None,
    ):
        self._registry = registry
        self._vector_store = vector_store
        self._embedders = embedders
        self._rerankers = rerankers or {}
        # kb_id -> {doc_id -> DocumentMeta}
        self._docs: dict[str, dict[str, DocumentMeta]] = {}
        self._lock = asyncio.Lock()
        # 持久化路径：增删文档时同步写盘，重启后从磁盘恢复
        self._docs_index_path = docs_index_path
        if self._docs_index_path is not None and self._docs_index_path.exists():
            try:
                self._load_docs_from_disk()
            except Exception as e:  # noqa: BLE001
                logger.warning("docs_index_load_failed", error=str(e))

    def _load_docs_from_disk(self) -> None:
        """从 JSON 索引加载 DocumentMeta（运行时持久化）。"""
        import json

        if self._docs_index_path is None or not self._docs_index_path.exists():
            return
        try:
            raw = json.loads(self._docs_index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("docs_index_parse_failed", path=str(self._docs_index_path), error=str(e))
            return
        for kb_id, docs in raw.items():
            self._docs.setdefault(kb_id, {})
            for doc_id, meta_dict in docs.items():
                try:
                    self._docs[kb_id][doc_id] = DocumentMeta(**meta_dict)
                except Exception as e:  # noqa: BLE001
                    logger.warning("docs_index_record_invalid", kb_id=kb_id, doc_id=doc_id, error=str(e))
        logger.info("docs_index_loaded", count=sum(len(d) for d in self._docs.values()))

    def _save_docs_to_disk(self) -> None:
        """把当前 _docs 字典写盘（增删文档后调用）。"""
        import json

        if self._docs_index_path is None:
            return
        try:
            self._docs_index_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                kb_id: {doc_id: meta.model_dump(mode="json") for doc_id, meta in docs.items()}
                for kb_id, docs in self._docs.items()
            }
            self._docs_index_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("docs_index_save_failed", path=str(self._docs_index_path), error=str(e))

    # ---- KB 级别 ----

    async def init_kb(self, kb_id: str) -> bool:
        """初始化一个知识库（创建 collection）。

        若 KB 引用的 embedder 未加载（缺依赖被跳过），跳过该 KB 不抛错。
        这样部分 provider 不可用时服务仍能起来（用 in-memory fallback 或后续修复）。

        collection schema 与检索策略**解耦**：始终优先创建带 BM25 的 collection
        （Milvus 2.5+），使任意策略（dense / sparse / hybrid / sparse_milvus /
        hybrid_milvus）均可 per-request 使用，无需因切换策略重建 collection。
        不支持 BM25 时降级为普通 dense collection。

        Returns:
            True  - collection 创建成功
            False - 跳过（embedder 缺失）
        """
        cfg = self._registry.get_required(kb_id)
        embedder = self._embedders.get(cfg.embedding)
        if embedder is None:
            logger.warning(
                "kb_skip_init_embedder_missing",
                kb_id=kb_id,
                embedder=cfg.embedding,
                hint=(
                    f"embedder '{cfg.embedding}' not loaded (missing dependency?). "
                    "Skipping collection creation for this KB."
                ),
            )
            return False

        # 合并策略参数获取 analyzer_type（BM25 分词器，不影响 dense 检索）
        global_cfg = self.get_retrieval_config()
        effective_strategy = _resolve_strategy(cfg, global_cfg)
        merged = _merge_strategy_params(
            effective_strategy, global_cfg, cfg.retrieval_params, None
        )
        analyzer_type = merged.get("analyzer_type", "chinese")

        await self._ensure_collection(
            name=cfg.collection,
            dimension=embedder.dimension,
            analyzer_type=analyzer_type,
        )
        self._docs.setdefault(kb_id, {})
        return True

    async def _ensure_collection(
        self,
        name: str,
        dimension: int,
        analyzer_type: str = "chinese",
    ) -> None:
        """创建 collection，优先 BM25 schema，不支持时降级普通 schema。

        - 已存在则跳过（不破坏已有数据）
        - vector store 支持 create_collection_with_bm25 时优先用 BM25 schema
          （Milvus 2.5+），使 collection 同时具备 dense + 全文检索能力
        - 不支持 BM25（旧版 Milvus / pymilvus / 非 milvus provider）降级普通 collection
        """
        if await self._vector_store.has_collection(name):
            logger.debug("collection_exists_skip_create", collection=name)
            return

        if hasattr(self._vector_store, "create_collection_with_bm25"):
            try:
                await self._vector_store.create_collection_with_bm25(
                    name=name,
                    dimension=dimension,
                    analyzer_type=analyzer_type,
                )
                logger.info(
                    "collection_created_bm25",
                    collection=name,
                    dim=dimension,
                    analyzer=analyzer_type,
                )
                return
            except (ImportError, RuntimeError, AttributeError) as e:
                logger.warning(
                    "bm25_collection_unavailable_fallback_dense",
                    collection=name,
                    error=str(e),
                    hint="Ensure Milvus >= 2.5 and pymilvus >= 2.5 for BM25 support",
                )

        await self._vector_store.create_collection(name=name, dimension=dimension)
        logger.info("collection_created_dense", collection=name, dim=dimension)

    async def init_all(self) -> None:
        """初始化所有知识库。"""
        succeeded = 0
        skipped = 0
        failed = 0
        for cfg in self._registry.all_cfgs(enabled_only=True):
            try:
                if await self.init_kb(cfg.kb_id):
                    succeeded += 1
                else:
                    skipped += 1
            except Exception as e:  # noqa: BLE001
                # 单个 KB 失败不影响其他 KB 启动
                logger.error(
                    "kb_init_failed",
                    kb_id=cfg.kb_id,
                    error=str(e),
                    exc_info=True,
                )
                failed += 1
        logger.info(
            "kb_init_summary",
            succeeded=succeeded,
            skipped=skipped,
            failed=failed,
        )

    async def list_summaries(self) -> list[KnowledgeBaseSummary]:
        """列出所有 KB 的摘要。

        chunk_count 通过 vector store 的 O(1) count() 拿，不拉数据。
        """
        vs_provider = (
            self._vector_store.__class__.__name__
            if hasattr(self._vector_store, "__class__")
            else None
        )
        # 简单映射：InMemoryVectorStore -> "inmemory"
        if vs_provider == "InMemoryVectorStore":
            vs_provider = "inmemory"
        elif vs_provider and vs_provider.endswith("VectorStore"):
            vs_provider = vs_provider[: -len("VectorStore")].lower()
        return [
            KnowledgeBaseSummary(
                kb_id=cfg.kb_id,
                name=cfg.name,
                description=cfg.description,
                department=cfg.department,
                enabled=cfg.enabled,
                document_count=len(self._docs.get(cfg.kb_id, {})),
                chunk_count=await self.get_total_chunk_count(cfg.kb_id),
                embedder=cfg.embedding,
                reranker=cfg.reranker,
                retrieval_strategy=cfg.retrieval_strategy,
                vector_store=vs_provider,
                collection=cfg.collection,
            )
            for cfg in self._registry.all_cfgs()
        ]

    async def get_summary(self, kb_id: str) -> KnowledgeBaseSummary | None:
        """获取单个 KB 的摘要（含 chunk_count）。

        chunk_count 通过 vector store 的 O(1) count() 拿。
        """
        cfg = self._registry.get(kb_id)
        if cfg is None:
            return None
        vs_provider = (
            self._vector_store.__class__.__name__
            if hasattr(self._vector_store, "__class__")
            else None
        )
        if vs_provider == "InMemoryVectorStore":
            vs_provider = "inmemory"
        elif vs_provider and vs_provider.endswith("VectorStore"):
            vs_provider = vs_provider[: -len("VectorStore")].lower()
        return KnowledgeBaseSummary(
            kb_id=cfg.kb_id,
            name=cfg.name,
            description=cfg.description,
            department=cfg.department,
            enabled=cfg.enabled,
            document_count=len(self._docs.get(kb_id, {})),
            chunk_count=await self.get_total_chunk_count(kb_id),
            embedder=cfg.embedding,
            reranker=cfg.reranker,
            retrieval_strategy=cfg.retrieval_strategy,
            vector_store=vs_provider,
            collection=cfg.collection,
        )

    async def recreate_collection(self, kb_id: str) -> dict[str, Any]:
        """重建 KB 的 collection（升级为 BM25 schema），保留已有向量数据。

        适用场景：旧 collection 是 dense-only schema（init_kb 解耦前创建），
        需要用 sparse_milvus / hybrid_milvus 策略检索时，通过本方法升级 schema。

        流程：
          1. 从旧 collection 分页拉取所有 chunks（含 vector / text / metadata）
          2. drop 旧 collection
          3. 创建新 collection（优先 BM25 schema，不支持时降级普通）
          4. 重新 insert 所有 chunks（向量已存在，无需重新 embedding）
          5. 失效检索引擎缓存

        不会修改 _docs 索引（DocumentMeta 不受影响）。
        """
        cfg = self._registry.get_required(kb_id)
        collection = cfg.collection
        embedder = self.get_embedder(kb_id)

        # 1. 拉取旧数据（分页，避免单次拉取过大）
        existing = await self._vector_store.has_collection(collection)
        records: list[VectorRecord] = []
        if existing:
            offset = 0
            batch = 500
            while True:
                batch_records = await self._vector_store.query(
                    collection, limit=batch, offset=offset
                )
                if not batch_records:
                    break
                records.extend(batch_records)
                if len(batch_records) < batch:
                    break
                offset += batch

        # 2. drop 旧 collection
        if existing:
            await self._vector_store.drop_collection(collection)

        # 3. 创建新 collection（优先 BM25）
        bm25_enabled = False
        if hasattr(self._vector_store, "create_collection_with_bm25"):
            try:
                await self._vector_store.create_collection_with_bm25(
                    name=collection,
                    dimension=embedder.dimension,
                    analyzer_type="chinese",
                )
                bm25_enabled = True
            except (ImportError, RuntimeError, AttributeError) as e:
                logger.warning(
                    "recreate_bm25_fallback_dense",
                    collection=collection,
                    error=str(e),
                )
                await self._vector_store.create_collection(
                    name=collection, dimension=embedder.dimension
                )
        else:
            await self._vector_store.create_collection(
                name=collection, dimension=embedder.dimension
            )

        # 4. 重新 insert（向量已存在，无需重新 embedding）
        if records:
            await self._vector_store.insert(collection, records)

        # 5. 失效检索引擎缓存（schema 变了，缓存的引擎需要重建）
        await self._invalidate_retrieval_engines(kb_id)

        logger.info(
            "collection_recreated",
            kb_id=kb_id,
            collection=collection,
            migrated_chunks=len(records),
            bm25_enabled=bm25_enabled,
        )
        return {
            "kb_id": kb_id,
            "collection": collection,
            "migrated_chunks": len(records),
            "bm25_enabled": bm25_enabled,
        }

    # ---- Document 级别 ----

    async def add_document(self, doc: DocumentCreate) -> DocumentMeta:
        """添加文档到指定知识库。

        简单实现：整篇文档作为一个 chunk。
        后续可接入 LlamaIndex 的 NodeParser 做更细的切分。
        """
        cfg = self._registry.get_required(doc.kb_id)
        # 走 get_embedder：缺依赖时抛 ComponentUnavailableError（4xx）而不是 KeyError（5xx）
        embedder = self.get_embedder(doc.kb_id)

        # 1. Embedding
        vector = await embedder.embed_query(doc.content)

        # 2. 写入向量库
        # 注意：metadata 必须包含 "doc_id"，否则 delete_document 时
        # filter `metadata["doc_id"] == "X"` 匹配 0 条 → Milvus 残留。
        # add_document_chunked / pipeline 都有 doc_id，这里是历史 bug 修复。
        record = VectorRecord(
            id=doc.doc_id,
            vector=vector,
            text=doc.content,
            metadata={
                "title": doc.title,
                "source": doc.source or "",
                "doc_id": doc.doc_id,
            },
        )
        await self._vector_store.insert(cfg.collection, [record])

        # 3. 保存 DocumentMeta（含内容）
        meta = DocumentMeta(
            doc_id=doc.doc_id,
            kb_id=doc.kb_id,
            title=doc.title,
            source=doc.source,
            content=doc.content,
            metadata=doc.metadata,
            # whole 模式：1 个 chunk，正文字符数 = content 长度
            chunk_count=1,
            char_count=len(doc.content),
            parser="whole",
        )
        async with self._lock:
            self._docs.setdefault(doc.kb_id, {})[doc.doc_id] = meta
        self._save_docs_to_disk()
        # 文档变更 → 失效检索引擎缓存（Sparse 的 BM25 索引需重建）
        await self._invalidate_retrieval_engines(doc.kb_id)
        return meta

    def get_document(self, kb_id: str, doc_id: str) -> DocumentMeta | None:
        return self._docs.get(kb_id, {}).get(doc_id)

    async def delete_document(self, kb_id: str, doc_id: str) -> bool:
        """删除文档。

        从 vector store 删除该 doc_id 对应的所有 chunks（用 filter_expr），
        同时从内存 _docs 移除 metadata。
        vector store 错误不影响内存删除（最坏情况：向量库有残留，UI 看到 0 文档
        但向量库还有 chunk；下次重建 collection 时会清掉）。
        """
        cfg = self._registry.get(kb_id)
        if cfg is None:
            return False
        # 1. vector store 删 chunks（用 filter，失败不抛）
        try:
            n = await self._vector_store.delete(
                cfg.collection,
                filter_expr={"doc_id": doc_id},
            )
            logger.info(
                "Deleted chunks from %s for doc_id=%s: %s",
                cfg.collection,
                doc_id,
                n,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "vector_store_delete_failed",
                kb_id=kb_id,
                doc_id=doc_id,
                error=str(e),
            )
        # 2. 内存 _docs 删 meta（必须成功）
        async with self._lock:
            existed = self._docs.get(kb_id, {}).pop(doc_id, None) is not None
        if existed:
            self._save_docs_to_disk()
        # 文档变更 → 失效检索引擎缓存（Sparse 的 BM25 索引需重建）
        if existed:
            await self._invalidate_retrieval_engines(kb_id)
        return existed

    def list_documents(self, kb_id: str) -> list[DocumentMeta]:
        return list(self._docs.get(kb_id, {}).values())

    async def list_chunks(
        self,
        kb_id: str,
        doc_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ChunkInfo]:
        """获取文档的所有 chunks（从向量库查）。

        Args:
            kb_id: 知识库 ID（不存在 → KeyError）
            doc_id: 文档 ID
            limit: 返回的 chunk 数上限
            offset: 分页偏移

        Returns:
            按 chunk_index 升序排列的 ChunkInfo 列表
        """
        cfg = self._registry.get(kb_id)
        if cfg is None:
            raise KeyError(f"Unknown kb_id: {kb_id}")
        # 拿足够多记录后按 chunk_index 排序，再分页
        # 这样 pagination 对调用方来说是稳定的（按切块顺序）
        records = await self._vector_store.query(
            cfg.collection,
            filter_expr={"doc_id": doc_id},
            limit=limit + offset,
            offset=0,
        )
        records.sort(key=lambda r: r.metadata.get("chunk_index", 0))
        page = records[offset:offset + limit]
        return [
            ChunkInfo(
                chunk_id=r.id,
                chunk_index=r.metadata.get("chunk_index", 0),
                text=r.text,
                char_count=len(r.text),
                metadata=r.metadata,
            )
            for r in page
        ]

    async def get_chunk_count(self, kb_id: str, doc_id: str) -> int:
        """获取文档的 chunk 总数。"""
        cfg = self._registry.get(kb_id)
        if cfg is None:
            raise KeyError(f"Unknown kb_id: {kb_id}")
        records = await self._vector_store.query(
            cfg.collection,
            filter_expr={"doc_id": doc_id},
            limit=10000,  # 业务上限，文档很少有 > 10k chunks
        )
        return len(records)

    async def get_doc_total_chars(self, kb_id: str, doc_id: str) -> int:
        """获取文档在向量库里的所有 chunks 的总字符数。

        比 DocumentMeta.char_count 准：不受 content 截断（pipeline 截 1000 字符）
        或历史 enrich bug 影响。等于把所有 chunk text 拼起来的长度。
        """
        cfg = self._registry.get(kb_id)
        if cfg is None:
            raise KeyError(f"Unknown kb_id: {kb_id}")
        records = await self._vector_store.query(
            cfg.collection,
            filter_expr={"doc_id": doc_id},
            limit=10000,
        )
        return sum(len(r.text) for r in records)

    async def get_doc_parser(self, kb_id: str, doc_id: str) -> str | None:
        """从 vector store 的 chunks metadata 里取真实的 parser 标签。

        老的 doc DocumentMeta.parser 可能是 None（旧代码漏存），或 fallback 到
        strategy 名（之前的 bug），或正确命名。统一以 vector store 为准。
        """
        cfg = self._registry.get(kb_id)
        if cfg is None:
            raise KeyError(f"Unknown kb_id: {kb_id}")
        records = await self._vector_store.query(
            cfg.collection,
            filter_expr={"doc_id": doc_id},
            limit=1,  # 只要一个 chunk 的 metadata 就够
        )
        if not records:
            return None
        return records[0].metadata.get("parser")

    async def get_total_chunk_count(self, kb_id: str) -> int:
        """获取 KB 整个 collection 的总 chunk 数。

        用 O(1) 的 count() API，不拉数据。
        collection 不存在 / count 失败 → 返回 0（不抛错）。
        """
        cfg = self._registry.get(kb_id)
        if cfg is None:
            return 0
        try:
            return await self._vector_store.count(cfg.collection)
        except Exception:  # noqa: BLE001
            logger.warning("get_total_chunk_count_failed", kb_id=kb_id)
            return 0

    # ---- 检索（被 MCP Tool 调用） ----

    def get_embedder(self, kb_id: str) -> Embedder:
        """获取 KB 对应的 embedder 实例。

        缺依赖时抛 ComponentUnavailableError（HTTP 503/400 友好），不再让 KeyError 漏出。
        """
        cfg = self._registry.get_required(kb_id)
        embedder = self._embedders.get(cfg.embedding)
        if embedder is None:
            available = list(self._embedders.keys())
            raise ComponentUnavailableError(
                component="embedder",
                name=cfg.embedding,
                kb_id=kb_id,
                hint=(
                    f"Embedder '{cfg.embedding}' not loaded (dependency missing or "
                    f"config error). Available: {available or '[]'}. "
                    f"Install the matching extra (e.g. study-rag[embedding-bge]) "
                    f"or change this KB's embedding to a loaded provider."
                ),
            )
        return embedder

    def get_vector_store(self) -> VectorStore:
        """获取共享的 vector store 实例。"""
        return self._vector_store

    def get_reranker(self, name: str) -> Reranker:
        """按配置名获取 reranker 实例（未加载时按需实例化并缓存）。

        检索调试时可指定 reranker.yaml 中任意已配置的 reranker，
        即使没有被 KB 引用也会即时实例化（保持启动时按需加载的设计）。
        """
        if name in self._rerankers:
            return self._rerankers[name]

        # 按需加载：从 reranker.yaml 读取配置并实例化
        cfg = _load_single_reranker_config(name)
        if cfg is None:
            raise ComponentUnavailableError(
                component="reranker",
                name=name,
                hint=(
                    f"Reranker '{name}' not found in reranker.yaml. "
                    f"Loaded: {list(self._rerankers.keys()) or '[]'}."
                ),
            )
        try:
            instance = create_reranker(cfg)
        except (ImportError, ValueError) as e:
            raise ComponentUnavailableError(
                component="reranker",
                name=name,
                hint=f"Reranker '{name}' failed to initialize: {e}",
            ) from e
        self._rerankers[name] = instance  # 缓存，后续直接命中
        logger.info("reranker_loaded_on_demand", name=name)
        return instance

    def get_reranker_for_kb(self, kb_id: str) -> Reranker | None:
        """获取 KB 对应的 reranker 实例。

        Returns:
            Reranker: KB 显式配置了 reranker 时返回实例
            None:    KB 未配置 reranker（按 cfg.reranker 为空判断）
        """
        cfg = self._registry.get_required(kb_id)
        if not cfg.reranker:
            return None
        return self.get_reranker(cfg.reranker)

    # ---- 热加载（配置修改后即时刷新运行时实例） ----

    def reload_embedder(self, name: str) -> bool:
        """热加载单个 embedder（从 YAML 重新读取配置并替换实例）。

        配置不存在时移除旧实例；依赖缺失时 warn 并不替换（保留旧实例或保持空缺）。
        Returns: True=成功替换, False=配置不存在或加载失败
        """
        from . import config_store

        try:
            raw = config_store.get_embedder_config_raw(name)
        except config_store.ConfigNotFoundError:
            self._embedders.pop(name, None)
            logger.info("embedder_hot_remove", name=name, reason="config_deleted")
            return False

        try:
            cfg = EmbeddingConfig.from_raw(raw)
            instance = create_embedder(cfg)
            self._embedders[name] = instance
            # 清除可能依赖此 embedder 的 llamaindex 引擎缓存
            if hasattr(self, "_li_engines"):
                self._li_engines.clear()
            logger.info("embedder_hot_reload_success", name=name)
            return True
        except (ImportError, ValueError) as e:
            logger.warning("embedder_hot_reload_failed", name=name, error=str(e))
            return False

    def reload_reranker(self, name: str) -> bool:
        """热加载单个 reranker（从 YAML 重新读取配置并替换实例）。

        Returns: True=成功替换, False=配置不存在或加载失败
        """
        from ..capabilities.embedding.base import _resolve_env
        from . import config_store

        try:
            raw = config_store.get_reranker_config_raw(name)
        except config_store.ConfigNotFoundError:
            self._rerankers.pop(name, None)
            logger.info("reranker_hot_remove", name=name, reason="config_deleted")
            return False

        try:
            cfg = RerankerConfig(**_resolve_env(raw))
            instance = create_reranker(cfg)
            self._rerankers[name] = instance
            if hasattr(self, "_li_engines"):
                self._li_engines.clear()
            logger.info("reranker_hot_reload_success", name=name)
            return True
        except (ImportError, ValueError) as e:
            logger.warning("reranker_hot_reload_failed", name=name, error=str(e))
            return False

    def remove_embedder(self, name: str) -> None:
        """移除 embedder 实例（配置删除时调用）。"""
        self._embedders.pop(name, None)
        if hasattr(self, "_li_engines"):
            self._li_engines.clear()

    def remove_reranker(self, name: str) -> None:
        """移除 reranker 实例（配置删除时调用）。"""
        self._rerankers.pop(name, None)
        if hasattr(self, "_li_engines"):
            self._li_engines.clear()

    # ---- LlamaIndex 检索路径 ----

    def get_llamaindex_engine(self, kb_id: str) -> LlamaIndexRetrievalEngine:
        """获取 KB 对应的 LlamaIndexRetrievalEngine（懒构造，单例缓存）。

        缺 llama-index-core 时抛 ImportError。
        """
        if not hasattr(self, "_li_engines"):
            self._li_engines: dict[str, LlamaIndexRetrievalEngine] = {}
        if kb_id in self._li_engines:
            return self._li_engines[kb_id]

        from ..capabilities.llamaindex import LlamaIndexRetrievalEngine

        cfg = self._registry.get_required(kb_id)
        engine = LlamaIndexRetrievalEngine(
            embedder=self.get_embedder(kb_id),
            vector_store=self._vector_store,
            collection=cfg.collection,
            reranker=self.get_reranker_for_kb(kb_id),
            top_k=5,
        )
        self._li_engines[kb_id] = engine
        return engine

    async def search_via_llamaindex(
        self, kb_id: str, query: str, top_k: int = 5
    ) -> list[dict]:
        """用 LlamaIndex 检索（与默认检索路径并存）。

        适合需要 LI 扩展能力（查询改写、HyDE）的场景。
        普通 RAG 用默认路径即可。
        """
        engine = self.get_llamaindex_engine(kb_id)
        results = await engine.aretrieve(query)
        return [
            {
                "doc_id": r.id,
                "title": r.metadata.get("title", r.id),
                "text": r.text,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in results
        ]

    # ---- 策略化检索（Dense / Sparse / Hybrid） ----

    def get_retrieval_config(self) -> RetrievalConfig:
        """获取检索策略全局配置（从 retrieval.yaml 加载）。"""
        if not hasattr(self, "_retrieval_config"):
            self._retrieval_config = _load_retrieval_config()
        return self._retrieval_config

    async def _invalidate_retrieval_engines(self, kb_id: str) -> None:
        """使 KB 对应的检索引擎缓存失效。

        文档增删后调用，确保 Sparse 引擎的 BM25 索引在下次检索时重建。
        Dense / Hybrid 引擎本身无状态（直接查向量库），但清理缓存让参数
        变更也能即时生效。

        此方法是 async 的，因为 Sparse 引擎的 invalidate_index 需要获取
        asyncio.Lock（与 _ensure_index 互斥，避免索引构建中被清空）。
        """
        if not hasattr(self, "_retrieval_engines"):
            return
        keys_to_remove = [k for k in self._retrieval_engines if k.startswith(f"{kb_id}:")]
        for k in keys_to_remove:
            engine = self._retrieval_engines.pop(k, None)
            # Sparse 引擎有内存 BM25 索引，显式释放（需 await 获取锁）
            if engine is not None and hasattr(engine, "invalidate_index"):
                await engine.invalidate_index()
            logger.info("retrieval_engine_cache_invalidated", kb_id=kb_id, cache_key=k)

    def get_retrieval_engine(
        self,
        kb_id: str,
        strategy: RetrievalStrategy | None = None,
        params: dict[str, Any] | None = None,
    ) -> RetrievalEngine:
        """获取 KB 对应的检索引擎（懒构造，单例缓存）。

        策略选择优先级：
          1. 参数 strategy（显式指定）
          2. KB 配置的 retrieval_strategy
          3. 全局默认策略（retrieval.yaml 的 default_strategy）

        参数覆盖优先级：
          1. 参数 params（显式指定）
          2. KB 配置的 retrieval_params
          3. 全局默认参数（retrieval.yaml 各策略参数）
        """
        if not hasattr(self, "_retrieval_engines"):
            self._retrieval_engines: dict[str, RetrievalEngine] = {}

        # 缓存 key = kb_id + strategy（不同策略可能共存）
        cfg = self._registry.get_required(kb_id)
        effective_strategy = strategy or _resolve_strategy(cfg, self.get_retrieval_config())
        cache_key = f"{kb_id}:{effective_strategy.value}"

        if cache_key in self._retrieval_engines:
            return self._retrieval_engines[cache_key]

        # 合并参数：全局默认 < KB 配置 < 显式指定
        global_cfg = self.get_retrieval_config()
        merged_params = _merge_strategy_params(
            effective_strategy, global_cfg, cfg.retrieval_params, params
        )

        engine = create_retrieval_engine(
            strategy=effective_strategy,
            embedder=self.get_embedder(kb_id),
            vector_store=self._vector_store,
            collection=cfg.collection,
            reranker=self.get_reranker_for_kb(kb_id),
            params=merged_params,
        )

        self._retrieval_engines[cache_key] = engine
        logger.info(
            "retrieval_engine_created",
            kb_id=kb_id,
            strategy=effective_strategy.value,
            cache_key=cache_key,
        )
        return engine

    async def search_via_strategy(
        self,
        kb_id: str,
        query: str,
        top_k: int = 5,
        use_rerank: bool = True,
        strategy: RetrievalStrategy | None = None,
        params: dict[str, Any] | None = None,
        filter_expr: dict[str, Any] | None = None,
        reranker_name: str | None = None,
    ) -> RetrievalResponse:
        """用策略化检索接口执行搜索。

        Args:
            kb_id: 知识库 ID
            query: 检索查询
            top_k: 返回结果数
            use_rerank: 是否启用 rerank
            strategy: 检索策略（None = 使用 KB 配置 / 全局默认）
            params: 策略参数覆盖
            filter_expr: metadata 过滤条件
            reranker_name: 显式指定 reranker（覆盖 KB 默认）

        Returns:
            RetrievalResponse: 检索结果
        """
        # 处理 reranker_name 覆盖
        effective_reranker = None
        if use_rerank:
            if reranker_name:
                try:
                    effective_reranker = self.get_reranker(reranker_name)
                except ComponentUnavailableError:
                    effective_reranker = self.get_reranker_for_kb(kb_id)
            else:
                effective_reranker = self.get_reranker_for_kb(kb_id)

        # 获取检索引擎
        engine = self.get_retrieval_engine(kb_id, strategy=strategy, params=params)

        # 解析最终策略 + 合并参数（用于请求记录与 reranker 覆盖场景）
        cfg = self._registry.get_required(kb_id)
        global_cfg = self.get_retrieval_config()
        effective_strategy = strategy or _resolve_strategy(cfg, global_cfg)
        merged_params = _merge_strategy_params(
            effective_strategy, global_cfg, cfg.retrieval_params, params
        )

        # 如果指定了不同的 reranker，需要重新创建引擎（绕过缓存，因缓存 key 不含 reranker）
        if effective_reranker is not None and reranker_name:
            engine = create_retrieval_engine(
                strategy=effective_strategy,
                embedder=self.get_embedder(kb_id),
                vector_store=self._vector_store,
                collection=cfg.collection,
                reranker=effective_reranker,
                params=merged_params,
            )

        request = RetrievalRequest(
            kb_id=kb_id,
            query=query,
            top_k=top_k,
            use_rerank=use_rerank,
            strategy=effective_strategy,
            strategy_params=merged_params,
            filter_expr=filter_expr,
            reranker_name=reranker_name,
        )

        # 通过熔断器执行检索，保护下游（vector store / embedder / reranker）
        # 连续失败超过阈值时熔断，快速失败而非逐个超时
        from ..observability.circuit_breaker import get_search_breaker

        async def _do_retrieve() -> RetrievalResponse:
            return await engine.retrieve(request)

        return await get_search_breaker().call(_do_retrieve)

    async def add_document_chunked(
        self,
        kb_id: str,
        doc_id: str,
        title: str,
        content: str,
        source: str = "",
        parser_config: dict | None = None,
        parser_name: str | None = None,
    ) -> int:
        """用 NodeParser 把文档切块后写入。

        Args:
            parser_config: 原始 parser 配置 dict（strategy / chunk_size / ...）。
                           为 None 时回退到 sentence / 512 / 50。
            parser_name: 命名 parser 名（如 'sentence_512'），存到 DocumentMeta.parser
                         让前端显示人类可读的名字。若 None，使用 strategy 作为兜底。

        Returns:
            int: 切成的块数

        与 add_document 的区别：
          - add_document: 整篇文档作为一个 chunk（简单）
          - add_document_chunked: 按 NodeParser 切块（更细粒度）
        """
        from ..capabilities.llamaindex import (
            LIEmbeddingAdapter,
            NodeParserFactory,
        )
        from ..capabilities.vector_store import VectorRecord

        cfg = self._registry.get_required(kb_id)
        embedder = self.get_embedder(kb_id)
        li_embedder = LIEmbeddingAdapter(embedder)
        parser = NodeParserFactory.from_raw(
            parser_config or {"strategy": "sentence", "chunk_size": 512},
            embed_model=li_embedder,
        )
        nodes = parser.parse(content, doc_id=doc_id, title=title, source=source)
        if not nodes:
            return 0

        # 用命名 parser 标识（如 'sentence_512'），没传时回退到 strategy
        parser_label = parser_name or parser._config.strategy

        count = 0
        for n in nodes:
            vec = await embedder.embed_query(n.text)
            rec = VectorRecord(
                id=n.node_id,
                vector=vec,
                text=n.text,
                metadata={
                    "title": title,
                    "source": source,
                    "doc_id": doc_id,
                    "chunk_index": n.chunk_index,
                    "parser": parser_label,
                },
            )
            await self._vector_store.insert(cfg.collection, [rec])
            count += 1

        # 持久化 DocumentMeta（含 content / chunk_count / parser），
        # 让 list_documents / get_document / delete_document 都能找到。
        meta = DocumentMeta(
            doc_id=doc_id,
            kb_id=kb_id,
            title=title,
            source=source,
            content=content,
            metadata={"parser": parser_label},
            chunk_count=count,
            char_count=len(content),
            parser=parser_label,
        )
        async with self._lock:
            self._docs.setdefault(kb_id, {})[doc_id] = meta
        self._save_docs_to_disk()
        # 文档变更 → 失效检索引擎缓存（Sparse 的 BM25 索引需重建）
        await self._invalidate_retrieval_engines(kb_id)
        return count

    async def add_document_from_upload(
        self,
        kb_id: str,
        doc_id: str,
        title: str,
        content: str,
        source: str = "",
        metadata: dict | None = None,
        parser_name: str | None = None,
    ) -> int:
        """上传文件入库（content 已经是 reader 解析后的纯文本）。

        与 add_document_chunked 的区别：
          - 切块策略用 configs/llamaindex.yaml 的命名 parser（parser_name）
          - 默认回退到 sentence 512
          - 额外把 upload 时的 metadata（filename/format/...）合入 chunk metadata
          - 写完块后会保存 DocumentMeta（含 content）到 _docs

        Args:
            parser_name: 命名 parser（configs/llamaindex.yaml），如 'sentence_512'。
                         为 None 时回退到 sentence / 512 / 50（向后兼容）。

        Returns:
            切成的 chunk 数；0 表示内容为空。
        """
        from ..capabilities.llamaindex import (
            LIEmbeddingAdapter,
            NodeParserFactory,
            get_parser_registry,
        )
        from ..capabilities.vector_store import VectorRecord

        if not content or not content.strip():
            return 0

        cfg = self._registry.get_required(kb_id)
        embedder = self.get_embedder(kb_id)
        li_embedder = LIEmbeddingAdapter(embedder)

        if parser_name:
            factory = get_parser_registry().get(
                parser_name, embed_model=li_embedder
            )
        else:
            factory = NodeParserFactory.from_raw(
                {"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 50},
                embed_model=li_embedder,
            )

        nodes = factory.parse(content, doc_id=doc_id, title=title, source=source)
        if not nodes:
            return 0

        # 用命名 parser 名（如 'sentence_512'）做人类可读标识。
        # 没传 parser_name 时回退到 strategy（如 'sentence'）。
        parser_label = parser_name or factory._config.strategy

        meta_extra = dict(metadata or {})
        filename = meta_extra.get("filename", "")

        count = 0
        for n in nodes:
            vec = await embedder.embed_query(n.text)
            rec = VectorRecord(
                id=n.node_id,
                vector=vec,
                text=n.text,
                metadata={
                    "title": title,
                    "source": source,
                    "doc_id": doc_id,
                    "chunk_index": n.chunk_index,
                    "parser": parser_label,
                    "filename": filename,
                },
            )
            await self._vector_store.insert(cfg.collection, [rec])
            count += 1

        # 持久化 DocumentMeta（含 content），让 get_document / list_documents 能拿到
        meta = DocumentMeta(
            doc_id=doc_id,
            kb_id=kb_id,
            title=title,
            source=source,
            content=content,
            metadata=meta_extra,
            chunk_count=count,
            char_count=len(content),
            parser=parser_label,
        )
        async with self._lock:
            self._docs.setdefault(kb_id, {})[doc_id] = meta
        self._save_docs_to_disk()
        # 文档变更 → 失效检索引擎缓存（Sparse 的 BM25 索引需重建）
        await self._invalidate_retrieval_engines(kb_id)
        return count


# ---- 工厂 ----

_manager_singleton: KnowledgeBaseManager | None = None


def build_default_manager() -> KnowledgeBaseManager:
    """构建默认的 KnowledgeBaseManager（单例）。

    使用 mock embedder + mock vector store。
    后续可改为从 YAML 加载真实实现。
    """
    global _manager_singleton
    if _manager_singleton is not None:
        return _manager_singleton


    registry = get_registry()

    # 收集所有 KB 用到的 embedding 配置
    # 企业级：依赖缺失时降级（warn + 跳过），不让 dev 环境因 BGE/Cohere 缺失崩溃
    embedders: dict[str, Embedder] = {}
    for cfg_name, cfg in _load_embedding_configs().items():  # type: ignore[assignment]
        try:
            embedders[cfg_name] = create_embedder(cfg)  # type: ignore[arg-type]
        except ImportError as e:
            logger.warning(
                "embedding_provider_unavailable",
                provider=cfg_name,
                error=str(e),
                hint="install the relevant extra or use a mock provider",
            )

    # 收集所有 KB 用到的 reranker 配置
    rerankers: dict[str, Reranker] = {}
    for cfg_name, cfg in _load_reranker_configs().items():  # type: ignore[assignment]
        try:
            rerankers[cfg_name] = create_reranker(cfg)  # type: ignore[arg-type]
        except ImportError as e:
            logger.warning(
                "reranker_provider_unavailable",
                provider=cfg_name,
                error=str(e),
                hint="install the relevant extra or use a mock provider",
            )

    # 单个共享 vector store
    vs_config = _load_vector_store_config()
    vector_store = create_vector_store(vs_config)

    _manager_singleton = KnowledgeBaseManager(
        registry=registry,
        vector_store=vector_store,
        embedders=embedders,
        rerankers=rerankers,
        docs_index_path=AppPaths.DOCS_INDEX,
    )
    logger.info(
        "KnowledgeBaseManager 初始化完成: %d embedders, %d rerankers, 1 vector_store",
        len(embedders),
        len(rerankers),
    )
    return _manager_singleton


def reset_manager_singleton() -> None:
    """测试用：重置 manager 单例。"""
    global _manager_singleton
    _manager_singleton = None


# ===== 热加载辅助（供 admin API 调用） =====


def hot_reload_embedder(name: str) -> bool:
    """热加载 embedder（写入 YAML 后立即刷新运行时实例）。"""
    if _manager_singleton is None:
        logger.warning("hot_reload_embedder_no_manager", name=name)
        return False
    return _manager_singleton.reload_embedder(name)


def hot_reload_reranker(name: str) -> bool:
    """热加载 reranker（写入 YAML 后立即刷新运行时实例）。"""
    if _manager_singleton is None:
        logger.warning("hot_reload_reranker_no_manager", name=name)
        return False
    return _manager_singleton.reload_reranker(name)


def hot_remove_embedder(name: str) -> None:
    """移除 embedder 实例（配置删除时调用）。"""
    if _manager_singleton is None:
        return
    _manager_singleton.remove_embedder(name)


def hot_remove_reranker(name: str) -> None:
    """移除 reranker 实例（配置删除时调用）。"""
    if _manager_singleton is None:
        return
    _manager_singleton.remove_reranker(name)


def _load_embedding_configs() -> dict[str, EmbeddingConfig]:
    """从 YAML 加载所有 embedding 配置。

    - 自动处理 ${ENV_VAR} 替换
    - 只返回"被引用"的配置（按需加载，避免启动时加载大模型）
    """
    from ..capabilities.embedding import EmbeddingConfig

    path = AppPaths.EMBEDDING_CONFIG
    if not path.exists():
        return {}
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    all_configs = data.get("embeddings", {})

    # 收集 KB 中实际引用的 embedding 名
    registry = get_registry()
    referenced_names = {cfg.embedding for cfg in registry.all_cfgs(enabled_only=True)}

    # 只构造被引用的配置；缺配置时 warn + 跳过（与依赖缺失降级策略一致）
    configs: dict[str, EmbeddingConfig] = {}
    for name in referenced_names:
        if name not in all_configs:
            logger.warning(
                "embedding_config_missing",
                embedding=name,
                hint="KB will be skipped; define it in embeddings.yaml or remove the reference",
            )
            continue
        configs[name] = EmbeddingConfig.from_raw(all_configs[name])

    return configs


def _load_vector_store_config() -> VectorStoreConfig:
    """从 YAML 加载 vector store 配置，支持 ${ENV_VAR} 占位符。"""
    from ..capabilities.embedding.base import _resolve_env
    from ..capabilities.vector_store import VectorStoreConfig

    path = AppPaths.VECTOR_STORE_CONFIG
    if not path.exists():
        # 兜底使用 mock
        return VectorStoreConfig(provider="mock")
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("vector_store", {"provider": "mock"})
    return VectorStoreConfig(**_resolve_env(raw))


def _load_reranker_configs() -> dict[str, RerankerConfig]:
    """从 YAML 加载所有 reranker 配置。

    - 自动处理 ${ENV_VAR} 替换
    - 只返回"被引用"的配置（按需加载，避免启动时加载大模型）

    注意：reranker 为 None 的 KB（cfg.reranker == None）不会引用任何 reranker 配置，
    因此不被引用的 reranker 配置会被跳过。
    """
    from ..capabilities.embedding.base import _resolve_env

    path = AppPaths.RERANKER_CONFIG
    if not path.exists():
        return {}
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    all_configs = data.get("rerankers", {})

    # 收集 KB 中实际引用的 reranker 名
    registry = get_registry()
    referenced_names: set[str] = set()
    for cfg in registry.all_cfgs(enabled_only=True):
        if cfg.reranker:
            referenced_names.add(cfg.reranker)

    # 只构造被引用的配置；缺配置时 warn + 跳过（与依赖缺失降级策略一致）
    configs: dict[str, RerankerConfig] = {}
    for name in referenced_names:
        if name not in all_configs:
            logger.warning(
                "reranker_config_missing",
                reranker=name,
                hint="KB will use no reranker; define it in reranker.yaml or remove the reference",
            )
            continue
        raw = _resolve_env(all_configs[name])
        configs[name] = RerankerConfig(**raw)

    return configs


def _load_single_reranker_config(name: str) -> RerankerConfig | None:
    """从 reranker.yaml 按名加载单个配置（不受 KB 引用过滤限制）。

    用于检索调试时按需实例化未被 KB 引用的 reranker（如对比不同模型效果）。
    返回 None 表示配置文件中不存在该 name。
    """
    from ..capabilities.embedding.base import _resolve_env

    path = AppPaths.RERANKER_CONFIG
    if not path.exists():
        return None
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    all_configs = data.get("rerankers", {})
    if name not in all_configs:
        return None
    raw = _resolve_env(all_configs[name])
    return RerankerConfig(**raw)


# ===== 管理面辅助（前端下拉 / KB CRUD） =====


def list_available_embedders() -> list[dict]:
    """列出 embeddings.yaml 里所有 embedder 配置（不管是否被 KB 引用）。

    给前端下拉用：每个 embedder 都返回，无论依赖是否装齐。
    """
    from ..capabilities.embedding import EmbeddingConfig

    path = AppPaths.EMBEDDING_CONFIG
    if not path.exists():
        return []
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("embeddings", {})

    # 当前 manager 已加载的 embedder 名
    loaded: set[str] = set()
    if _manager_singleton is not None:
        loaded = set(_manager_singleton._embedders.keys())  # type: ignore[attr-defined]

    out: list[dict] = []
    for name, item in raw.items():
        try:
            cfg = EmbeddingConfig.from_raw(item)
        except Exception:  # noqa: BLE001
            # 配置有问题，包装成兜底
            out.append(
                {
                    "name": name,
                    "provider": str(item.get("provider", "?")),
                    "model_name": str(item.get("model_name", "?")),
                    "dimension": int(item.get("dimension", 0)),
                    "batch_size": int(item.get("batch_size", 32)),
                    "loaded": False,
                    "description": f"[配置错误] {item.get('description', '')}",
                }
            )
            continue
        out.append(
            {
                "name": name,
                "provider": cfg.provider,
                "model_name": cfg.model_name,
                "dimension": cfg.dimension,
                "batch_size": cfg.batch_size,
                "loaded": name in loaded,
                "description": str(item.get("description", "")),
            }
        )
    return out


def list_available_rerankers() -> list[dict]:
    """列出 reranker.yaml 里所有 reranker 配置（不管是否被 KB 引用）。"""
    path = AppPaths.RERANKER_CONFIG
    if not path.exists():
        return []
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("rerankers", {})

    loaded: set[str] = set()
    if _manager_singleton is not None:
        loaded = set(_manager_singleton._rerankers.keys())  # type: ignore[attr-defined]

    out: list[dict] = []
    for name, item in raw.items():
        try:
            top_k = int(item.get("top_k", 5))
        except (TypeError, ValueError):
            top_k = 5
        out.append(
            {
                "name": name,
                "provider": str(item.get("provider", "?")),
                "model_name": str(item.get("model_name", "")),
                "top_k": top_k,
                "loaded": name in loaded,
                "description": str(item.get("description", "")),
            }
        )
    return out


async def delete_kb_collection(kb_id: str) -> None:
    """删除 KB：drop collection + 清空 in-memory 文档。

    不动 registry（registry 由上层调用 create/update/delete_kb 改）。
    """
    if _manager_singleton is None:
        return
    cfg = _manager_singleton._registry.get(kb_id)  # type: ignore[attr-defined]
    if cfg is None:
        return
    # drop collection；失败不抛（collection 可能没建过 / 不可写）
    try:
        await _manager_singleton._vector_store.drop_collection(cfg.collection)  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "drop_collection_failed", kb_id=kb_id, collection=cfg.collection, error=str(e)
        )
    # 清空 _docs
    async with _manager_singleton._lock:  # type: ignore[attr-defined]
        _manager_singleton._docs.pop(kb_id, None)  # type: ignore[attr-defined]
    # 清空 llamaindex 引擎缓存
    if hasattr(_manager_singleton, "_li_engines"):  # type: ignore[attr-defined]
        _manager_singleton._li_engines.pop(kb_id, None)  # type: ignore[attr-defined]
    # 清空策略化检索引擎缓存
    await _manager_singleton._invalidate_retrieval_engines(kb_id)  # type: ignore[attr-defined]


async def recreate_collection(kb_id: str) -> dict[str, Any]:
    """重建 KB 的 collection（升级为 BM25 schema），保留已有向量数据。

    适用场景：旧 collection 是 dense-only schema（init_kb 解耦前创建），
    需要用 sparse_milvus / hybrid_milvus 策略检索时，通过本方法升级 schema。

    流程：
      1. 从旧 collection 分页拉取所有 chunks（含 vector / text / metadata）
      2. drop 旧 collection
      3. 创建新 collection（优先 BM25 schema，不支持时降级普通）
      4. 重新 insert 所有 chunks（向量已存在，无需重新 embedding）
      5. 失效检索引擎缓存

    不会修改 _docs 索引（DocumentMeta 不受影响）。

    Returns:
        {"kb_id", "collection", "migrated_chunks", "bm25_enabled"}

    Raises:
        KBNotFoundError / ComponentUnavailableError
    """
    if _manager_singleton is None:
        raise RuntimeError("KnowledgeBaseManager not initialized")
    return await _manager_singleton.recreate_collection(kb_id)  # type: ignore[attr-defined]


# ===== 检索策略辅助函数 =====


def _load_retrieval_config() -> RetrievalConfig:
    """从 retrieval.yaml 加载检索策略配置。"""
    path = AppPaths.RETRIEVAL_CONFIG
    if not path.exists():
        logger.info("retrieval_config_not_found_using_defaults")
        return RetrievalConfig()

    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("retrieval", {})

    return RetrievalConfig(**raw)


def _resolve_strategy(
    cfg: Any,  # KnowledgeBaseConfig
    global_cfg: RetrievalConfig,
) -> RetrievalStrategy:
    """解析 KB 实际使用的检索策略。

    优先级：KB 配置 > 全局默认
    """
    if cfg.retrieval_strategy:
        try:
            return RetrievalStrategy(cfg.retrieval_strategy)
        except ValueError:
            logger.warning(
                "invalid_retrieval_strategy_fallback_to_default",
                kb_id=cfg.kb_id,
                strategy=cfg.retrieval_strategy,
                default=global_cfg.default_strategy.value,
            )
    return global_cfg.default_strategy


def _merge_strategy_params(
    strategy: RetrievalStrategy,
    global_cfg: RetrievalConfig,
    kb_params: dict[str, Any],
    request_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """合并策略参数：全局默认 < KB 配置 < 请求覆盖。"""
    # 全局默认参数
    if strategy == RetrievalStrategy.DENSE:
        base = global_cfg.dense.model_dump()
    elif strategy == RetrievalStrategy.SPARSE:
        base = global_cfg.sparse.model_dump()
    elif strategy == RetrievalStrategy.HYBRID:
        base = global_cfg.hybrid.model_dump()
    elif strategy in (RetrievalStrategy.SPARSE_MILVUS, RetrievalStrategy.HYBRID_MILVUS):
        base = global_cfg.milvus_bm25.model_dump()
    else:
        base = {}

    # KB 配置覆盖
    base.update(kb_params)

    # 请求级覆盖
    if request_params:
        base.update(request_params)

    return base
