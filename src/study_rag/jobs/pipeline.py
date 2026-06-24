"""上传分块 pipeline。

职责：跑 parser → chunker → embedder → store 完整流程。
所有依赖通过参数注入（不直接 import app.state.*）。

典型用法（在 admin upload endpoint 里）：
    job_id = await jobs.submit(
        "upload_doc",
        lambda jid, on_progress, is_cancelled: run_chunking_pipeline(
            job_id=jid,
            on_progress=on_progress,
            is_cancelled=is_cancelled,
            file_content=data,
            filename=filename,
            doc_id=doc_id,
            title=title,
            parser_name=parser_name,
            kb_id=kb_id,
            embedder_registry=...,
            parser_registry=...,
            kb_manager=...,
        ),
        kb_id=kb_id,
        doc_id=doc_id,
        filename=filename,
    )
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from .manager import CancelCheck, ProgressCallback
from .models import JobStage

logger = logging.getLogger(__name__)


async def run_chunking_pipeline(
    *,
    job_id: str,
    on_progress: ProgressCallback,
    is_cancelled: CancelCheck,
    # 输入
    file_content: bytes | str,
    filename: str,
    doc_id: str,
    title: str,
    parser_name: str,
    kb_id: str,
    source: str = "",
    metadata: dict[str, Any] | None = None,
    # 服务依赖（注入）
    embedder_registry: Any,
    parser_registry: Any,
    kb_manager: Any,
) -> None:
    """完整的上传 → 切块 → 向量化 → 入库 流程。

    on_progress(stage, current, total, message) 报告进度。
    is_cancelled() 返回 True 时应尽快退出（已抛 ``CancelledError`` 友好的姿势）。
    """
    from ..capabilities.llamaindex import (
        LIEmbeddingAdapter,
        get_parser_registry,
    )
    from ..capabilities.vector_store import VectorRecord
    from ..knowledge_bases.models import DocumentMeta

    meta_extra = dict(metadata or {})
    if isinstance(file_content, bytes):
        text = file_content.decode("utf-8", errors="replace") if file_content else ""
    else:
        text = file_content
    if not text or not text.strip():
        await on_progress(JobStage.CHUNKING, 0, 0, "内容为空")
        return

    # ---- Stage 1: parsing（读文件 → 文本）----
    await on_progress(JobStage.PARSING, 0, 1, "正在解析文件")
    if is_cancelled():
        raise asyncio.CancelledError()
    await asyncio.sleep(0)  # 让出 event loop
    await on_progress(JobStage.PARSING, 1, 1, f"已解析 {len(text)} 字符")

    # ---- Stage 2: chunking（按 parser 切）----
    await on_progress(JobStage.CHUNKING, 0, 1, "正在切块")
    if is_cancelled():
        raise asyncio.CancelledError()

    # 拿 embedder（先拿，因为 semantic parser 需要）
    cfg = kb_manager._registry.get_required(kb_id)  # type: ignore[attr-defined]
    embedder_name = getattr(cfg, "embedding", None) or getattr(
        cfg, "embedder_name", ""
    )
    embedder = embedder_registry.get(embedder_name)  # type: ignore[attr-defined]

    # 准备 embed_model 适配（semantic parser 需要）
    li_embedder = None
    try:
        from ..capabilities.llamaindex import LIEmbeddingAdapter

        li_embedder = LIEmbeddingAdapter(embedder)
    except Exception:  # noqa: BLE001
        li_embedder = None

    # parser_registry 可能是我们包装的（get_parser_registry）也可能是测试里的 Fake
    if hasattr(parser_registry, "get"):
        try:
            # 优先尝试带 embed_model 的 get（semantic 需要）
            factory = parser_registry.get(parser_name, embed_model=li_embedder)
        except (TypeError, KeyError):
            try:
                factory = parser_registry.get(parser_name, embed_model=None)
            except TypeError:
                factory = parser_registry.get(parser_name)
    else:
        try:
            factory = get_parser_registry().get(
                parser_name, embed_model=li_embedder
            )
        except Exception:  # noqa: BLE001
            factory = get_parser_registry().get(parser_name)

    # 实际切块
    if hasattr(factory, "parse"):
        # 兼容两种调用签名：
        # 1) parse(content, doc_id=, title=, source=)
        # 2) parse(text, **kwargs)
        # ⚠️ factory.parse 是同步（LlamaIndex 内部 embedder 调 HTTP 是阻塞的）
        #    在 async 上下文里直接 await 会卡住整个 event loop，连响应序列化都卡
        #    → 用 asyncio.to_thread 扔到默认 threadpool
        def _do_parse() -> list[Any]:
            try:
                return factory.parse(
                    text, doc_id=doc_id, title=title, source=source
                )
            except TypeError:
                return factory.parse(text, doc_id=doc_id)
        nodes = await asyncio.to_thread(_do_parse)
    else:
        nodes = []

    n_chunks = len(nodes)
    await on_progress(JobStage.CHUNKING, 1, 1, f"已切成 {n_chunks} 块")
    if n_chunks == 0:
        return

    if is_cancelled():
        raise asyncio.CancelledError()

    # ---- Stage 3: embedding（最慢，按 chunk 报告）----
    await on_progress(
        JobStage.EMBEDDING, 0, n_chunks, f"准备向量化 {n_chunks} 块"
    )

    records: list[VectorRecord] = []
    progress_step = max(1, n_chunks // 20)  # 至少 20 个 tick
    for i, node in enumerate(nodes):
        if is_cancelled():
            raise asyncio.CancelledError()

        node_text = getattr(node, "text", None) or str(node)
        node_meta = getattr(node, "metadata", {}) or {}

        # 兼容 embedder.embed_query / 我们的 Embedder 接口
        if hasattr(embedder, "embed_query"):
            vec = await embedder.embed_query(node_text)
        else:
            # LlamaIndex BaseEmbedding 协议
            li_adapter = LIEmbeddingAdapter(embedder)
            vec = await li_adapter._aget_text_embedding(node_text)

        node_id = getattr(node, "node_id", f"{doc_id}#chunk-{i}")
        chunk_index = getattr(node, "chunk_index", i)
        records.append(
            VectorRecord(
                id=node_id,
                vector=vec,
                text=node_text,
                metadata={
                    "title": title,
                    "source": source,
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "parser": getattr(
                        getattr(factory, "_config", None), "strategy", parser_name
                    ),
                    "filename": filename,
                    **node_meta,
                },
            )
        )

        if (i + 1) % progress_step == 0 or (i + 1) == n_chunks:
            await on_progress(
                JobStage.EMBEDDING,
                i + 1,
                n_chunks,
                f"向量化 {i + 1}/{n_chunks}",
            )

    # ---- Stage 4: saving to vector store ----
    if is_cancelled():
        raise asyncio.CancelledError()
    await on_progress(JobStage.SAVING, 0, 1, "正在写入向量库")

    await kb_manager._vector_store.insert(cfg.collection, records)  # type: ignore[attr-defined]

    # ---- Stage 5: saving metadata ----
    meta = DocumentMeta(
        doc_id=doc_id,
        kb_id=kb_id,
        title=title,
        source=source,
        content=text[:1000],  # 限长，避免大文档塞 metadata
        metadata={**meta_extra, "n_chunks": n_chunks, "parser": parser_name},
    )
    async with kb_manager._lock:  # type: ignore[attr-defined]
        kb_manager._docs.setdefault(kb_id, {})[doc_id] = meta  # type: ignore[attr-defined]
    save_fn = getattr(kb_manager, "_save_docs_to_disk", None)
    if callable(save_fn):
        save_fn()

    await on_progress(JobStage.SAVING, 1, 1, f"完成，共 {n_chunks} 块")


__all__ = ["run_chunking_pipeline"]
