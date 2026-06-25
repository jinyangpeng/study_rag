"""文档类 Tool：增删查 + 批量 + LlamaIndex 切块添加。

所有管理类 Tool 都需要写权限（user.can_write(kb_id) == True）。
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from ...knowledge_bases.models import DocumentCreate
from ..context import MCPContext
from ..errors import (
    DocumentAlreadyExistsError,
    DocumentNotFoundError,
    InvalidParameterError,
    KBNotFoundError,
)

logger = logging.getLogger(__name__)


# ---- Response Models ----


class Document(BaseModel):
    """文档详情。"""

    doc_id: str
    kb_id: str
    title: str
    content: str
    source: str | None = None
    metadata: dict


class DocumentSummary(BaseModel):
    """文档摘要（list 接口用）。"""

    doc_id: str
    kb_id: str
    title: str
    source: str | None = None
    metadata: dict = Field(default_factory=dict)
    has_content: bool = True


class AddDocumentResult(BaseModel):
    """添加文档的返回。"""

    doc_id: str
    kb_id: str
    title: str
    chunks: int = 1  # add_document=1，add_document_chunked 返回实际块数


class DeleteResult(BaseModel):
    """删除文档的返回。"""

    doc_id: str
    kb_id: str
    deleted: bool


# ---- 内部 helper ----


def _get_ctx(ctx: MCPContext | None) -> MCPContext:
    if ctx is None:
        from ..context import MCPContext as _Ctx

        return _Ctx.default()
    return ctx


def _require_kb(ctx: MCPContext, kb_id: str) -> None:
    """校验 KB 存在且启用。"""
    from ...knowledge_bases.registry import get_registry

    cfg = get_registry().get(kb_id)
    if cfg is None or not cfg.enabled:
        raise KBNotFoundError(f"KB not found or disabled: {kb_id}")


async def _check_write(ctx: MCPContext, api_key: str, kb_id: str) -> Any:
    """鉴权 + 写权限校验。返回 user。"""
    user = await ctx.auth.resolve(api_key)
    _require_kb(ctx, kb_id)
    ctx.auth.check_kb_write_access(user, kb_id)
    return user


# ---- Tool: get_document ----


async def get_document(
    api_key: str,
    kb_id: str,
    doc_id: str,
    ctx: MCPContext | None = None,
) -> Document:
    """获取指定文档的完整内容。

    适用场景:
      - search_kb 返回结果后，查看完整文档
      - Agent 需要引用具体文档内容

    参数:
      api_key: 用户凭证
      kb_id: 知识库 ID
      doc_id: 文档 ID

    异常:
      KBNotFoundError: 知识库不存在
      DocumentNotFoundError: 文档不存在
      PermissionDenied: 无权访问
    """
    ctx = _get_ctx(ctx)

    user = await ctx.auth.resolve(api_key)
    _require_kb(ctx, kb_id)
    ctx.auth.check_kb_access(user, kb_id)

    meta = ctx.manager.get_document(kb_id, doc_id)
    if meta is None:
        raise DocumentNotFoundError(f"Document not found: {kb_id}/{doc_id}")

    return Document(
        doc_id=meta.doc_id,
        kb_id=meta.kb_id,
        title=meta.title,
        content=meta.content or "",
        source=meta.source,
        metadata=meta.metadata,
    )


# ---- Tool: list_documents ----


async def list_documents(
    api_key: str,
    kb_id: str,
    include_content: bool = False,
    limit: int = 100,
    offset: int = 0,
    ctx: MCPContext | None = None,
) -> list[DocumentSummary]:
    """列出指定 KB 中的文档摘要。

    适用场景:
      - 列出 KB 中现有文档（用于管理界面、批量操作）
      - 不需要拉取完整内容

    参数:
      api_key: 用户凭证
      kb_id: 知识库 ID
      include_content: 是否在结果中包含 content 字段（默认 False，仅返回摘要）
      limit: 返回条数上限（默认 100）
      offset: 跳过条数（用于分页）

    异常:
      KBNotFoundError: 知识库不存在
      PermissionDenied: 无权访问
    """
    ctx = _get_ctx(ctx)

    user = await ctx.auth.resolve(api_key)
    _require_kb(ctx, kb_id)
    ctx.auth.check_kb_access(user, kb_id)

    if limit <= 0 or limit > 1000:
        raise InvalidParameterError("limit must be in (0, 1000]")

    docs = ctx.manager.list_documents(kb_id)
    docs.sort(key=lambda d: d.created_at, reverse=True)
    sliced = docs[offset : offset + limit]

    return [
        DocumentSummary(
            doc_id=d.doc_id,
            kb_id=d.kb_id,
            title=d.title,
            source=d.source,
            metadata=d.metadata,
            has_content=bool(d.content),
        )
        for d in sliced
    ]


# ---- Tool: add_document ----


async def add_document(
    api_key: str,
    kb_id: str,
    doc_id: str,
    title: str,
    content: str,
    source: str | None = None,
    metadata: dict | None = None,
    overwrite: bool = False,
    ctx: MCPContext | None = None,
) -> AddDocumentResult:
    """添加一个文档到指定知识库（整篇作为一个 chunk）。

    适用场景:
      - 短文档直接入库
      - 已经预处理好的结构化数据

    参数:
      api_key: 用户凭证
      kb_id: 知识库 ID
      doc_id: 文档 ID（同一 KB 内唯一）
      title: 标题
      content: 正文
      source: 来源（可选；会被存入 metadata['source']，可用于后续 filter）
      metadata: 额外 metadata（与 source 合并；后续 filter 可按这里字段过滤）
      overwrite: 文档已存在时是否覆盖（默认 False 抛 DocumentAlreadyExistsError）

    异常:
      KBNotFoundError: 知识库不存在
      DocumentAlreadyExistsError: doc_id 已存在且 overwrite=False
      PermissionDenied: 无写权限
    """
    ctx = _get_ctx(ctx)
    await _check_write(ctx, api_key, kb_id)

    if not doc_id or not title:
        raise InvalidParameterError("doc_id and title must not be empty")
    if not content:
        raise InvalidParameterError("content must not be empty")

    # 已存在检查
    existing = ctx.manager.get_document(kb_id, doc_id)
    if existing is not None and not overwrite:
        raise DocumentAlreadyExistsError(
            f"Document {kb_id}/{doc_id} already exists. Set overwrite=True to replace."
        )

    # 合并 metadata（source 单独存一份，便于 filter）
    full_meta = dict(metadata or {})
    if source is not None:
        full_meta.setdefault("source", source)

    doc = DocumentCreate(
        kb_id=kb_id,
        doc_id=doc_id,
        title=title,
        content=content,
        source=source,
        metadata=full_meta,
    )
    meta = await ctx.manager.add_document(doc)
    logger.info("[add_document] %s/%s ok", kb_id, doc_id)
    return AddDocumentResult(
        doc_id=meta.doc_id,
        kb_id=meta.kb_id,
        title=meta.title,
        chunks=1,
    )


# ---- Tool: add_document_chunked ----


async def add_document_chunked(
    api_key: str,
    kb_id: str,
    doc_id: str,
    title: str,
    content: str,
    source: str | None = None,
    metadata: dict | None = None,
    parser_config: dict | None = None,
    overwrite: bool = False,
    ctx: MCPContext | None = None,
) -> AddDocumentResult:
    """添加一个文档到指定知识库，用 NodeParser 切块后入库。

    与 add_document 的区别:
      - add_document: 整篇作为一个 chunk（简单）
      - add_document_chunked: 按 NodeParser 切块（更细粒度，检索更准）

    parser_config 示例:
      {
        "strategy": "sentence",   # sentence / token / semantic 等
        "chunk_size": 512,
        "chunk_overlap": 50,
      }
    """
    ctx = _get_ctx(ctx)
    await _check_write(ctx, api_key, kb_id)

    if not doc_id or not title:
        raise InvalidParameterError("doc_id and title must not be empty")
    if not content:
        raise InvalidParameterError("content must not be empty")

    existing = ctx.manager.get_document(kb_id, doc_id)
    if existing is not None and not overwrite:
        raise DocumentAlreadyExistsError(
            f"Document {kb_id}/{doc_id} already exists. Set overwrite=True to replace."
        )

    # 如果 overwrite，先删旧的
    if existing is not None:
        await ctx.manager.delete_document(kb_id, doc_id)

    # 合并 metadata
    full_meta = dict(metadata or {})
    if source is not None:
        full_meta.setdefault("source", source)

    chunks = await ctx.manager.add_document_chunked(
        kb_id=kb_id,
        doc_id=doc_id,
        title=title,
        content=content,
        source=source or "",
        parser_config=parser_config,
    )
    # add_document_chunked 内部已经把 DocumentMeta（含 chunk_count / char_count /
    # parser / content）写到 _docs，这里不要再覆盖（覆盖会清掉 chunk_count=0/char_count=0
    # 等关键字段）。
    # 如果用户传了 metadata，合并到已有 meta 上。
    existing = ctx.manager.get_document(kb_id, doc_id)
    if existing is not None and full_meta:
        merged = existing.model_copy(update={"metadata": {**existing.metadata, **full_meta}})
        async with ctx.manager._lock:  # type: ignore[attr-defined]
            ctx.manager._docs[kb_id][doc_id] = merged  # type: ignore[attr-defined]

    logger.info("[add_document_chunked] %s/%s -> %d chunks", kb_id, doc_id, chunks)
    return AddDocumentResult(doc_id=doc_id, kb_id=kb_id, title=title, chunks=chunks)


# ---- Tool: add_documents_batch ----


async def add_documents_batch(
    api_key: str,
    kb_id: str,
    documents: list[dict],
    overwrite: bool = False,
    ctx: MCPContext | None = None,
) -> list[AddDocumentResult]:
    """批量添加文档到指定知识库（每个文档整篇作为一个 chunk）。

    documents: 列表，每项为 {"doc_id", "title", "content", "source"?, "metadata"?}
    失败的单条文档不影响其他（返回的 list 不会包含失败的项，错误通过日志记录）。
    """
    ctx = _get_ctx(ctx)
    await _check_write(ctx, api_key, kb_id)

    if not documents:
        return []
    if len(documents) > 500:
        raise InvalidParameterError("batch size must be <= 500")

    results: list[AddDocumentResult] = []
    for d in documents:
        try:
            r = await add_document(
                api_key=api_key,
                kb_id=kb_id,
                doc_id=d["doc_id"],
                title=d["title"],
                content=d["content"],
                source=d.get("source"),
                metadata=d.get("metadata"),
                overwrite=overwrite,
                ctx=ctx,
            )
            results.append(r)
        except Exception as e:
            logger.warning(
                "[add_documents_batch] skip %s: %s", d.get("doc_id", "?"), e
            )
    logger.info(
        "[add_documents_batch] %s ok=%d total=%d",
        kb_id,
        len(results),
        len(documents),
    )
    return results


# ---- Tool: delete_document ----


async def delete_document(
    api_key: str,
    kb_id: str,
    doc_id: str,
    ctx: MCPContext | None = None,
) -> DeleteResult:
    """删除指定文档（向量 + meta 一起删）。

    异常:
      DocumentNotFoundError: 文档不存在
      PermissionDenied: 无写权限
    """
    ctx = _get_ctx(ctx)
    await _check_write(ctx, api_key, kb_id)

    deleted = await ctx.manager.delete_document(kb_id, doc_id)
    if not deleted:
        raise DocumentNotFoundError(f"Document not found: {kb_id}/{doc_id}")
    logger.info("[delete_document] %s/%s", kb_id, doc_id)
    return DeleteResult(doc_id=doc_id, kb_id=kb_id, deleted=True)
