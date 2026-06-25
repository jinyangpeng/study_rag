"""管理面 REST API：KB CRUD、文档管理、检索调试、健康检查。

端点（OpenAPI 标签: admin）：
  GET    /admin/kbs                          列出所有 KB
  POST   /admin/kbs                          新建 KB（UI 配置 embedding/reranker）
  PATCH  /admin/kbs/{kb_id}                  更新 KB（部分字段）
  DELETE /admin/kbs/{kb_id}                  删除 KB（不可恢复）
  GET    /admin/kbs/{kb_id}                  获取 KB 详情
  GET    /admin/embedders                    列出可用 embedder（下拉用）
  GET    /admin/rerankers                    列出可用 reranker（下拉用）
  GET    /admin/kbs/{kb_id}/documents        列出 KB 文档
  POST   /admin/kbs/{kb_id}/documents        添加文档（整篇一个 chunk）
  POST   /admin/kbs/{kb_id}/documents/chunked       添加文档（NodeParser 切块）
  POST   /admin/kbs/{kb_id}/documents/batch         批量添加
  GET    /admin/kbs/{kb_id}/documents/{doc_id}      获取单个文档
  DELETE /admin/kbs/{kb_id}/documents/{doc_id}      删除文档
  POST   /admin/kbs/{kb_id}/search           检索（管理调试用，与 MCP 共享后端）
  GET    /admin/health/detailed              详细健康检查

鉴权:
  - 简单 Bearer Token 鉴权
  - 通过环境变量 STUDY_RAG_ADMIN_TOKEN 配置（未设置则不启用，开发模式）
  - 通过依赖 admin_auth_dep 注入

限流:
  - 默认按 client IP（X-Forwarded-For > X-Real-IP > client.host）限流
  - 配置：ServerSettings.admin_ratelimit_capacity / admin_ratelimit_per_sec
  - 超出 → 429 Too Many Requests，附 Retry-After 头
"""

from __future__ import annotations

import os
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..jobs import (
    JobManager,
    JobStatus,
    run_chunking_pipeline,
)
from ..knowledge_bases.manager import (
    ComponentUnavailableError,
    DocumentCreate,
    DocumentMeta,
    KnowledgeBaseManager,
    KnowledgeBaseSummary,
    build_default_manager,
    delete_kb_collection,
    list_available_embedders,
    list_available_rerankers,
)
from ..knowledge_bases.models import (
    EmbedderInfo,
    KnowledgeBaseConfig,
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    RerankerInfo,
)
from ..knowledge_bases.registry import (
    create_kb,
    delete_kb,
    get_registry,
    update_kb,
)
from ..mcp.context import MCPContext
from ..mcp.tools.search import search_kb as mcp_search_kb
from ..observability.logging import get_logger
from ..observability.metrics import AdminMetrics, get_metrics
from ..observability.ratelimit import get_admin_limiter

log = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])
security = HTTPBearer(auto_error=False)


# ---- 鉴权 ----


def _admin_token() -> str | None:
    return os.environ.get("STUDY_RAG_ADMIN_TOKEN")


def admin_auth_dep(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
) -> str:
    """验证 admin Bearer token。

    - 未配置 STUDY_RAG_ADMIN_TOKEN 时：不启用鉴权（开发环境）
    - 配置了：必须传匹配的 Bearer token
    """
    expected = _admin_token()
    if expected is None:
        return "dev"
    if creds is None or creds.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return creds.credentials


# ---- 限流 ----


def _client_key(request: Request) -> str:
    """从 request 提取限流维度（IP）。优先信任反向代理头。"""
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "unknown"


def admin_ratelimit_dep(request: Request) -> str:
    """Admin 限流依赖（按 client IP）。

    Returns:
        str: 用于计数的 client key

    Raises:
        HTTPException 429: 触发限流
    """
    key = _client_key(request)
    limiter = get_admin_limiter()
    if not limiter.allow(key):
        retry_after = max(1, int(limiter.retry_after(key) + 0.999))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for {key}",
            headers={"Retry-After": str(retry_after)},
        )
    return key


# ---- manager 依赖 ----


def get_manager() -> KnowledgeBaseManager:
    return build_default_manager()


# ---- 端点 ----


@router.get(
    "/kbs",
    response_model=list[KnowledgeBaseSummary],
    summary="列出所有知识库",
    description=(
        "返回所有已配置 KB 的摘要（kb_id、name、description、department、enabled、document_count）。\n\n"
        "用于管理界面或 Agent 端做 KB 发现。"
    ),
)
async def list_kbs(
    _: Annotated[str, Depends(admin_auth_dep)],
    __: Annotated[str, Depends(admin_ratelimit_dep)],
) -> list[KnowledgeBaseSummary]:
    """列出所有知识库。"""
    get_metrics().inc(AdminMetrics.REQUESTS, {"endpoint": "list_kbs"})
    return await get_manager().list_summaries()


@router.get(
    "/kbs/{kb_id}",
    response_model=KnowledgeBaseSummary,
    summary="获取知识库详情",
    responses={404: {"description": "KB 不存在"}},
)
async def get_kb(
    kb_id: str,
    _: Annotated[str, Depends(admin_auth_dep)],
    __: Annotated[str, Depends(admin_ratelimit_dep)],
) -> KnowledgeBaseSummary:
    """获取知识库详情。"""
    summary = await get_manager().get_summary(kb_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"KB not found: {kb_id}")
    return summary


# ---- KB CRUD（管理面：创建 / 更新 / 删除） ----


@router.post(
    "/kbs",
    response_model=KnowledgeBaseConfig,
    status_code=status.HTTP_201_CREATED,
    summary="创建知识库",
    description=(
        "动态新建一个 KB，立即生效（无需重启服务）。\n\n"
        "**流程**：\n"
        "1. 校验 kb_id 格式（`^[a-z][a-z0-9_]*$`）\n"
        "2. 校验 embedding / reranker 配置名是否存在\n"
        "3. 写入 `configs/knowledge_bases.yaml`（原子写）\n"
        "4. 触发 `manager.init_kb()` 建 collection（依赖缺失时 KB 会被 skip，但创建本身成功）\n\n"
        "**注意**：\n"
        "- 维度由 embedding 配置决定，不需要在请求里传\n"
        "- `embedding` 引用未加载的 provider 时 KB 会被 skip（warn log），先装依赖再 init\n"
        "- 同一 kb_id 已存在返回 409"
    ),
    responses={
        400: {"description": "参数校验失败（如 kb_id 格式不对、embedding 不存在）"},
        409: {"description": "kb_id 已存在"},
    },
)
async def create_kb_endpoint(
    payload: KnowledgeBaseCreate,
    _: Annotated[str, Depends(admin_auth_dep)],
    __: Annotated[str, Depends(admin_ratelimit_dep)],
) -> KnowledgeBaseConfig:
    """创建知识库。"""
    # 校验 embedding 配置存在
    available = {e["name"] for e in list_available_embedders()}
    if payload.embedding not in available:
        raise HTTPException(
            status_code=400,
            detail=(
                f"embedding '{payload.embedding}' not in embeddings.yaml. "
                f"Available: {sorted(available)}"
            ),
        )
    # 校验 reranker 配置存在（None 表示不要重排）
    if payload.reranker is not None:
        available_r = {r["name"] for r in list_available_rerankers()}
        if payload.reranker not in available_r:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"reranker '{payload.reranker}' not in reranker.yaml. "
                    f"Available: {sorted(available_r)}"
                ),
            )

    # 写入 yaml
    try:
        cfg = create_kb(payload)
    except Exception as e:
        msg = str(e)
        if "already exists" in msg:
            raise HTTPException(status_code=409, detail=msg) from e
        raise HTTPException(status_code=400, detail=msg) from e

    # 触发 init_kb（异步，失败也不影响创建成功）
    try:
        await get_manager().init_kb(cfg.kb_id)
    except Exception as e:  # noqa: BLE001
        log.warning("init_kb_after_create_failed", kb_id=cfg.kb_id, error=str(e))

    get_metrics().inc(
        AdminMetrics.REQUESTS, {"endpoint": "create_kb", "status": "ok"}
    )
    return cfg


@router.patch(
    "/kbs/{kb_id}",
    response_model=KnowledgeBaseConfig,
    summary="更新知识库（部分字段）",
    description=(
        "只更新传入的字段。**不支持**改 `embedding` / `collection` / `kb_id`"
        "（这三者改了需要重建 collection，破坏数据）。\n\n"
        "可改字段：name / description / department / reranker / enabled"
    ),
    responses={
        400: {"description": "参数校验失败或试图改不可改字段"},
        404: {"description": "KB 不存在"},
    },
)
async def update_kb_endpoint(
    kb_id: str,
    patch: KnowledgeBaseUpdate,
    _: Annotated[str, Depends(admin_auth_dep)],
    __: Annotated[str, Depends(admin_ratelimit_dep)],
) -> KnowledgeBaseConfig:
    """更新知识库。"""
    # 校验 reranker
    if patch.reranker is not None:
        available_r = {r["name"] for r in list_available_rerankers()}
        if patch.reranker not in available_r:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"reranker '{patch.reranker}' not in reranker.yaml. "
                    f"Available: {sorted(available_r)}"
                ),
            )
    try:
        new_cfg = update_kb(kb_id, patch)
    except Exception as e:
        msg = str(e)
        if "Unknown kb_id" in msg:
            raise HTTPException(status_code=404, detail=msg) from e
        raise HTTPException(status_code=400, detail=msg) from e

    get_metrics().inc(
        AdminMetrics.REQUESTS, {"endpoint": "update_kb", "status": "ok"}
    )
    return new_cfg


@router.delete(
    "/kbs/{kb_id}",
    summary="删除知识库",
    description=(
        "删除 KB 配置 + drop vector store collection + 清空 in-memory 文档。\n\n"
        "⚠️ **不可恢复**，所有文档向量会丢失。\n"
        "如果只想临时停用，用 `PATCH /kbs/{kb_id}` 把 `enabled` 设为 false。"
    ),
    responses={
        200: {"description": "删除成功，返回被删除的 KB"},
        404: {"description": "KB 不存在"},
    },
)
async def delete_kb_endpoint(
    kb_id: str,
    _: Annotated[str, Depends(admin_auth_dep)],
    __: Annotated[str, Depends(admin_ratelimit_dep)],
) -> KnowledgeBaseConfig:
    """删除知识库。"""
    try:
        removed = delete_kb(kb_id)
    except Exception as e:
        msg = str(e)
        if "Unknown kb_id" in msg:
            raise HTTPException(status_code=404, detail=msg) from e
        raise HTTPException(status_code=400, detail=msg) from e

    # drop collection + 清 in-memory 状态
    await delete_kb_collection(kb_id)

    get_metrics().inc(
        AdminMetrics.REQUESTS, {"endpoint": "delete_kb", "status": "ok"}
    )
    return removed


@router.get(
    "/embedders",
    response_model=list[EmbedderInfo],
    summary="列出可用 embedder 配置",
    description=(
        "返回 `embeddings.yaml` 里所有 embedder 配置（不限于已加载的）。\n\n"
        "**给管理前端用**：新建 KB 时下拉选 embedder。\n"
        "`loaded: false` 表示当前 manager 没加载（依赖未装），选了也不会报错，"
        "只是 KB init 时会被 skip（warn log）。\n\n"
        "要解锁未加载的 embedder：装对应依赖（如 `pip install study-rag[embedding-bge]`）"
        "或运行对应的本地服务（如 TEI 部署 BGE）。"
    ),
)
async def list_embedders_endpoint(
    _: Annotated[str, Depends(admin_auth_dep)],
    __: Annotated[str, Depends(admin_ratelimit_dep)],
) -> list[EmbedderInfo]:
    """列出可用 embedder。"""
    return [EmbedderInfo(**e) for e in list_available_embedders()]


@router.get(
    "/rerankers",
    response_model=list[RerankerInfo],
    summary="列出可用 reranker 配置",
    description=(
        "返回 `reranker.yaml` 里所有 reranker 配置。\n\n"
        "**给管理前端用**：新建 KB 时下拉选 reranker。"
    ),
)
async def list_rerankers_endpoint(
    _: Annotated[str, Depends(admin_auth_dep)],
    __: Annotated[str, Depends(admin_ratelimit_dep)],
) -> list[RerankerInfo]:
    """列出可用 reranker。"""
    return [RerankerInfo(**r) for r in list_available_rerankers()]


@router.post(
    "/kbs/{kb_id}/documents",
    response_model=DocumentMeta,
    summary="添加文档（整篇一个 chunk）",
    description=(
        "把整篇文档作为一个向量入库。\n\n"
        "适合短文档或已经预处理好的结构化数据。\n"
        "如果需要切块，请用 `POST /kbs/{kb_id}/documents/chunked`。"
    ),
    responses={
        400: {"description": "kb_id 路径与 body 不一致"},
        409: {"description": "doc_id 已存在且未传 overwrite=true"},
    },
)
async def add_document(
    kb_id: str,
    doc: DocumentCreate,
    _: Annotated[str, Depends(admin_auth_dep)],
    __: Annotated[str, Depends(admin_ratelimit_dep)],
) -> DocumentMeta:
    """添加文档到知识库（整篇作为一个 chunk）。"""
    if doc.kb_id != kb_id:
        raise HTTPException(
            status_code=400,
            detail=f"kb_id mismatch: path={kb_id}, body={doc.kb_id}",
        )
    start = time.perf_counter()
    try:
        meta = await get_manager().add_document(doc)
        get_metrics().inc(AdminMetrics.DOCUMENTS, {"op": "add", "status": "ok"})
        return meta
    except ComponentUnavailableError as e:
        # KB 的 embedder / reranker / vector store 未就绪 → 4xx，提示用户修
        get_metrics().inc(AdminMetrics.DOCUMENTS, {"op": "add", "status": "error"})
        log.warning("admin_add_doc_component_unavailable", kb_id=kb_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        get_metrics().inc(AdminMetrics.DOCUMENTS, {"op": "add", "status": "error"})
        log.warning("admin_add_doc_failed", kb_id=kb_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"add_document failed: {e}") from e
    finally:
        get_metrics().observe(
            AdminMetrics.LATENCY,
            (time.perf_counter() - start) * 1000,
            {"endpoint": "add_document"},
        )


@router.post(
    "/kbs/{kb_id}/documents/chunked",
    summary="添加文档（NodeParser 切块）",
    description=(
        "用 LlamaIndex 的 NodeParser 把文档切块后入库，检索粒度更细。\n\n"
        "**Query 参数**（路径上）:\n"
        "- `doc_id`: 文档 ID（必填）\n"
        "- `title`: 标题（必填）\n"
        "- `content`: 正文（必填）\n\n"
        "**Body**（JSON）:\n"
        "```json\n"
        "{\n"
        '  "parser_config": {"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 50},\n'
        '  "source": "wiki",\n'
        '  "overwrite": false\n'
        "}\n"
        "```"
    ),
    responses={
        200: {
            "description": "成功",
            "content": {
                "application/json": {
                    "example": {"kb_id": "rd_frontend", "doc_id": "d1", "chunks": 6}
                }
            },
        },
        409: {"description": "doc_id 已存在且未传 overwrite=true"},
    },
)
async def add_document_chunked(
    kb_id: str,
    doc_id: str,
    title: str,
    content: str,
    request: Request,
    _: Annotated[str, Depends(admin_auth_dep)],
    __: Annotated[str, Depends(admin_ratelimit_dep)],
) -> dict[str, Any]:
    """添加文档（用 NodeParser 切块）。"""
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        body = {}
    parser_config = body.get("parser_config")
    parser_name = body.get("parser_name")  # 命名 parser（如 'sentence_512'）
    source = body.get("source", "")
    overwrite = body.get("overwrite", False)

    manager = get_manager()
    if not overwrite and manager.get_document(kb_id, doc_id) is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Document already exists: {doc_id}",
        )

    n_chunks = await manager.add_document_chunked(
        kb_id=kb_id,
        doc_id=doc_id,
        title=title,
        content=content,
        source=source,
        parser_config=parser_config,
        parser_name=parser_name,
    )
    get_metrics().inc(AdminMetrics.DOCUMENTS, {"op": "add_chunked", "status": "ok"})
    return {"kb_id": kb_id, "doc_id": doc_id, "chunks": n_chunks}


# ---- Parsers / Preview / Upload (Phase 2) ----


@router.get(
    "/parsers",
    summary="列出可用 parser（前端下拉用）",
    description=(
        "返回 `configs/llamaindex.yaml` 里所有命名 parser 的元信息（name / strategy / "
        "chunk_size / chunk_overlap / paragraph_separator）。\n\n"
        "前端在「添加文档」弹窗里下拉选；与 embedder / reranker 不同，"
        "parser 不需要单独加载（依赖 llama-index-core 即可），所有策略都可用。\n\n"
        "**注意**：包含 `semantic` 策略时需要 embed_model；preview 端点不支持 "
        "semantic（会返回 400），完整入库由 `add_document_from_upload` 处理，"
        "自动注入 KB 的 embedder。"
    ),
)
async def list_parsers_endpoint(
    _: Annotated[str, Depends(admin_auth_dep)],
    __: Annotated[str, Depends(admin_ratelimit_dep)],
) -> list[dict[str, Any]]:
    """列出可用 parser。"""
    from ..capabilities.llamaindex import get_parser_registry

    get_metrics().inc(AdminMetrics.REQUESTS, {"endpoint": "list_parsers"})
    return get_parser_registry().list_summaries()


@router.post(
    "/kbs/{kb_id}/documents/preview-chunk",
    summary="预览分块（不入库）",
    description=(
        "把 `content` 按指定 parser 切块，返回每个 chunk 的内容/大小/metadata，"
        "不写入向量库。\n\n"
        "前端在「添加文档」弹窗提交前用这个让用户看到切分效果，确认后再真正入库。\n\n"
        "**Body**:\n"
        "```json\n"
        "{\n"
        '  "content": "...",\n'
        '  "parser": "sentence_512",\n'
        '  "title": "preview"\n'
        "}\n"
        "```\n\n"
        "**注意**：`semantic` 策略会消费 embedder 调用（用 KB 配置的 embedder），"
        "preview 阶段不写向量库但会真的跑一次 embed。"
    ),
    responses={
        200: {
            "description": "成功",
            "content": {
                "application/json": {
                    "example": {
                        "parser": "sentence_512",
                        "chunks": [
                            {
                                "chunk_index": 0,
                                "text": "First part...",
                                "char_count": 200,
                                "metadata": {"title": "preview"},
                            }
                        ],
                        "total_chunks": 3,
                        "total_chars": 600,
                    }
                }
            },
        },
        400: {
            "description": "content 为空 / parser 不存在 / semantic 缺 embedder"
        },
    },
)
async def preview_chunk(
    kb_id: str,
    request: Request,
    _: Annotated[str, Depends(admin_auth_dep)],
    __: Annotated[str, Depends(admin_ratelimit_dep)],
) -> dict[str, Any]:
    """预览分块。"""
    body = await request.json()
    content = body.get("content", "")
    parser_name = body.get("parser") or "sentence_512"
    title = body.get("title", "preview")

    if not content or not content.strip():
        raise HTTPException(status_code=400, detail="content is empty")

    # 校验 KB 存在（KB 不存在时 get_required 抛错）
    manager = get_manager()
    try:
        manager._registry.get_required(kb_id)  # type: ignore[attr-defined]
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"KB not found: {kb_id}") from e

    from ..capabilities.llamaindex import (
        LIEmbeddingAdapter,
        get_parser_registry,
    )

    # semantic 策略需要 embedder：尝试从 KB 拿 embedder
    embed_model: Any = None
    try:
        embedder = manager.get_embedder(kb_id)
        embed_model = LIEmbeddingAdapter(embedder)
    except ComponentUnavailableError as e:
        # KB 存在但 embedder 没加载好 → 4xx，提示用户修
        if parser_name == "semantic":
            raise HTTPException(status_code=400, detail=str(e)) from e
        # 其它策略不强制要 embedder，静默忽略
    except Exception:
        # 其它未预期错误：non-semantic 仍允许；semantic 会到 factory 那里报 ValueError
        pass

    try:
        factory = get_parser_registry().get(parser_name, embed_model=embed_model)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        nodes = factory.parse(content, doc_id="preview", title=title)
    except ValueError as e:
        # semantic 缺 embed_model 时报
        raise HTTPException(status_code=400, detail=str(e)) from e

    get_metrics().inc(
        AdminMetrics.REQUESTS,
        {"endpoint": "preview_chunk", "parser": parser_name, "status": "ok"},
    )
    return {
        "parser": parser_name,
        "chunks": [
            {
                "chunk_index": n.chunk_index,
                "text": n.text,
                "char_count": len(n.text),
                "metadata": n.metadata,
            }
            for n in nodes
        ],
        "total_chunks": len(nodes),
        "total_chars": len(content),
    }


@router.post(
    "/kbs/{kb_id}/documents/upload",
    status_code=status.HTTP_202_ACCEPTED,
    summary="上传文件入库（multipart，异步）",
    description=(
        "支持格式：`txt` / `md` / `html` / `pdf` / `docx`（取决于 pyproject optional deps）。\n\n"
        "**异步执行**：上传是耗时操作（切块 + embedding + 写库），本端点会立即返回"
        '``{"job_id": "...", "status": "pending"}``，实际处理在后台 asyncio.Task 里跑。\n\n'
        "前端通过 ``GET /admin/jobs/{job_id}`` 轮询状态（推荐间隔 1s）。\n\n"
        "**Form 字段**：\n"
        "- `file`: 文件（必填）\n"
        "- `doc_id`: 文档 ID（必填，KB 内唯一）\n"
        "- `title`: 标题（必填）\n"
        "- `parser`: 命名 parser，如 `sentence_512`（可选，默认 `sentence_512`）\n"
        "- `source`: 来源标识（可选）\n"
        "- `overwrite`: `true` / `false`（可选，默认 `false`）\n\n"
        "**错误码**：\n"
        "- 400：参数缺失、文件格式不支持、parser 不存在\n"
        "- 404：KB 不存在\n"
        "- 409：doc_id 已存在且未传 `overwrite=true`"
    ),
    responses={
        202: {
            "description": "已接收，后台处理中",
            "content": {
                "application/json": {
                    "example": {
                        "job_id": "abc123def456",
                        "status": "pending",
                        "kb_id": "rd_frontend",
                        "doc_id": "react_perf_001",
                    }
                }
            },
        },
        400: {"description": "参数缺失 / 文件格式不支持 / parser 不存在"},
        404: {"description": "KB 不存在"},
        409: {"description": "doc_id 已存在且未传 overwrite=true"},
    },
)
async def upload_document(
    kb_id: str,
    request: Request,
    _: Annotated[str, Depends(admin_auth_dep)],
    __: Annotated[str, Depends(admin_ratelimit_dep)],
) -> dict[str, Any]:
    """上传文件入库（异步）。

    立即返回 job_id；实际切块/embedding/写库在后台跑。
    """
    from ..capabilities.llamaindex import UnsupportedFormatError, read_document

    form = await request.form()
    file = form.get("file")
    if file is None or not hasattr(file, "filename") or not file.filename:
        raise HTTPException(status_code=400, detail="file is required")
    from starlette.datastructures import UploadFile

    if not isinstance(file, UploadFile):
        raise HTTPException(status_code=400, detail="file must be an uploaded file")
    doc_id = str(form.get("doc_id", "") or "").strip()
    title = str(form.get("title", "") or "").strip()
    parser_name = str(form.get("parser") or "sentence_512")
    source = str(form.get("source", "") or "")
    overwrite_raw = str(form.get("overwrite", "false") or "false")
    overwrite = overwrite_raw.lower() == "true"

    if not doc_id:
        raise HTTPException(status_code=400, detail="doc_id is required")
    if not title:
        raise HTTPException(status_code=400, detail="title is required")

    # 读文件（同步发生在请求上下文；后续切块/embedding 在后台）
    data = await file.read()
    try:
        text, reader_meta = read_document(
            data,
            filename=file.filename,
            mime=file.content_type,
        )
    except UnsupportedFormatError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # 校验 parser 存在（避免后台跑一半抛 KeyError）
    from ..capabilities.llamaindex import get_parser_registry

    try:
        get_parser_registry().get(parser_name)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # KB 存在性
    manager = get_manager()
    try:
        manager._registry.get_required(kb_id)  # type: ignore[attr-defined]
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"KB not found: {kb_id}") from e

    # overwrite 校验
    if not overwrite and manager.get_document(kb_id, doc_id) is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Document already exists: {kb_id}/{doc_id}",
        )

    # 拿 job manager（app.state.jobs 在 create_app 里 wire）
    jobs: JobManager = request.app.state.jobs  # type: ignore[attr-defined]

    # 准备 embedder_registry 和 parser_registry 引用（供 pipeline 闭包用）
    from ..capabilities.llamaindex import get_parser_registry as _gpr

    embedder_registry = manager._embedders  # type: ignore[attr-defined]
    parser_registry = _gpr()

    # 提交后台任务
    async def _runner(
        job_id: str,
        on_progress: Any,
        is_cancelled: Any,
    ) -> None:
        await run_chunking_pipeline(
            job_id=job_id,
            on_progress=on_progress,
            is_cancelled=is_cancelled,
            file_content=text,
            filename=file.filename or "upload.bin",
            doc_id=doc_id,
            title=title,
            parser_name=parser_name,
            kb_id=kb_id,
            source=source,
            metadata={**reader_meta, "uploaded": True},
            embedder_registry=embedder_registry,
            parser_registry=parser_registry,
            kb_manager=manager,
        )

    jid = await jobs.submit(
        "upload_doc",
        _runner,
        kb_id=kb_id,
        doc_id=doc_id,
        filename=file.filename,
    )

    get_metrics().inc(
        AdminMetrics.DOCUMENTS, {"op": "upload", "status": "queued"}
    )
    get_metrics().inc(
        AdminMetrics.REQUESTS,
        {
            "endpoint": "upload",
            "kb_id": kb_id,
            "parser": parser_name,
            "status": "queued",
        },
    )
    return {
        "job_id": jid,
        "status": JobStatus.PENDING.value,
        "kb_id": kb_id,
        "doc_id": doc_id,
        "format": reader_meta["format"],
        "size_bytes": reader_meta["size_bytes"],
        "parser": parser_name,
    }


@router.post(
    "/kbs/{kb_id}/documents/batch",
    summary="批量添加文档",
    description=(
        "单次最多 500 条；失败的 doc 不会影响其他。\n\n"
        "返回 succeeded / failed / counts 三个字段，failed 数组中每条携带 `error` 字段。"
    ),
    responses={
        200: {
            "description": "成功",
            "content": {
                "application/json": {
                    "example": {
                        "succeeded": ["doc1", "doc2"],
                        "failed": [{"doc_id": "doc3", "error": "already exists"}],
                        "counts": {"ok": 2, "fail": 1},
                    }
                }
            },
        },
        400: {"description": "documents 为空或超过 500 条"},
    },
)
async def add_documents_batch(
    kb_id: str,
    request: Request,
    _: Annotated[str, Depends(admin_auth_dep)],
    __: Annotated[str, Depends(admin_ratelimit_dep)],
) -> dict[str, Any]:
    """批量添加文档。

    body:
      {
        "documents": [{"doc_id", "title", "content", "source"?, "metadata"?}, ...],
        "overwrite": false
      }
    """
    body = await request.json()
    documents = body.get("documents", [])
    overwrite = body.get("overwrite", False)
    if not isinstance(documents, list) or not documents:
        raise HTTPException(status_code=400, detail="documents must be a non-empty list")
    if len(documents) > 500:
        raise HTTPException(status_code=400, detail="too many documents (limit 500)")

    manager = get_manager()
    succeeded: list[str] = []
    failed: list[dict[str, str]] = []
    for d in documents:
        try:
            doc_id = d.get("doc_id") or ""
            title = d.get("title") or doc_id
            content = d.get("content", "")
            if not doc_id or not content:
                failed.append({"doc_id": doc_id, "error": "missing doc_id or content"})
                continue
            if not overwrite and manager.get_document(kb_id, doc_id) is not None:
                failed.append({"doc_id": doc_id, "error": "already exists"})
                continue
            await manager.add_document(
                DocumentCreate(
                    kb_id=kb_id,
                    doc_id=doc_id,
                    title=title,
                    content=content,
                    source=d.get("source"),
                    metadata=d.get("metadata"),
                )
            )
            succeeded.append(doc_id)
        except Exception as e:
            log.warning("batch_add_failed", doc_id=d.get("doc_id"), error=str(e))
            failed.append({"doc_id": d.get("doc_id", ""), "error": str(e)})

    get_metrics().inc(
        AdminMetrics.DOCUMENTS,
        {"op": "add_batch", "status": "ok" if not failed else "partial"},
    )
    return {
        "succeeded": succeeded,
        "failed": failed,
        "counts": {"ok": len(succeeded), "fail": len(failed)},
    }


async def _enrich_doc(manager, kb_id: str, doc: DocumentMeta) -> DocumentMeta:
    """实时从 vector store 拉取 doc 的真实 chunk_count / char_count / parser。

    用于 list_documents 和 get_document 两个端点，确保无论历史数据怎么错
    （被旧 enrich 错误覆盖、写入时漏存），前端拿到的都是真实值。

    失败时回退到 doc 自身的字段（即使是历史错误值也比 0 强）。
    """
    try:
        n = await manager.get_chunk_count(kb_id, doc.doc_id)
    except Exception:  # noqa: BLE001
        n = doc.chunk_count
    try:
        total_chars = await manager.get_doc_total_chars(kb_id, doc.doc_id)
    except Exception:  # noqa: BLE001
        total_chars = doc.char_count
    try:
        real_parser = await manager.get_doc_parser(kb_id, doc.doc_id)
    except Exception:  # noqa: BLE001
        real_parser = doc.parser
    return doc.model_copy(update={
        "chunk_count": n,
        "char_count": total_chars,
        "parser": real_parser if real_parser else doc.parser,
    })


@router.get(
    "/kbs/{kb_id}/documents",
    response_model=list[DocumentMeta],
    summary="列出 KB 下的所有文档",
    description=(
        "返回该 KB 下所有已添加文档的元信息（doc_id、title、source、created_at 等），\n"
        "不返回 content（避免响应体过大）。"
    ),
    responses={404: {"description": "KB 不存在"}},
)
async def list_documents(
    kb_id: str,
    _: Annotated[str, Depends(admin_auth_dep)],
    __: Annotated[str, Depends(admin_ratelimit_dep)],
) -> list[DocumentMeta]:
    """列出 KB 下的所有文档。

    每个 DocumentMeta 的 chunk_count / char_count / parser 会被后端实时用
    vector store 覆盖（与 list/get_document 共享 enrich 逻辑）：
    - chunk_count: vector store 实际 chunk 数（可能与写入时的值不一致，
      比如重复上传后部分 chunk 失败）
    - char_count: 所有 chunk 文本长度之和（不受 content 截断 / 历史错误影响）
    - parser: chunks metadata['parser'] 的真实值（修复老数据漏存 / 存错的问题）
    """
    get_metrics().inc(AdminMetrics.REQUESTS, {"endpoint": "list_documents"})
    manager = get_manager()
    docs = manager.list_documents(kb_id)
    # 实时回填：并发拉以避免 KB 文档多时阻塞
    import asyncio

    return await asyncio.gather(*[_enrich_doc(manager, kb_id, d) for d in docs])


@router.get(
    "/kbs/{kb_id}/documents/{doc_id}",
    response_model=DocumentMeta,
    summary="获取单个文档完整内容",
    description=(
        "返回 doc 完整 metadata 与 content；用于管理界面预览或调试。\n\n"
        "**注意**：返回的 chunk_count / char_count / parser 是实时从 vector store "
        "覆盖后的值（与 list_documents 一致），不直接是内存中 _docs 的值。"
    ),
    responses={
        404: {"description": "KB 或文档不存在"},
    },
)
async def get_document(
    kb_id: str,
    doc_id: str,
    _: Annotated[str, Depends(admin_auth_dep)],
    __: Annotated[str, Depends(admin_ratelimit_dep)],
) -> DocumentMeta:
    """获取单个文档完整内容（实时 enrich）。"""
    manager = get_manager()
    doc = manager.get_document(kb_id, doc_id)
    if doc is None:
        raise HTTPException(
            status_code=404, detail=f"Document not found: {kb_id}/{doc_id}"
        )
    return await _enrich_doc(manager, kb_id, doc)


@router.get(
    "/kbs/{kb_id}/documents/{doc_id}/chunks",
    summary="查看文档的所有分块",
    description=(
        "按 doc_id 过滤向量库，返回该文档的所有 chunk 列表（按 chunk_index 升序）。\n\n"
        "支持分页：\n"
        "- `limit`: 返回的 chunk 数上限（默认 100）\n"
        "- `offset`: 分页偏移（默认 0）\n\n"
        "用于管理 UI 的「查看分块」功能，调试切块效果。"
    ),
    responses={
        404: {"description": "KB 不存在"},
        200: {"description": "返回 chunks 列表（可能为空）"},
    },
)
async def list_document_chunks(
    kb_id: str,
    doc_id: str,
    request: Request,
    _: Annotated[str, Depends(admin_auth_dep)],
    __: Annotated[str, Depends(admin_ratelimit_dep)],
) -> dict[str, Any]:
    """列出文档的所有 chunks。"""
    # 解析 query params
    qp = request.query_params
    try:
        limit = int(qp.get("limit", "100"))
        offset = int(qp.get("offset", "0"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"invalid query param: {e}") from e
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be in [1, 1000]")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    manager = get_manager()
    # 验证 KB 存在
    if manager._registry.get(kb_id) is None:
        raise HTTPException(status_code=404, detail=f"KB '{kb_id}' not found")

    try:
        total = await manager.get_chunk_count(kb_id, doc_id)
        chunks = await manager.list_chunks(kb_id, doc_id, limit=limit, offset=offset)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    get_metrics().inc(AdminMetrics.REQUESTS, {"endpoint": "list_document_chunks"})
    return {
        "kb_id": kb_id,
        "doc_id": doc_id,
        "total": total,
        "limit": limit,
        "offset": offset,
        "chunks": [c.model_dump(mode="json") for c in chunks],
    }


@router.delete(
    "/kbs/{kb_id}/documents/{doc_id}",
    summary="删除文档",
    description="从向量库中删除 doc_id 对应的全部 chunks；幂等。",
    responses={
        200: {
            "description": "删除成功",
            "content": {
                "application/json": {
                    "example": {"status": "deleted", "kb_id": "rd_frontend", "doc_id": "d1"}
                }
            },
        },
        404: {"description": "KB 或文档不存在"},
    },
)
async def delete_document(
    kb_id: str,
    doc_id: str,
    _: Annotated[str, Depends(admin_auth_dep)],
    __: Annotated[str, Depends(admin_ratelimit_dep)],
) -> dict[str, str]:
    """删除文档。"""
    ok = await get_manager().delete_document(kb_id, doc_id)
    if not ok:
        raise HTTPException(
            status_code=404, detail=f"Document not found: {kb_id}/{doc_id}"
        )
    get_metrics().inc(AdminMetrics.DOCUMENTS, {"op": "delete", "status": "ok"})
    return {"status": "deleted", "kb_id": kb_id, "doc_id": doc_id}


@router.post(
    "/kbs/{kb_id}/search",
    summary="检索（管理调试用）",
    description=(
        "与 MCP `search_kb` 共享后端，行为完全一致。\n\n"
        "**Body**:\n"
        "```json\n"
        "{\n"
        '  "query": "React 性能优化",\n'
        '  "top_k": 5,\n'
        '  "use_rerank": true,\n'
        '  "filter_expr": {"source": "wiki"}\n'
        "}\n"
        "```\n\n"
        "**filter_expr**：metadata 过滤表达式，支持 `==` / `!=` / `in` / `and` / `or`。\n"
        "示例：`{\"source\": \"wiki\"}`、`{\"year\": {\"$gte\": 2024}}`。"
    ),
    responses={
        200: {
            "description": "命中结果",
            "content": {
                "application/json": {
                    "example": {
                        "kb_id": "rd_frontend",
                        "query": "React 性能优化",
                        "hits": [
                            {
                                "doc_id": "react_perf_001",
                                "chunk_id": "c0",
                                "score": 0.87,
                                "text": "React 应用常见的性能优化点...",
                                "metadata": {"source": "wiki"},
                            }
                        ],
                        "duration_ms": 42.1,
                    }
                }
            },
        },
        400: {"description": "query 为空 / KB 不存在 / 检索失败"},
        429: {"description": "限流触发"},
    },
)
async def search_kb_admin(
    kb_id: str,
    request: Request,
    _: Annotated[str, Depends(admin_auth_dep)],
    __: Annotated[str, Depends(admin_ratelimit_dep)],
) -> dict[str, Any]:
    """检索（管理调试用，与 MCP search_kb 共享后端）。

    body:
      {
        "query": "...",
        "top_k": 5,
        "use_rerank": true,
        "filter_expr": {"source": "wiki"}  // 可选
      }
    """
    body = await request.json()
    query = body.get("query", "")
    top_k = int(body.get("top_k", 5))
    use_rerank = bool(body.get("use_rerank", True))
    filter_expr = body.get("filter_expr")

    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    # 复用 MCP 后端（保证 admin 和 MCP 行为完全一致）
    ctx = MCPContext.default()
    start = time.perf_counter()
    try:
        hits = await mcp_search_kb(
            api_key="admin",
            kb_id=kb_id,
            query=query,
            top_k=top_k,
            use_rerank=use_rerank,
            filter_expr=filter_expr,
            ctx=ctx,
        )
    except Exception as e:
        log.warning("admin_search_failed", kb_id=kb_id, error=str(e))
        get_metrics().inc(
            AdminMetrics.REQUESTS,
            {"endpoint": "search", "kb_id": kb_id, "status": "error"},
        )
        raise HTTPException(status_code=400, detail=str(e)) from e

    duration_ms = (time.perf_counter() - start) * 1000
    get_metrics().inc(
        AdminMetrics.REQUESTS,
        {"endpoint": "search", "kb_id": kb_id, "status": "ok"},
    )
    get_metrics().observe(
        AdminMetrics.LATENCY, duration_ms, {"endpoint": "search"}
    )
    return {
        "kb_id": kb_id,
        "query": query,
        "hits": [h.model_dump() for h in hits],
        "duration_ms": round(duration_ms, 2),
    }


# /admin/health/detailed 也提供（与 /health/detailed 等价）
@router.get(
    "/health/detailed",
    summary="详细健康检查（admin 命名空间下）",
    description=(
        "在 admin 命名空间下的别名；与 `/health/detailed` 行为一致。\n\n"
        "返回 KB 数量、组件加载情况、鉴权开关等。"
    ),
)
async def admin_health_detailed() -> dict[str, Any]:
    """详细健康检查（admin 前缀，便于在 admin 鉴权后访问）。"""
    manager = get_manager()
    kbs = await manager.list_summaries()
    return {
        "status": "ok",
        "kbs_total": len(kbs),
        "kbs_enabled": sum(1 for k in kbs if k.enabled),
        "embedders": len(manager._embedders),  # type: ignore[attr-defined]
        "rerankers": len(manager._rerankers),  # type: ignore[attr-defined]
        "auth_enabled": _admin_token() is not None,
        "registry_loaded": get_registry() is not None,
    }

