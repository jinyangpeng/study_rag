"""检索类 Tool：在指定 KB 中搜索。"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from ...capabilities.embedding import Embedder
from ...capabilities.reranker import Reranker
from ...capabilities.vector_store import SearchResult as VSResult
from ...capabilities.vector_store import VectorStore
from ...knowledge_bases.manager import ComponentUnavailableError
from ..context import MCPContext
from ..errors import InvalidParameterError, KBNotFoundError

logger = logging.getLogger(__name__)


class SearchHit(BaseModel):
    """MCP 返回的检索结果。"""

    doc_id: str
    title: str
    text: str
    score: float
    metadata: dict = Field(default_factory=dict)


# ---- 内部 helper ----

# 向量检索多召回数 = top_k * RERANK_OVER_FETCH
# 经验值：reranker 在 5-10x 召回范围里表现最好
RERANK_OVER_FETCH = 4


def _load_embedder_for_kb(kb_id: str, ctx: MCPContext) -> Embedder:
    """从 manager 获取 KB 对应的 embedder（共享实例）。"""
    return ctx.manager.get_embedder(kb_id)


def _load_vector_store(ctx: MCPContext) -> VectorStore:
    """从 manager 获取共享的 vector store 实例。"""
    return ctx.manager.get_vector_store()


def _load_reranker_for_kb(kb_id: str, ctx: MCPContext) -> Reranker | None:
    """从 manager 获取 KB 对应的 reranker，未配置时返回 None。"""
    return ctx.manager.get_reranker_for_kb(kb_id)


async def _rerank_with_fallback(
    reranker: Reranker | None,
    query: str,
    candidates: list[VSResult],
    top_k: int | None,
    fallback_top_k: int,
) -> list[VSResult]:
    """调用 reranker 重排，失败时降级为截断原始顺序。

    Enterprise 实践：
      - rerank 失败不应阻塞搜索（即使精度降低）
      - 记录 warning 便于排查

    参数:
      top_k: 传给 reranker 的 top_k；None 表示让 reranker 用自身配置
      fallback_top_k: 降级（无 reranker 或 rerank 失败）时截断用的数量
    """
    if not candidates:
        return []

    if reranker is None:
        # KB 未配置 reranker，直接截断
        return candidates[:fallback_top_k]

    try:
        return await reranker.rerank(query=query, results=candidates, top_k=top_k)
    except Exception as e:
        logger.warning(
            "Reranker failed for query=%r, falling back to original order: %s",
            query[:30],
            e,
        )
        return candidates[:fallback_top_k]


# ---- Tool ----

async def search_kb(
    api_key: str,
    kb_id: str,
    query: str,
    top_k: int | None = None,
    use_rerank: bool = True,
    filter_expr: dict | None = None,
    reranker_name: str | None = None,
    ctx: MCPContext | None = None,
) -> list[SearchHit]:
    """在指定知识库中检索相关内容。

    工作流:
      1. 调用 list_accessible_kbs 获取可访问的 KB
      2. 根据 KB description 判断相关性，选定 kb_id
      3. 调用本 Tool 检索

    参数:
      api_key: 用户凭证
      kb_id: 知识库 ID（命名规范: {dept}_{name}，如 rd_frontend）
      query: 检索问题
      top_k: 返回结果数量（范围 (0, 50]）。None 表示跟随 reranker 配置的 top_k；
        无 reranker 时用默认值 5。显式传值会覆盖 reranker 配置。
      use_rerank: 是否启用重排（默认 True）。当无可用 reranker 时该参数无效。
      filter_expr: 可选的 metadata 过滤条件，例如 {"source": "wiki", "year__gte": 2024}
      reranker_name: 可选，显式指定使用的 reranker 配置名（覆盖 KB 默认绑定的 reranker）。
        为 None 时使用 KB 配置的 reranker；指定时从已加载的 reranker 中按名取用。
        常用于检索调试时对比不同 reranker 的效果。仅在 use_rerank=True 时生效。

    返回:
      list[SearchHit]: 命中结果，按相关度倒序

    异常:
      KBNotFoundError: 知识库不存在
      PermissionDenied: 用户无权访问该 KB
      InvalidParameterError: 参数非法（如 filter_expr 字段名非法、reranker_name 不存在）
    """
    if ctx is None:
        from ..context import MCPContext

        ctx = MCPContext.default()

    if not query or not query.strip():
        raise InvalidParameterError("query must not be empty")
    if top_k is not None and (top_k <= 0 or top_k > 50):
        raise InvalidParameterError("top_k must be in (0, 50] or None")

    # 鉴权
    user = await ctx.auth.resolve(api_key)
    ctx.auth.check_kb_access(user, kb_id)

    # KB 必须存在
    from ...knowledge_bases.registry import get_registry

    registry = get_registry()
    cfg = registry.get(kb_id)
    if cfg is None or not cfg.enabled:
        raise KBNotFoundError(f"KB not found or disabled: {kb_id}")

    # 1. Embedding
    try:
        embedder = _load_embedder_for_kb(kb_id, ctx)
        query_vector = await embedder.embed_query(query)
    except ComponentUnavailableError as e:
        # KB 存在但 embedder 未加载：传成 InvalidParameterError，提示用户修配置/装依赖
        raise InvalidParameterError(str(e)) from e

    # 2. Reranker 选择：显式 reranker_name 覆盖 KB 默认绑定的 reranker
    #    - use_rerank=False：不重排
    #    - use_rerank=True + reranker_name：用指定 reranker（用于检索调试对比）
    #    - use_rerank=True + reranker_name=None：用 KB 配置的 reranker（原行为）
    reranker: Reranker | None = None
    if use_rerank:
        if reranker_name:
            try:
                reranker = ctx.manager.get_reranker(reranker_name)
            except ComponentUnavailableError as e:
                raise InvalidParameterError(str(e)) from e
        else:
            reranker = _load_reranker_for_kb(kb_id, ctx)

    # 解析向量召回数（embedding 返回的数量，即用户在检索测试页填的 Top K）：
    #   - 用户显式传了 top_k → 用用户的值
    #   - top_k=None → 默认 5
    recall_k = top_k if top_k is not None else 5

    # 3. Vector Search：recall_k 即向量召回数
    #    启用 reranker 时为保证 rerank 有足够候选，按 recall_k * OVER_FETCH 多召回，
    #    最终由 reranker 用自身配置的 top_k 过滤到目标数量。
    vector_store = _load_vector_store(ctx)
    candidate_k = recall_k * RERANK_OVER_FETCH if reranker else recall_k
    candidates: list[VSResult] = await vector_store.search(
        collection=cfg.collection,
        query_vector=query_vector,
        top_k=candidate_k,
        filter_expr=filter_expr,
    )

    # 3. Rerank（KB 配置了 reranker 时才生效；失败则降级为截断）
    #    始终传 top_k=None 给 reranker，让它用自身配置的 top_k 过滤
    #    （用户的 Top K 是向量召回数，不覆盖 reranker 的重排保留数）
    results = await _rerank_with_fallback(
        reranker=reranker,
        query=query,
        candidates=candidates,
        top_k=None,
        fallback_top_k=recall_k,
    )

    return [
        SearchHit(
            doc_id=r.id,
            title=r.metadata.get("title", r.id),
            text=r.text,
            score=r.score,
            metadata=r.metadata,
        )
        for r in results
    ]


async def search_all_accessible_kbs(
    api_key: str,
    query: str,
    top_k: int = 5,
    use_rerank: bool = True,
    filter_expr: dict | None = None,
    ctx: MCPContext | None = None,
) -> list[SearchHit]:
    """在所有可访问的知识库中综合检索。

    适用场景:
      - 不确定该查哪个 KB
      - 需要跨 KB 综合答案
      - 兜底检索

    返回结果按相关度倒序，包含来源 KB 信息。

    实现：每个 KB 内部已用各自的 reranker 重排，跨 KB 层仅做分数合并排序。

    filter_expr：会下推到每个 KB 的向量检索层（同一规则在所有 KB 间复用）。
    """
    if ctx is None:
        from ..context import MCPContext

        ctx = MCPContext.default()

    user = await ctx.auth.resolve(api_key)
    accessible = list(user.accessible_kbs)

    all_hits: list[SearchHit] = []
    for kb_id in accessible:
        try:
            hits = await search_kb(
                api_key=api_key,
                kb_id=kb_id,
                query=query,
                top_k=top_k,
                use_rerank=use_rerank,
                filter_expr=filter_expr,
                ctx=ctx,
            )
            # 标记来源
            for h in hits:
                h.metadata["_source_kb"] = kb_id
            all_hits.extend(hits)
        except Exception:
            # 单个 KB 失败不影响其他
            continue

    # 跨 KB 层：按分数倒序。
    # 注：不再次调用 reranker（不同 KB 的 reranker 分数量纲不同，硬合并不严谨）。
    all_hits.sort(key=lambda x: x.score, reverse=True)
    return all_hits[:top_k]
