"""发现类 Tool：列出可访问的 KB、查看 KB 详情。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..context import MCPContext
from ..errors import KBNotFoundError


class KBInfo(BaseModel):
    """知识库摘要信息。"""

    kb_id: str
    name: str
    description: str
    department: str
    enabled: bool
    document_count: int


class KBDetail(BaseModel):
    """知识库详情。"""

    kb_id: str
    name: str
    description: str
    department: str
    enabled: bool
    document_count: int
    embedding: str
    reranker: str | None
    extra: dict = Field(default_factory=dict)


async def list_accessible_kbs(
    api_key: str = "",
    ctx: MCPContext | None = None,
) -> list[KBInfo]:
    """列出当前用户可访问的所有知识库及其描述。

    适用场景:
      - 任何检索操作的第一步（强烈建议先调用）
      - 不确定有哪些 KB 可用
      - 需要根据 KB 描述判断该查哪个

    参数:
      api_key: 用户凭证（占位实现：可空；非空时 user_id=api_key）。
        是否强制要求由 ServerSettings.mcp_require_api_key 控制（默认 False）。

    返回:
      list[KBInfo]: 知识库列表，每个元素包含：
        - kb_id: 知识库唯一标识
        - name: 显示名
        - description: 内容描述（Agent 据此判断相关性）
        - department: 所属部门
        - document_count: 文档数量
    """
    if ctx is None:
        from ..context import MCPContext as _Ctx

        ctx = _Ctx.default()
    user = await ctx.auth.resolve(api_key)
    summaries = await ctx.manager.list_summaries()

    return [
        KBInfo(
            kb_id=s.kb_id,
            name=s.name,
            description=s.description,
            department=s.department,
            enabled=s.enabled,
            document_count=s.document_count,
        )
        for s in summaries
        if s.kb_id in user.accessible_kbs and s.enabled
    ]


async def get_kb_info(
    api_key: str = "",
    kb_id: str = "",
    ctx: MCPContext | None = None,
) -> KBDetail:
    """获取指定知识库的详细信息。

    适用场景:
      - 在调用 search_kb 之前确认 KB 内容范围
      - 查看 KB 的技术细节（embedding、reranker 等）

    参数:
      api_key: 用户凭证（可空）
      kb_id: 知识库 ID

    异常:
      KBNotFoundError: 知识库不存在或用户无权访问
    """
    if ctx is None:
        from ..context import MCPContext as _Ctx

        ctx = _Ctx.default()
    user = await ctx.auth.resolve(api_key)
    ctx.auth.check_kb_access(user, kb_id)

    summary = await ctx.manager.get_summary(kb_id)
    if summary is None:
        raise KBNotFoundError(f"KB not found: {kb_id}")

    # 从 registry 取详细配置
    from ...knowledge_bases.registry import get_registry

    registry = get_registry()
    cfg = registry.get_required(kb_id)

    return KBDetail(
        kb_id=cfg.kb_id,
        name=cfg.name,
        description=cfg.description,
        department=cfg.department,
        enabled=cfg.enabled,
        document_count=summary.document_count,
        embedding=cfg.embedding,
        reranker=cfg.reranker,
        extra=cfg.extra,
    )
