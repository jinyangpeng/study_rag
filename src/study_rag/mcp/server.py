"""MCP Server：基于 FastMCP 注册 10 个核心 Tool + Resources + Prompts。

Tools:
  1.  list_accessible_kbs       - 发现类
  2.  get_kb_info               - 发现类
  3.  search_kb                 - 检索类（支持 filter_expr）
  4.  search_all_accessible_kbs - 检索类（支持 filter_expr）
  5.  get_document              - 文档类
  6.  list_documents            - 文档类（管理）
  7.  add_document              - 文档类（管理，需写权限）
  8.  add_document_chunked      - 文档类（管理，LlamaIndex 切块）
  9.  add_documents_batch       - 文档类（管理，批量）
  10. delete_document           - 文档类（管理）

Resources (URI 形式暴露给 agent):
  - kb://all                       列出所有 KB 的描述
  - kb://{kb_id}                   KB 详情
  - doc://{kb_id}/{doc_id}         文档内容

Prompts (查询模板):
  - search-query                   引导 agent 构造有效检索 query
  - kb-discovery                   引导 agent 发现可用的 KB

返回值约定：
  - Tool 函数返回 JSON 字符串（用 ensure_ascii=False）
  - 避免 FastMCP 把 list[dict] 当作 Content 列表自动转换
  - 客户端调用后用 json.loads(text) 解析
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from .context import MCPContext
from .tools import discovery, document
from .tools import search as search_tools


def create_mcp_server(ctx: MCPContext | None = None) -> FastMCP:
    """创建 MCP Server 实例并注册所有 Tool / Resource / Prompt。"""
    if ctx is None:
        ctx = MCPContext.default()

    mcp = FastMCP(
        name="study-rag",
        instructions=(
            "study_rag 是一个企业知识库检索服务。\n"
            "\n"
            "典型工作流：\n"
            "1. 先调用 list_accessible_kbs() 了解可用知识库\n"
            "   或者读取资源 kb://all 一次性获取所有 KB 列表\n"
            "2. 根据 KB 描述判断该查哪个\n"
            "3. 调用 search_kb(kb_id, query) 检索\n"
            "4. 调用 get_document(kb_id, doc_id) 查看完整文档\n"
            "   或者直接读取资源 doc://{kb_id}/{doc_id}\n"
            "\n"
            "kb_id 命名规范: {dept}_{name}（如 rd_frontend、hr_policies）\n"
            "如果不确定查哪个 KB，可直接调 search_all_accessible_kbs 跨 KB 检索。\n"
            "\n"
            "所有 Tool 的 api_key 参数当前为占位实现，可留空字符串。\n"
            "后续接入 JWT/OAuth 后，配置 STUDY_RAG_MCP_REQUIRE_API_KEY=true 即强制校验。"
        ),
    )

    # ---- 注册 Tool ----

    @mcp.tool()
    async def list_accessible_kbs_tool(api_key: str = "") -> str:
        """列出当前用户可访问的所有知识库及其描述（JSON 字符串）。

        任何检索操作的第一步（强烈建议先调用）。
        api_key: 用户凭证（占位实现：可留空字符串；非空时 user_id=api_key）。
        """
        kbs = await discovery.list_accessible_kbs(api_key, ctx)
        return json.dumps([kb.model_dump() for kb in kbs], ensure_ascii=False)

    @mcp.tool()
    async def get_kb_info_tool(api_key: str = "", kb_id: str = "") -> str:
        """获取指定知识库的详细信息（返回 JSON 字符串）。

        在调用 search_kb 之前确认 KB 内容范围。
        api_key: 用户凭证（可留空字符串）。
        kb_id: 知识库 ID。
        """
        detail = await discovery.get_kb_info(api_key, kb_id, ctx)
        return json.dumps(detail.model_dump(), ensure_ascii=False)

    @mcp.tool()
    async def search_kb_tool(
        api_key: str = "",
        kb_id: str = "",
        query: str = "",
        top_k: int = 5,
        use_rerank: bool = True,
        filter_expr: dict | None = None,
    ) -> str:
        """在指定知识库中检索相关内容（返回 JSON 字符串）。

        工作流:
          1. 调用 list_accessible_kbs 获取可访问的 KB
          2. 根据 KB description 判断相关性，选定 kb_id
          3. 调用本 Tool 检索

        kb_id 命名规范: {dept}_{name}，例如 rd_frontend、rd_infra、hr_policies。

        filter_expr: 可选 metadata 过滤，例 {"source": "wiki"}、{"year__gte": 2024}、{"tag__in": ["a", "b"]}。
        支持的操作符后缀: __eq __ne __in __nin __gt __gte __lt __lte __contains __exists。
        """
        hits = await search_tools.search_kb(
            api_key=api_key,
            kb_id=kb_id,
            query=query,
            top_k=top_k,
            use_rerank=use_rerank,
            filter_expr=filter_expr,
            ctx=ctx,
        )
        return json.dumps([hit.model_dump() for hit in hits], ensure_ascii=False)

    @mcp.tool()
    async def search_all_accessible_kbs_tool(
        api_key: str = "",
        query: str = "",
        top_k: int = 5,
        use_rerank: bool = True,
        filter_expr: dict | None = None,
    ) -> str:
        """在所有可访问的知识库中综合检索（返回 JSON 字符串）。

        适用场景: 不确定该查哪个 KB，或需要跨 KB 综合答案。
        返回结果包含 _source_kb 字段标识来源。
        filter_expr 与 search_kb 一致，会下推到每个 KB 的向量检索层。
        """
        hits = await search_tools.search_all_accessible_kbs(
            api_key=api_key,
            query=query,
            top_k=top_k,
            use_rerank=use_rerank,
            filter_expr=filter_expr,
            ctx=ctx,
        )
        return json.dumps([hit.model_dump() for hit in hits], ensure_ascii=False)

    @mcp.tool()
    async def get_document_tool(api_key: str = "", kb_id: str = "", doc_id: str = "") -> str:
        """获取指定文档的完整内容（返回 JSON 字符串）。

        适用场景: search_kb 返回结果后，查看完整文档内容。
        """
        doc = await document.get_document(api_key, kb_id, doc_id, ctx)
        return json.dumps(doc.model_dump(), ensure_ascii=False)

    @mcp.tool()
    async def list_documents_tool(
        api_key: str = "",
        kb_id: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> str:
        """列出指定知识库中的文档摘要（返回 JSON 字符串）。

        适用场景: 列出 KB 已有文档（管理界面、批量任务）。
        返回: doc_id / kb_id / title / source / metadata / has_content 字段。
        limit 范围 (0, 1000]。
        """
        docs = await document.list_documents(
            api_key=api_key, kb_id=kb_id, limit=limit, offset=offset, ctx=ctx
        )
        return json.dumps([d.model_dump() for d in docs], ensure_ascii=False)

    @mcp.tool()
    async def add_document_tool(
        api_key: str = "",
        kb_id: str = "",
        doc_id: str = "",
        title: str = "",
        content: str = "",
        source: str | None = None,
        metadata: dict | None = None,
        overwrite: bool = False,
    ) -> str:
        """向指定知识库添加一个文档（整篇作为一个 chunk，返回 JSON 字符串）。

        需要写权限。doc_id 在同一 KB 内唯一；已存在时除非 overwrite=True 否则报错。
        source 字段会存入 metadata['source']，可在 search_kb 的 filter_expr 里引用。
        """
        result = await document.add_document(
            api_key=api_key,
            kb_id=kb_id,
            doc_id=doc_id,
            title=title,
            content=content,
            source=source,
            metadata=metadata,
            overwrite=overwrite,
            ctx=ctx,
        )
        return json.dumps(result.model_dump(), ensure_ascii=False)

    @mcp.tool()
    async def add_document_chunked_tool(
        api_key: str = "",
        kb_id: str = "",
        doc_id: str = "",
        title: str = "",
        content: str = "",
        source: str | None = None,
        metadata: dict | None = None,
        parser_config: dict | None = None,
        overwrite: bool = False,
    ) -> str:
        """向指定知识库添加一个文档（用 NodeParser 切块后入库，返回 JSON 字符串）。

        parser_config 示例: {"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 50}
        需要写权限。
        """
        result = await document.add_document_chunked(
            api_key=api_key,
            kb_id=kb_id,
            doc_id=doc_id,
            title=title,
            content=content,
            source=source,
            metadata=metadata,
            parser_config=parser_config,
            overwrite=overwrite,
            ctx=ctx,
        )
        return json.dumps(result.model_dump(), ensure_ascii=False)

    @mcp.tool()
    async def add_documents_batch_tool(
        api_key: str = "",
        kb_id: str = "",
        documents: list[dict] | None = None,
        overwrite: bool = False,
    ) -> str:
        """批量添加文档到指定知识库（每篇一个 chunk，返回 JSON 字符串）。

        documents: 每项为 {"doc_id", "title", "content", "source"?, "metadata"?}。
        单条失败不影响其他（错误通过日志记录，返回的 list 不含失败项）。
        上限 500 条/次。
        """
        results = await document.add_documents_batch(
            api_key=api_key,
            kb_id=kb_id,
            documents=documents,
            overwrite=overwrite,
            ctx=ctx,
        )
        return json.dumps([r.model_dump() for r in results], ensure_ascii=False)

    @mcp.tool()
    async def delete_document_tool(api_key: str = "", kb_id: str = "", doc_id: str = "") -> str:
        """删除指定知识库中的文档（向量 + meta 一起删，返回 JSON 字符串）。

        需要写权限。文档不存在时报 DocumentNotFoundError。
        """
        result = await document.delete_document(
            api_key=api_key, kb_id=kb_id, doc_id=doc_id, ctx=ctx
        )
        return json.dumps(result.model_dump(), ensure_ascii=False)

    # ---- 注册 Resource ----
    # URI 形式：让 agent 可以用 read_resource 一次性获取完整 KB 列表或文档

    @mcp.resource("kb://all")
    async def resource_kb_all() -> str:
        """所有可访问知识库的列表（JSON 字符串）。

        适用场景: agent 想一次性拿到所有 KB 描述，而不是循环调用 list_accessible_kbs_tool。
        占位实现下返回所有 KB；真实鉴权接入后可按 query 参数 ?api_key=xxx 鉴权。
        """
        # 占位鉴权：anonymous（ServerSettings.mcp_require_api_key 决定是否强制）
        user = await ctx.auth.resolve("")
        summaries = await ctx.manager.list_summaries()
        accessible = [
            s.model_dump() for s in summaries if s.kb_id in user.accessible_kbs
        ]
        return json.dumps(accessible, ensure_ascii=False)

    @mcp.resource("kb://{kb_id}")
    async def resource_kb_detail(kb_id: str) -> str:
        """单个知识库的详细信息（JSON 字符串）。"""
        user = await ctx.auth.resolve("")
        ctx.auth.check_kb_access(user, kb_id)
        summary = await ctx.manager.get_summary(kb_id)
        if summary is None:
            raise ValueError(f"KB not found: {kb_id}")
        return json.dumps(summary.model_dump(), ensure_ascii=False)

    @mcp.resource("doc://{kb_id}/{doc_id}")
    async def resource_doc(kb_id: str, doc_id: str) -> str:
        """文档完整内容（JSON 字符串）。"""
        user = await ctx.auth.resolve("")
        ctx.auth.check_kb_access(user, kb_id)
        doc = ctx.manager.get_document(kb_id, doc_id)
        if doc is None:
            raise ValueError(f"Document not found: {kb_id}/{doc_id}")
        return json.dumps(doc.model_dump(), ensure_ascii=False)

    # ---- 注册 Prompt ----
    # 查询模板：引导 agent 构造更好的检索 query

    @mcp.prompt()
    def search_query(kb_id: str, question: str) -> str:
        """构造检索 query 模板（带入 KB 上下文 + 用户问题）。

        Args:
            kb_id: 目标知识库 ID
            question: 用户原始问题
        """
        return (
            f"你是一个检索助手。基于以下信息生成最佳检索 query。\n\n"
            f"目标知识库：{kb_id}\n"
            f"用户问题：{question}\n\n"
            f"请：\n"
            f"1. 把模糊口语问题改写为更精确的关键词组合\n"
            f"2. 提取核心实体/概念\n"
            f"3. 直接输出改写后的 query（无需解释）"
        )

    @mcp.prompt()
    def kb_discovery(question: str) -> str:
        """KB 发现提示：引导 agent 根据用户问题选择正确的 KB。

        Args:
            question: 用户原始问题
        """
        return (
            f"用户提问：{question}\n\n"
            f"可用知识库（先用 list_accessible_kbs 获取完整列表）：\n"
            f"- kb_id 命名规范：{{dept}}_{{name}}\n"
            f"- 重点参考每个 KB 的 description 判断相关性\n\n"
            f"任务：\n"
            f"1. 列出最可能相关的 1-3 个 KB\n"
            f"2. 解释为什么相关（基于 description）\n"
            f"3. 给出选定的 kb_id 用于后续 search_kb 调用"
        )

    return mcp


# 模块级默认实例
_mcp_server: FastMCP | None = None


def get_mcp_server() -> FastMCP:
    """获取全局 MCP Server（单例）。"""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = create_mcp_server()
    return _mcp_server
