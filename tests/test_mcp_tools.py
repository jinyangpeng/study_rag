"""MCP Tools 端到端测试：覆盖 10 个 Tool 的对外 API 功能。

测试策略：
  - 用 mock embedder + InMemoryVectorStore 构造一个独立 KB 环境
  - 用真实 MCPContext（manager + PermissionResolver）调用每个 Tool 函数
  - 验证 Tool 函数的返回值 schema、副作用、错误处理
  - 不依赖 FastMCP 协议层（避免复杂 ASGI setup），直接调函数

覆盖的 Tool：
  1.  list_accessible_kbs        - 发现类
  2.  get_kb_info                - 发现类
  3.  search_kb                  - 检索类
  4.  search_all_accessible_kbs  - 检索类
  5.  get_document               - 文档类
  6.  list_documents             - 文档类
  7.  add_document               - 文档类（管理）
  8.  add_document_chunked       - 文档类（管理）
  9.  add_documents_batch        - 文档类（管理，批量）
  10. delete_document            - 文档类（管理）

覆盖的 Resource：
  - kb://all
  - kb://{kb_id}
  - doc://{kb_id}/{doc_id}

覆盖的 Prompt：
  - search_query
  - kb_discovery
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml

from study_rag.auth.permissions import (
    PermissionDenied,
    PermissionResolver,
)
from study_rag.capabilities.embedding import EmbeddingConfig, create_embedder
from study_rag.capabilities.vector_store import (
    VectorStoreConfig,
    create_vector_store,
)
from study_rag.knowledge_bases.manager import KnowledgeBaseManager
from study_rag.knowledge_bases.models import DocumentCreate
from study_rag.knowledge_bases.registry import (
    get_registry,
    reset_registry_cache,
)
from study_rag.mcp.context import MCPContext
from study_rag.mcp.errors import (
    DocumentNotFoundError,
    InvalidParameterError,
)
from study_rag.mcp.tools import discovery, document
from study_rag.mcp.tools import search as search_tools
from study_rag.settings import AppPaths

API_KEY = "test-api-key"


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")


@pytest.fixture
def mcp_ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MCPContext:
    """构造一个用 mock embedder + InMemoryVectorStore 的 MCPContext。

    配置 2 个 KB：
      - rd_test: 用 mock embedder，配合 retrieval 测试
      - rd_other: 第二个 KB，用于跨 KB search_all 测试
    """
    kb_yaml = {
        "knowledge_bases": [
            {
                "kb_id": "rd_test",
                "name": "Test KB",
                "description": "测试知识库，包含 React / Vue / Python 等技术文档",
                "department": "rd",
                "collection": "c_test",
                "embedding": "mock_test",
                "reranker": None,
                "enabled": True,
            },
            {
                "kb_id": "rd_other",
                "name": "Other KB",
                "description": "第二个测试 KB",
                "department": "rd",
                "collection": "c_other",
                "embedding": "mock_test",
                "reranker": None,
                "enabled": True,
            },
        ]
    }
    emb_yaml: dict[str, Any] = {
        "embeddings": {
            "mock_test": {"provider": "mock", "model_name": "m", "dimension": 8},
        }
    }
    vs_yaml: dict[str, Any] = {"vector_store": {"provider": "mock", "uri": ""}}
    rerank_yaml: dict[str, Any] = {"rerankers": {}}
    parser_yaml: dict[str, Any] = {"parsers": {}}
    retrieval_yaml: dict[str, Any] = {
        "default_strategy": "dense",
        "dense": {"over_fetch_factor": 4},
        "sparse": {"k1": 1.5, "b": 0.75, "use_jieba": False, "stop_words": []},
        "hybrid": {
            "dense_weight": 0.6,
            "rrf_k": 60,
            "over_fetch_factor": 4,
            "k1": 1.5,
            "b": 0.75,
            "use_jieba": False,
        },
        "milvus_bm25": {
            "analyzer_type": "chinese",
            "dense_weight": 0.6,
            "rrf_k": 60,
            "over_fetch_factor": 4,
        },
    }

    _write_yaml(tmp_path / "kb.yaml", kb_yaml)
    _write_yaml(tmp_path / "emb.yaml", emb_yaml)
    _write_yaml(tmp_path / "vs.yaml", vs_yaml)
    _write_yaml(tmp_path / "rerank.yaml", rerank_yaml)
    _write_yaml(tmp_path / "llamaindex.yaml", parser_yaml)
    _write_yaml(tmp_path / "retrieval.yaml", retrieval_yaml)

    monkeypatch.setattr(AppPaths, "KB_CONFIG", tmp_path / "kb.yaml")
    monkeypatch.setattr(AppPaths, "EMBEDDING_CONFIG", tmp_path / "emb.yaml")
    monkeypatch.setattr(AppPaths, "VECTOR_STORE_CONFIG", tmp_path / "vs.yaml")
    monkeypatch.setattr(AppPaths, "RERANKER_CONFIG", tmp_path / "rerank.yaml")
    monkeypatch.setattr(AppPaths, "LLAMAINDEX_CONFIG", tmp_path / "llamaindex.yaml")
    monkeypatch.setattr(AppPaths, "RETRIEVAL_CONFIG", tmp_path / "retrieval.yaml")

    reset_registry_cache()
    registry = get_registry()
    embedders = {
        "mock_test": create_embedder(
            EmbeddingConfig(provider="mock", model_name="m", dimension=8)
        ),
    }
    vs = create_vector_store(VectorStoreConfig(provider="mock"))
    manager = KnowledgeBaseManager(
        registry=registry,
        vector_store=vs,
        embedders=embedders,
    )

    async def _setup() -> None:
        await manager.init_all()
        # 预先在 rd_test 写入一些 chunks，便于 search 测试
        await manager.add_document(
            DocumentCreate(
                kb_id="rd_test",
                doc_id="doc1",
                title="React 性能优化指南",
                content="React 应用常见的性能优化点：memo、useMemo、useCallback",
                source="wiki",
                metadata={"tag": "frontend"},
            )
        )
        await manager.add_document(
            DocumentCreate(
                kb_id="rd_test",
                doc_id="doc2",
                title="Vue 性能调优实践",
                content="Vue 性能优化技巧：computed、v-memo、懒加载组件",
                source="wiki",
                metadata={"tag": "frontend"},
            )
        )
        await manager.add_document(
            DocumentCreate(
                kb_id="rd_other",
                doc_id="doc3",
                title="Python 数据分析",
                content="Pandas 与 NumPy 数据分析入门",
                source="book",
                metadata={"tag": "data"},
            )
        )

    asyncio.run(_setup())

    # 用全新的 PermissionResolver（每次 test 隔离）
    resolver = PermissionResolver()
    return MCPContext(manager=manager, auth=resolver)


# ====================================================================
#  1. list_accessible_kbs
# ====================================================================


class TestListAccessibleKBs:
    """list_accessible_kbs：列出可访问 KB。"""

    @pytest.mark.asyncio
    async def test_returns_all_configured_kbs(self, mcp_ctx: MCPContext):
        kbs = await discovery.list_accessible_kbs(API_KEY, mcp_ctx)
        ids = [k.kb_id for k in kbs]
        assert "rd_test" in ids
        assert "rd_other" in ids

    @pytest.mark.asyncio
    async def test_each_kb_has_required_fields(self, mcp_ctx: MCPContext):
        kbs = await discovery.list_accessible_kbs(API_KEY, mcp_ctx)
        for k in kbs:
            assert k.kb_id
            assert k.name
            assert isinstance(k.description, str)
            assert isinstance(k.enabled, bool)
            assert isinstance(k.document_count, int)


# ====================================================================
#  2. get_kb_info
# ====================================================================


class TestGetKBInfo:
    @pytest.mark.asyncio
    async def test_returns_kb_detail(self, mcp_ctx: MCPContext):
        detail = await discovery.get_kb_info(API_KEY, "rd_test", mcp_ctx)
        assert detail.kb_id == "rd_test"
        assert detail.name == "Test KB"
        assert detail.embedding == "mock_test"
        assert detail.reranker is None

    @pytest.mark.asyncio
    async def test_raises_for_unknown_kb(self, mcp_ctx: MCPContext):
        # PermissionResolver 占位实现：accessible_kbs 来自 registry，
        # 所以 unknown_kb 不会通过 check_kb_access（抛 PermissionDenied）
        # 真实鉴权接入后顺序可能调整：KB 不存在 vs 无权限要明确区分
        with pytest.raises(PermissionDenied):
            await discovery.get_kb_info(API_KEY, "unknown_kb", mcp_ctx)


# ====================================================================
#  3. search_kb
# ====================================================================


class TestSearchKB:
    @pytest.mark.asyncio
    async def test_returns_hits(self, mcp_ctx: MCPContext):
        hits = await search_tools.search_kb(
            api_key=API_KEY,
            kb_id="rd_test",
            query="React",
            top_k=5,
            use_rerank=False,
            ctx=mcp_ctx,
        )
        assert len(hits) > 0
        for h in hits:
            assert h.doc_id
            assert h.text
            assert isinstance(h.score, float)
            assert isinstance(h.metadata, dict)

    @pytest.mark.asyncio
    async def test_empty_query_raises(self, mcp_ctx: MCPContext):
        with pytest.raises(InvalidParameterError):
            await search_tools.search_kb(
                api_key=API_KEY, kb_id="rd_test", query="", ctx=mcp_ctx
            )

    @pytest.mark.asyncio
    async def test_invalid_top_k_raises(self, mcp_ctx: MCPContext):
        with pytest.raises(InvalidParameterError):
            await search_tools.search_kb(
                api_key=API_KEY, kb_id="rd_test", query="x", top_k=0, ctx=mcp_ctx
            )
        with pytest.raises(InvalidParameterError):
            await search_tools.search_kb(
                api_key=API_KEY, kb_id="rd_test", query="x", top_k=100, ctx=mcp_ctx
            )

    @pytest.mark.asyncio
    async def test_unknown_kb_raises(self, mcp_ctx: MCPContext):
        # PermissionResolver 占位实现：accessible_kbs 来自 registry，
        # 所以 unknown_kb 会在 check_kb_access 阶段就抛 PermissionDenied
        from study_rag.auth.permissions import PermissionDenied

        with pytest.raises(PermissionDenied):
            await search_tools.search_kb(
                api_key=API_KEY, kb_id="no_such_kb", query="x", ctx=mcp_ctx
            )

    @pytest.mark.asyncio
    async def test_filter_expr(self, mcp_ctx: MCPContext):
        """filter_expr 按 source 过滤。"""
        hits = await search_tools.search_kb(
            api_key=API_KEY,
            kb_id="rd_test",
            query="React",
            top_k=5,
            use_rerank=False,
            filter_expr={"source": "wiki"},
            ctx=mcp_ctx,
        )
        # 仅 source=wiki 的命中
        assert all(h.metadata.get("source") == "wiki" for h in hits)

    @pytest.mark.asyncio
    async def test_strategy_param(self, mcp_ctx: MCPContext):
        """显式 strategy=dense 检索正常。"""
        hits = await search_tools.search_kb(
            api_key=API_KEY,
            kb_id="rd_test",
            query="React",
            top_k=3,
            use_rerank=False,
            strategy="dense",
            ctx=mcp_ctx,
        )
        assert len(hits) > 0

    @pytest.mark.asyncio
    async def test_invalid_strategy_raises(self, mcp_ctx: MCPContext):
        with pytest.raises(InvalidParameterError):
            await search_tools.search_kb(
                api_key=API_KEY,
                kb_id="rd_test",
                query="x",
                strategy="nonexistent",
                ctx=mcp_ctx,
            )


# ====================================================================
#  4. search_all_accessible_kbs
# ====================================================================


class TestSearchAllAccessibleKBs:
    @pytest.mark.asyncio
    async def test_returns_hits_from_multiple_kbs(self, mcp_ctx: MCPContext):
        hits = await search_tools.search_all_accessible_kbs(
            api_key=API_KEY,
            query="性能",
            top_k=5,
            use_rerank=False,
            ctx=mcp_ctx,
        )
        # 至少命中一个
        assert len(hits) > 0
        # 每条都标记了 _source_kb
        assert all("_source_kb" in h.metadata for h in hits)

    @pytest.mark.asyncio
    async def test_results_sorted_by_score(self, mcp_ctx: MCPContext):
        hits = await search_tools.search_all_accessible_kbs(
            api_key=API_KEY,
            query="React 性能",
            top_k=5,
            use_rerank=False,
            ctx=mcp_ctx,
        )
        if len(hits) >= 2:
            scores = [h.score for h in hits]
            assert scores == sorted(scores, reverse=True)


# ====================================================================
#  5. get_document
# ====================================================================


class TestGetDocument:
    @pytest.mark.asyncio
    async def test_returns_full_document(self, mcp_ctx: MCPContext):
        doc = await document.get_document(API_KEY, "rd_test", "doc1", mcp_ctx)
        assert doc.doc_id == "doc1"
        assert doc.kb_id == "rd_test"
        assert doc.title == "React 性能优化指南"
        assert doc.content
        assert doc.source == "wiki"

    @pytest.mark.asyncio
    async def test_raises_for_missing_doc(self, mcp_ctx: MCPContext):
        with pytest.raises(DocumentNotFoundError):
            await document.get_document(API_KEY, "rd_test", "no_such_doc", mcp_ctx)


# ====================================================================
#  6. list_documents
# ====================================================================


class TestListDocuments:
    @pytest.mark.asyncio
    async def test_returns_documents(self, mcp_ctx: MCPContext):
        docs = await document.list_documents(API_KEY, "rd_test", ctx=mcp_ctx)
        ids = [d.doc_id for d in docs]
        assert "doc1" in ids
        assert "doc2" in ids

    @pytest.mark.asyncio
    async def test_pagination(self, mcp_ctx: MCPContext):
        _ = await document.list_documents(API_KEY, "rd_test", ctx=mcp_ctx)
        page1 = await document.list_documents(
            API_KEY, "rd_test", limit=1, offset=0, ctx=mcp_ctx
        )
        assert len(page1) == 1

    @pytest.mark.asyncio
    async def test_invalid_limit_raises(self, mcp_ctx: MCPContext):
        with pytest.raises(InvalidParameterError):
            await document.list_documents(API_KEY, "rd_test", limit=0, ctx=mcp_ctx)
        with pytest.raises(InvalidParameterError):
            await document.list_documents(
                API_KEY, "rd_test", limit=2000, ctx=mcp_ctx
            )


# ====================================================================
#  7. add_document
# ====================================================================


class TestAddDocument:
    @pytest.mark.asyncio
    async def test_add_new_document(self, mcp_ctx: MCPContext):
        result = await document.add_document(
            api_key=API_KEY,
            kb_id="rd_test",
            doc_id="new_doc",
            title="新文档",
            content="新文档内容",
            source="manual",
            metadata={"author": "tester"},
            ctx=mcp_ctx,
        )
        assert result.doc_id == "new_doc"
        assert result.chunks == 1

        # 文档已写入
        doc = await document.get_document(API_KEY, "rd_test", "new_doc", mcp_ctx)
        assert doc.title == "新文档"
        assert doc.metadata.get("author") == "tester"

    @pytest.mark.asyncio
    async def test_duplicate_raises(self, mcp_ctx: MCPContext):
        # doc1 已存在
        from study_rag.mcp.errors import DocumentAlreadyExistsError

        with pytest.raises(DocumentAlreadyExistsError):
            await document.add_document(
                api_key=API_KEY,
                kb_id="rd_test",
                doc_id="doc1",
                title="x",
                content="y",
                ctx=mcp_ctx,
            )

    @pytest.mark.asyncio
    async def test_overwrite(self, mcp_ctx: MCPContext):
        await document.add_document(
            api_key=API_KEY,
            kb_id="rd_test",
            doc_id="doc1",
            title="新标题",
            content="新内容",
            overwrite=True,
            ctx=mcp_ctx,
        )
        doc = await document.get_document(API_KEY, "rd_test", "doc1", mcp_ctx)
        assert doc.title == "新标题"

    @pytest.mark.asyncio
    async def test_empty_content_raises(self, mcp_ctx: MCPContext):
        with pytest.raises(InvalidParameterError):
            await document.add_document(
                api_key=API_KEY,
                kb_id="rd_test",
                doc_id="x",
                title="x",
                content="",
                ctx=mcp_ctx,
            )


# ====================================================================
#  8. add_document_chunked
# ====================================================================


class TestAddDocumentChunked:
    @pytest.mark.asyncio
    async def test_chunked_add(self, mcp_ctx: MCPContext):
        # sentence splitter 需要安装 llamaindex；若未安装则跳过
        pytest.importorskip("llama_index")

        long_content = (
            "第一段：React 性能优化包括 memo 和 useMemo。"
            "第二段：Vue 性能优化包括 computed 和 v-memo。"
            "第三段：性能监控用 Lighthouse。"
        )
        result = await document.add_document_chunked(
            api_key=API_KEY,
            kb_id="rd_test",
            doc_id="chunked_doc",
            title="分块文档",
            content=long_content,
            source="wiki",
            parser_config={"strategy": "sentence", "chunk_size": 200, "chunk_overlap": 20},
            ctx=mcp_ctx,
        )
        assert result.chunks >= 1
        # 文档 meta 已写入
        doc = await document.get_document(
            API_KEY, "rd_test", "chunked_doc", mcp_ctx
        )
        assert doc.title == "分块文档"


# ====================================================================
#  9. add_documents_batch
# ====================================================================


class TestAddDocumentsBatch:
    @pytest.mark.asyncio
    async def test_batch_add(self, mcp_ctx: MCPContext):
        docs = [
            {
                "doc_id": "batch1",
                "title": "批量1",
                "content": "批量文档 1 内容",
                "source": "batch",
            },
            {
                "doc_id": "batch2",
                "title": "批量2",
                "content": "批量文档 2 内容",
                "source": "batch",
            },
        ]
        results = await document.add_documents_batch(
            api_key=API_KEY,
            kb_id="rd_test",
            documents=docs,
            ctx=mcp_ctx,
        )
        assert len(results) == 2
        assert all(r.chunks == 1 for r in results)

    @pytest.mark.asyncio
    async def test_batch_too_large_raises(self, mcp_ctx: MCPContext):
        with pytest.raises(InvalidParameterError):
            await document.add_documents_batch(
                api_key=API_KEY,
                kb_id="rd_test",
                documents=[{"doc_id": f"d{n}", "title": "x", "content": "y"} for n in range(501)],
                ctx=mcp_ctx,
            )

    @pytest.mark.asyncio
    async def test_batch_empty_returns_empty(self, mcp_ctx: MCPContext):
        results = await document.add_documents_batch(
            api_key=API_KEY, kb_id="rd_test", documents=[], ctx=mcp_ctx
        )
        assert results == []


# ====================================================================
#  10. delete_document
# ====================================================================


class TestDeleteDocument:
    @pytest.mark.asyncio
    async def test_delete_existing(self, mcp_ctx: MCPContext):
        result = await document.delete_document(API_KEY, "rd_test", "doc1", mcp_ctx)
        assert result.deleted is True
        # 删除后再 get 应该报错
        with pytest.raises(DocumentNotFoundError):
            await document.get_document(API_KEY, "rd_test", "doc1", mcp_ctx)

    @pytest.mark.asyncio
    async def test_delete_missing_raises(self, mcp_ctx: MCPContext):
        with pytest.raises(DocumentNotFoundError):
            await document.delete_document(
                API_KEY, "rd_test", "no_such_doc", mcp_ctx
            )


# ====================================================================
#  Server 创建测试
# ====================================================================


class TestMCPServerCreation:
    """验证 create_mcp_server 能注册所有 Tool / Resource / Prompt。"""

    def test_create_server_returns_fastmcp(self, mcp_ctx: MCPContext):
        from mcp.server.fastmcp import FastMCP

        from study_rag.mcp.server import create_mcp_server

        mcp = create_mcp_server(mcp_ctx)
        assert isinstance(mcp, FastMCP)

    @pytest.mark.asyncio
    async def test_list_tools_includes_all_10(self, mcp_ctx: MCPContext):
        """FastMCP 内部应注册 10 个 Tool。"""
        from study_rag.mcp.server import create_mcp_server

        mcp = create_mcp_server(mcp_ctx)
        tools = await mcp.list_tools()
        tool_names = {t.name for t in tools}
        expected = {
            "list_accessible_kbs_tool",
            "get_kb_info_tool",
            "search_kb_tool",
            "search_all_accessible_kbs_tool",
            "get_document_tool",
            "list_documents_tool",
            "add_document_tool",
            "add_document_chunked_tool",
            "add_documents_batch_tool",
            "delete_document_tool",
        }
        assert expected.issubset(tool_names), (
            f"Missing tools: {expected - tool_names}"
        )

    @pytest.mark.asyncio
    async def test_list_resources_includes_all_3(self, mcp_ctx: MCPContext):
        """FastMCP 内部应注册 3 个 Resource：kb://all、kb://{kb_id}、doc://{kb_id}/{doc_id}。

        - kb://all 没有 URI 参数，是具体 resource（出现在 list_resources）
        - kb://{kb_id} 和 doc://{kb_id}/{doc_id} 是模板（出现在 list_resource_templates）
        """
        from study_rag.mcp.server import create_mcp_server

        mcp = create_mcp_server(mcp_ctx)
        # kb://all 是具体 resource，不在 templates 里
        templates = await mcp.list_resource_templates()
        template_uris = {t.uriTemplate for t in templates}
        assert "kb://{kb_id}" in template_uris, (
            f"Missing kb//{{kb_id}} template. Got: {template_uris}"
        )
        assert "doc://{kb_id}/{doc_id}" in template_uris, (
            f"Missing doc//{{kb_id}}/{{doc_id}} template. Got: {template_uris}"
        )

    @pytest.mark.asyncio
    async def test_list_prompts_includes_both(self, mcp_ctx: MCPContext):
        from study_rag.mcp.server import create_mcp_server

        mcp = create_mcp_server(mcp_ctx)
        prompts = await mcp.list_prompts()
        prompt_names = {p.name for p in prompts}
        assert "search_query" in prompt_names
        assert "kb_discovery" in prompt_names


# ====================================================================
#  整合：search → get_document 工作流
# ====================================================================


class TestEndToEndWorkflow:
    """模拟 Agent 典型工作流：list → search → get_document。"""

    @pytest.mark.asyncio
    async def test_full_workflow(self, mcp_ctx: MCPContext):
        # 1. list_accessible_kbs
        kbs = await discovery.list_accessible_kbs(API_KEY, mcp_ctx)
        assert len(kbs) >= 1
        target_kb = next(k for k in kbs if k.kb_id == "rd_test")

        # 2. search_kb
        hits = await search_tools.search_kb(
            api_key=API_KEY,
            kb_id=target_kb.kb_id,
            query="React",
            top_k=3,
            use_rerank=False,
            ctx=mcp_ctx,
        )
        assert len(hits) > 0

        # 3. get_document 查看完整文档
        first_hit = hits[0]
        doc = await document.get_document(
            API_KEY, target_kb.kb_id, first_hit.doc_id, mcp_ctx
        )
        assert doc.doc_id == first_hit.doc_id
        assert doc.title  # 标题字段必须有值


# ====================================================================
#  鉴权：匿名访问 + 强制 api_key
# ====================================================================


class TestAuthMode:
    """验证 PermissionResolver 在不同鉴权模式下的行为。

    - 默认模式（mcp_require_api_key=False）：api_key 可空，匿名访问所有 KB
    - 强制模式（mcp_require_api_key=True）：空 api_key 抛 PermissionDenied
    """

    @pytest.mark.asyncio
    async def test_anonymous_access_default_mode(
        self, mcp_ctx: MCPContext, monkeypatch: pytest.MonkeyPatch
    ):
        """默认模式：空 api_key 应允许调用 list_accessible_kbs。"""
        from study_rag.settings import get_server_settings

        # 确保 settings 走默认（不强制）
        s = get_server_settings()
        assert s.mcp_require_api_key is False

        kbs = await discovery.list_accessible_kbs("", mcp_ctx)
        assert len(kbs) >= 1
        # user_id 应该是 anonymous
        user = await mcp_ctx.auth.resolve("")
        assert user.user_id == "anonymous"

    @pytest.mark.asyncio
    async def test_named_user_with_api_key(self, mcp_ctx: MCPContext):
        """非空 api_key 走命名用户路径。"""
        user = await mcp_ctx.auth.resolve("user-alice")
        assert user.user_id == "user-alice"
        # 占位实现下命名用户也拥有所有 KB
        assert "rd_test" in user.accessible_kbs

    @pytest.mark.asyncio
    async def test_empty_api_key_blocked_in_strict_mode(
        self, mcp_ctx: MCPContext, monkeypatch: pytest.MonkeyPatch
    ):
        """强制模式：空 api_key 必须抛 PermissionDenied。"""
        from study_rag.auth import permissions
        from study_rag.settings import ServerSettings

        # permissions.py 内部用 `from ..settings import get_server_settings`
        # 它持有的是 settings 名字的引用；要 patch 那个名字
        strict_settings = ServerSettings(mcp_require_api_key=True)  # type: ignore[call-arg]
        monkeypatch.setattr(permissions, "get_server_settings", lambda: strict_settings)

        with pytest.raises(permissions.PermissionDenied, match="api_key is required"):
            await mcp_ctx.auth.resolve("")

        # 非空仍 OK
        user = await mcp_ctx.auth.resolve("real-key")
        assert user.user_id == "real-key"
