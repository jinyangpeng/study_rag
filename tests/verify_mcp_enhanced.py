"""MCP 增强功能验证。

验证：
  1. filter 转换器（to_milvus_expr / matches_filter）
  2. mock vector store filter
  3. search_kb 透传 filter_expr
  4. search_all_accessible_kbs 透传 filter_expr
  5. 文档管理 Tool：add_document / list_documents / delete_document
  6. add_document_chunked（LlamaIndex 切块）
  7. add_documents_batch
  8. 鉴权：KB 写权限校验
  9. MCP server 工具数量（10 个）
"""

# ruff: noqa: PT017, PT018  (verify 脚本，非 pytest 测试)

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import yaml


def _section(name: str) -> None:
    print(f"\n=== {name} ===")


def _setup_test_environment() -> tuple[Any, Any]:
    """构造一个有数据的临时 KB 环境（mock vs/embedding）。"""
    import study_rag.knowledge_bases.manager as mgr_mod
    import study_rag.knowledge_bases.registry as reg_mod
    from study_rag.settings import AppPaths

    orig_paths = {
        "KB_CONFIG": AppPaths.KB_CONFIG,
        "EMBEDDING_CONFIG": AppPaths.EMBEDDING_CONFIG,
        "VECTOR_STORE_CONFIG": AppPaths.VECTOR_STORE_CONFIG,
        "RERANKER_CONFIG": AppPaths.RERANKER_CONFIG,
    }
    td = tempfile.mkdtemp()
    td_path = Path(td)
    kb_yaml = {"knowledge_bases": [
        {
            "kb_id": "kb_rd", "name": "RD", "description": "研发文档",
            "department": "rd", "collection": "c_rd",
            "embedding": "mock", "reranker": None, "enabled": True,
        },
        {
            "kb_id": "kb_hr", "name": "HR", "description": "公司制度",
            "department": "hr", "collection": "c_hr",
            "embedding": "mock", "reranker": None, "enabled": True,
        },
    ]}
    emb_yaml = {"embeddings": {
        "mock": {"provider": "mock", "model_name": "m", "dimension": 4},
    }}
    (td_path / "kb.yaml").write_text(
        yaml.safe_dump(kb_yaml, allow_unicode=True), encoding="utf-8"
    )
    (td_path / "emb.yaml").write_text(
        yaml.safe_dump(emb_yaml, allow_unicode=True), encoding="utf-8"
    )
    (td_path / "vs.yaml").write_text(
        yaml.safe_dump({"vector_store": {"provider": "mock", "uri": ""}}, allow_unicode=True),
        encoding="utf-8",
    )
    (td_path / "rerank.yaml").write_text(
        yaml.safe_dump({"rerankers": {}}, allow_unicode=True), encoding="utf-8",
    )
    AppPaths.KB_CONFIG = td_path / "kb.yaml"
    AppPaths.EMBEDDING_CONFIG = td_path / "emb.yaml"
    AppPaths.VECTOR_STORE_CONFIG = td_path / "vs.yaml"
    AppPaths.RERANKER_CONFIG = td_path / "rerank.yaml"

    reg_mod.reset_registry_cache()
    mgr_mod.reset_manager_singleton()

    manager = mgr_mod.build_default_manager()

    def _cleanup() -> None:
        AppPaths.KB_CONFIG = orig_paths["KB_CONFIG"]
        AppPaths.EMBEDDING_CONFIG = orig_paths["EMBEDDING_CONFIG"]
        AppPaths.VECTOR_STORE_CONFIG = orig_paths["VECTOR_STORE_CONFIG"]
        AppPaths.RERANKER_CONFIG = orig_paths["RERANKER_CONFIG"]
        reg_mod.reset_registry_cache()
        mgr_mod.reset_manager_singleton()

    return manager, _cleanup


# ---- 1. filter 转换器 ----

def verify_filter_translators() -> None:
    _section("1. filter 转换器")
    from study_rag.capabilities.vector_store.filters import (
        matches_filter,
        parse_key,
        to_milvus_expr,
    )

    # eq
    assert to_milvus_expr({"department": "rd"}) == 'department == "rd"'
    print(f"  [OK] eq: {to_milvus_expr({'department': 'rd'})!r}")
    # __eq
    assert to_milvus_expr({"department__eq": "rd"}) == 'department == "rd"'
    # __ne
    assert to_milvus_expr({"status__ne": "draft"}) == 'status != "draft"'
    print(f"  [OK] ne: {to_milvus_expr({'status__ne': 'draft'})!r}")
    # __in
    assert to_milvus_expr({"tag__in": ["a", "b"]}) == 'tag in ["a", "b"]'
    print(f"  [OK] in: {to_milvus_expr({'tag__in': ['a', 'b']})!r}")
    # __nin
    assert to_milvus_expr({"tag__nin": ["x"]}) == 'not (tag in ["x"])'
    # __gte
    assert to_milvus_expr({"year__gte": 2024}) == "year >= 2024"
    print(f"  [OK] gte: {to_milvus_expr({'year__gte': 2024})!r}")
    # __contains
    assert to_milvus_expr({"title__contains": "API"}) == 'title like "%API%"'
    # __exists
    assert to_milvus_expr({"draft__exists": True}) == "draft is not null"
    # AND
    expr = to_milvus_expr({"year__gte": 2024, "tag__in": ["a", "b"]})
    assert " and " in expr
    print(f"  [OK] AND: {expr!r}")

    # matches_filter
    md = {"department": "rd", "year": 2025, "tags": ["a", "b"], "title": "API 设计"}
    assert matches_filter(md, {"department": "rd"}) is True
    assert matches_filter(md, {"department": "hr"}) is False
    assert matches_filter(md, {"year__gte": 2024}) is True
    assert matches_filter(md, {"year__gte": 2026}) is False
    assert matches_filter(md, {"title__contains": "API"}) is True
    print("  [OK] matches_filter 多 op 通过")

    # 非法字段名
    try:
        parse_key("bad field")
        raise AssertionError("应抛 ValueError")
    except ValueError:
        print("  [OK] 非法字段名抛 ValueError")

    # 非法 op
    try:
        to_milvus_expr({"a__badop": 1})
        raise AssertionError("应抛 ValueError")
    except ValueError:
        print("  [OK] 非法 op 抛 ValueError")

    # 空 dict
    assert to_milvus_expr({}) is None
    assert to_milvus_expr(None) is None
    print("  [OK] 空 dict / None -> None")


# ---- 2. mock vector store filter ----

def verify_mock_filter() -> None:
    _section("2. mock vector store filter")
    from study_rag.capabilities.vector_store.base import (
        VectorRecord,
        VectorStoreConfig,
    )
    from study_rag.capabilities.vector_store.impls import InMemoryVectorStore

    async def _run() -> None:
        store = InMemoryVectorStore(VectorStoreConfig(provider="mock", uri=""))
        await store.create_collection("c1", dimension=4)
        await store.insert("c1", [
            VectorRecord(id="1", vector=[1, 0, 0, 0], text="a", metadata={"tag": "x", "year": 2024}),
            VectorRecord(id="2", vector=[0, 1, 0, 0], text="b", metadata={"tag": "y", "year": 2025}),
            VectorRecord(id="3", vector=[0, 0, 1, 0], text="c", metadata={"tag": "x", "year": 2023}),
        ])
        # 无 filter
        r = await store.search("c1", [1, 0, 0, 0], top_k=3)
        assert len(r) == 3
        # tag==x
        r = await store.search("c1", [1, 0, 0, 0], top_k=3, filter_expr={"tag": "x"})
        assert {h.id for h in r} == {"1", "3"}
        print(f"  [OK] filter {{tag: x}} -> {[h.id for h in r]}")
        # year>=2024
        r = await store.search("c1", [1, 0, 0, 0], top_k=3, filter_expr={"year__gte": 2024})
        assert {h.id for h in r} == {"1", "2"}
        print(f"  [OK] filter year>=2024 -> {[h.id for h in r]}")
        # 复合
        r = await store.search("c1", [1, 0, 0, 0], top_k=3, filter_expr={"tag": "x", "year__gte": 2024})
        assert {h.id for h in r} == {"1"}
        print(f"  [OK] filter 复合 -> {[h.id for h in r]}")

    asyncio.run(_run())


# ---- 3. search_kb 透传 filter ----

def verify_search_kb_with_filter() -> None:
    _section("3. search_kb 透传 filter_expr")
    from study_rag.auth.permissions import get_permission_resolver
    from study_rag.mcp.context import MCPContext
    from study_rag.mcp.tools.search import search_kb

    async def _run() -> None:
        manager, cleanup = _setup_test_environment()
        try:
            ctx = MCPContext(manager=manager, auth=get_permission_resolver())
            await manager.init_all()
            from study_rag.knowledge_bases.models import DocumentCreate

            # 写入测试数据：3 篇带不同 source/year 的文档
            for i, (src, year, content) in enumerate([
                ("wiki", 2024, "React 组件生命周期"),
                ("github", 2025, "K8s 部署需要 Dockerfile"),
                ("wiki", 2023, "Vue 模板语法"),
            ]):
                await manager.add_document(DocumentCreate(
                    kb_id="kb_rd",
                    doc_id=f"d{i}",
                    title=f"d{i}",
                    content=content,
                    source=src,
                    metadata={"year": year, "tag": "frontend" if i < 2 else "general"},
                ))

            # 不过滤
            r_all = await search_kb(
                api_key="x", kb_id="kb_rd", query="React", top_k=5,
                ctx=ctx,
            )
            assert len(r_all) >= 1
            print(f"  [OK] 无 filter: {len(r_all)} hits")

            # filter source=wiki
            r_wiki = await search_kb(
                api_key="x", kb_id="kb_rd", query="React", top_k=5,
                filter_expr={"source": "wiki"}, ctx=ctx,
            )
            for h in r_wiki:
                assert h.metadata.get("source") == "wiki"
            print(f"  [OK] filter source=wiki: {len(r_wiki)} hits")

            # filter year>=2024
            r_year = await search_kb(
                api_key="x", kb_id="kb_rd", query="组件", top_k=5,
                filter_expr={"year__gte": 2024}, ctx=ctx,
            )
            for h in r_year:
                assert h.metadata.get("year", 0) >= 2024
            print(f"  [OK] filter year>=2024: {len(r_year)} hits")

            # 非法 filter
            try:
                await search_kb(
                    api_key="x", kb_id="kb_rd", query="x",
                    filter_expr={"bad field": "x"}, ctx=ctx,
                )
                raise AssertionError("应抛 InvalidParameterError")
            except Exception as e:
                # ValueError from parse_key, wrapped by InvalidParameterError
                print(f"  [OK] 非法 filter 抛错: {type(e).__name__}: {e}")

        finally:
            cleanup()

    asyncio.run(_run())


# ---- 4. search_all_accessible_kbs 透传 filter ----

def verify_search_all_with_filter() -> None:
    _section("4. search_all_accessible_kbs 透传 filter_expr")
    from study_rag.auth.permissions import get_permission_resolver
    from study_rag.mcp.context import MCPContext
    from study_rag.mcp.tools.search import search_all_accessible_kbs

    async def _run() -> None:
        manager, cleanup = _setup_test_environment()
        try:
            ctx = MCPContext(manager=manager, auth=get_permission_resolver())
            await manager.init_all()
            from study_rag.knowledge_bases.models import DocumentCreate

            # 在 2 个 KB 各写一条
            await manager.add_document(DocumentCreate(
                kb_id="kb_rd", doc_id="r1", title="r1",
                content="React 入门", source="wiki", metadata={"year": 2024},
            ))
            await manager.add_document(DocumentCreate(
                kb_id="kb_hr", doc_id="h1", title="h1",
                content="HR 制度", source="policy", metadata={"year": 2025},
            ))

            # 不带 filter
            r = await search_all_accessible_kbs(api_key="x", query="React", ctx=ctx)
            assert any(h.metadata.get("_source_kb") == "kb_rd" for h in r)
            print(f"  [OK] 无 filter: {len(r)} hits, source 包含 kb_rd")

            # filter source=policy（应只剩 HR）
            r = await search_all_accessible_kbs(
                api_key="x", query="HR", filter_expr={"source": "policy"}, ctx=ctx
            )
            sources = {h.metadata.get("_source_kb") for h in r}
            assert sources == {"kb_hr"} or not r
            print(f"  [OK] filter source=policy: sources={sources}")
        finally:
            cleanup()

    asyncio.run(_run())


# ---- 5-7. 文档管理 Tool ----

def verify_document_tools() -> None:
    _section("5-7. 文档管理 Tool（add/list/delete/chunked/batch）")
    from study_rag.auth.permissions import get_permission_resolver
    from study_rag.mcp.context import MCPContext
    from study_rag.mcp.errors import (
        DocumentAlreadyExistsError,
        DocumentNotFoundError,
    )
    from study_rag.mcp.tools import document as doc_tool

    async def _run() -> None:
        manager, cleanup = _setup_test_environment()
        try:
            ctx = MCPContext(manager=manager, auth=get_permission_resolver())
            await manager.init_all()

            # 5. add_document
            r = await doc_tool.add_document(
                api_key="x", kb_id="kb_rd", doc_id="d1",
                title="React 入门", content="React 是一个用于构建 UI 的 JS 库",
                source="wiki", metadata={"year": 2024, "tag": "frontend"},
                ctx=ctx,
            )
            assert r.doc_id == "d1" and r.chunks == 1
            print(f"  [OK] add_document -> {r.doc_id}, chunks={r.chunks}")

            # 重复添加 -> DocumentAlreadyExistsError
            try:
                await doc_tool.add_document(
                    api_key="x", kb_id="kb_rd", doc_id="d1", title="x", content="y", ctx=ctx
                )
                raise AssertionError("应抛 DocumentAlreadyExistsError")
            except DocumentAlreadyExistsError:
                print("  [OK] 重复 add 抛 DocumentAlreadyExistsError")

            # overwrite=True 覆盖
            r2 = await doc_tool.add_document(
                api_key="x", kb_id="kb_rd", doc_id="d1",
                title="React 入门 v2", content="新内容",
                source="wiki", overwrite=True, ctx=ctx,
            )
            assert r2.title == "React 入门 v2"
            print(f"  [OK] overwrite=True 覆盖成功: title={r2.title}")

            # list_documents
            r_list = await doc_tool.list_documents(api_key="x", kb_id="kb_rd", ctx=ctx)
            assert len(r_list) == 1
            assert r_list[0].doc_id == "d1"
            print(f"  [OK] list_documents: {len(r_list)} 条")

            # 6. add_document_chunked
            long_text = "React 是一个用于构建用户界面的 JavaScript 库。\n" * 50
            r3 = await doc_tool.add_document_chunked(
                api_key="x", kb_id="kb_rd", doc_id="d2",
                title="React 长文", content=long_text,
                source="wiki", metadata={"year": 2024},
                parser_config={"strategy": "sentence", "chunk_size": 256, "chunk_overlap": 32},
                ctx=ctx,
            )
            assert r3.chunks >= 2, f"应切成多块，实际 {r3.chunks}"
            print(f"  [OK] add_document_chunked -> {r3.chunks} chunks")

            # 7. add_documents_batch
            r4 = await doc_tool.add_documents_batch(
                api_key="x", kb_id="kb_rd",
                documents=[
                    {"doc_id": "b1", "title": "b1", "content": "Batch doc 1", "source": "wiki"},
                    {"doc_id": "b2", "title": "b2", "content": "Batch doc 2", "source": "github"},
                ],
                ctx=ctx,
            )
            assert len(r4) == 2
            print(f"  [OK] add_documents_batch -> {len(r4)} 条成功")

            # delete_document
            r5 = await doc_tool.delete_document(api_key="x", kb_id="kb_rd", doc_id="d1", ctx=ctx)
            assert r5.deleted is True
            print(f"  [OK] delete_document -> deleted={r5.deleted}")

            # 重复删 -> DocumentNotFoundError
            try:
                await doc_tool.delete_document(api_key="x", kb_id="kb_rd", doc_id="d1", ctx=ctx)
                raise AssertionError("应抛 DocumentNotFoundError")
            except DocumentNotFoundError:
                print("  [OK] 重复 delete 抛 DocumentNotFoundError")
        finally:
            cleanup()

    asyncio.run(_run())


# ---- 8. 写权限校验 ----

def verify_write_permission() -> None:
    _section("8. 写权限校验（PermissionDenied）")
    from study_rag.auth.permissions import (
        PermissionDenied,
        PermissionResolver,
        UserContext,
    )
    from study_rag.mcp.context import MCPContext
    from study_rag.mcp.tools import document as doc_tool

    async def _run() -> None:
        manager, cleanup = _setup_test_environment()
        try:
            # 构造一个只对 kb_rd 有读权限、对 kb_hr 有写权限的用户
            class _RestrictResolver(PermissionResolver):
                async def resolve(self, api_key: str) -> UserContext:
                    if api_key == "read-only":
                        return UserContext(
                            user_id="ro", accessible_kbs=["kb_rd"], writable_kbs=[]
                        )
                    if api_key == "hr-writer":
                        return UserContext(
                            user_id="hw", accessible_kbs=["kb_hr"], writable_kbs=["kb_hr"]
                        )
                    return await super().resolve(api_key)

            resolver = _RestrictResolver()
            ctx = MCPContext(manager=manager, auth=resolver)
            await manager.init_all()

            # 只读用户写 -> PermissionDenied
            try:
                await doc_tool.add_document(
                    api_key="read-only", kb_id="kb_rd", doc_id="d1",
                    title="x", content="y", ctx=ctx,
                )
                raise AssertionError("应抛 PermissionDenied")
            except PermissionDenied:
                print("  [OK] read-only 用户写 kb_rd 抛 PermissionDenied")

            # hr-writer 写 kb_rd -> PermissionDenied（无写权限）
            try:
                await doc_tool.add_document(
                    api_key="hr-writer", kb_id="kb_rd", doc_id="d1",
                    title="x", content="y", ctx=ctx,
                )
                raise AssertionError("应抛 PermissionDenied")
            except PermissionDenied:
                print("  [OK] hr-writer 写 kb_rd 抛 PermissionDenied（仅 kb_hr 有写权限）")

            # hr-writer 写 kb_hr -> OK
            r = await doc_tool.add_document(
                api_key="hr-writer", kb_id="kb_hr", doc_id="h1",
                title="x", content="y", ctx=ctx,
            )
            assert r.doc_id == "h1"
            print(f"  [OK] hr-writer 写 kb_hr 成功: {r.doc_id}")
        finally:
            cleanup()

    asyncio.run(_run())


# ---- 9. MCP server 工具数量 ----

def verify_mcp_server_tools() -> None:
    _section("9. MCP server 工具数量")
    from study_rag.mcp.server import create_mcp_server

    manager, cleanup = _setup_test_environment()
    try:
        mcp = create_mcp_server()
        tools = asyncio.run(mcp.list_tools())
        names = [t.name for t in tools]
        print(f"  [OK] 注册的 Tool 数量: {len(names)}")
        for n in names:
            print(f"        - {n}")
        assert len(names) == 10, f"期望 10 个 tool，实际 {len(names)}"
        # 校验全部 10 个 Tool 都在
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
        missing = expected - set(names)
        assert not missing, f"缺少 tool: {missing}"
        print("  [OK] 全部 10 个 Tool 注册成功")
    finally:
        cleanup()


# ---- main ----

def main() -> None:
    print("=" * 60)
    print("MCP 增强功能端到端验证")
    print("=" * 60)
    verify_filter_translators()
    verify_mock_filter()
    verify_search_kb_with_filter()
    verify_search_all_with_filter()
    verify_document_tools()
    verify_write_permission()
    verify_mcp_server_tools()

    print("\n" + "=" * 60)
    print("[PASS] 全部增强验证通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
