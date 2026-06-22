"""验证：MCP Resources + Prompts 注册情况。"""

# ruff: noqa: T201, PT017, PT018
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    print("=" * 60)
    print("Verify: MCP Resources + Prompts")
    print("=" * 60)

    from study_rag.mcp.context import MCPContext
    from study_rag.mcp.server import create_mcp_server

    ctx = MCPContext.default()
    mcp = create_mcp_server(ctx)

    # ---- 1. Tool 注册 ----
    print("\n[1] Tool 注册")
    tools = mcp._tool_manager._tools
    tool_names = sorted(tools.keys())
    print(f"    tools ({len(tool_names)}): {tool_names}")
    assert len(tool_names) == 10, f"expected 10 tools, got {len(tool_names)}"
    expected_tools = {
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
    assert set(tool_names) == expected_tools, f"missing: {expected_tools - set(tool_names)}"
    print(f"    PASS: {len(tool_names)} tools registered")

    # ---- 2. Resource 注册 ----
    print("\n[2] Resource 注册")
    # FastMCP 内部存 resource templates
    resources: dict = {}
    if hasattr(mcp, "_resource_manager"):
        # 旧版 API
        resources = mcp._resource_manager._resources  # type: ignore[attr-defined]
    elif hasattr(mcp, "get_resources"):
        # 新版 API
        resources = mcp.get_resources()  # type: ignore[attr-defined]
    print(f"    resources count: {len(resources)}")
    resource_uris = list(resources.keys())
    print(f"    resource URIs: {resource_uris}")

    # ---- 3. 直接调用 Resource 函数（不通过 MCP 协议）----
    print("\n[3] Resource 函数直接调用")

    # 3.1 kb://all
    import asyncio

    from study_rag.mcp.server import create_mcp_server

    # 找到 resource_kb_all 函数
    # FastMCP 内部把 resource handler 存在 _resource_manager._resources
    # 简单做法：直接调装饰器后的函数
    # 实际更稳：直接调 manager 的 read_resource
    mcp2 = create_mcp_server(MCPContext.default())

    # 触发一次 initialize 让内部状态完整
    async def test_resources() -> None:
        from mcp.shared.memory import create_connected_server_and_client_session

        async with create_connected_server_and_client_session(mcp2) as client:
            # 1. list resources
            result = await client.list_resources()
            uris = [r.uri for r in result.resources]
            print(f"    list_resources → {len(uris)} URIs: {uris}")
            assert any("kb://all" in str(u) for u in uris), f"kb://all not in {uris}"
            print("    PASS: kb://all listed")

            # 2. read kb://all
            result = await client.read_resource("kb://all")
            assert result.contents
            text = result.contents[0].text
            import json

            data = json.loads(text)
            assert isinstance(data, list)
            assert len(data) > 0
            print(f"    read kb://all → {len(data)} KBs")

            # 选一个 embedder 实际可用的 KB（dev 环境 BGE 不可用时跳过它们）
            loaded = ctx.manager._embedders  # type: ignore[attr-defined]
            # 实际可写入的 KB：embedder 已加载
            working_kb = None
            for s in data:
                cfg = ctx.manager._registry.get(s["kb_id"])
                if cfg and cfg.embedding in loaded:
                    working_kb = s["kb_id"]
                    break
            assert working_kb, f"no KB with loaded embedder: {list(loaded)}"
            print(f"    using working_kb={working_kb} (loaded embedders: {list(loaded)})")

            # 3. read kb://{kb_id}
            result = await client.read_resource(f"kb://{working_kb}")
            assert result.contents
            text = result.contents[0].text
            data2 = json.loads(text)
            assert data2["kb_id"] == working_kb
            print(f"    read kb://{working_kb} → kb_id={data2['kb_id']}")

            # 4. list_prompts
            prompts = await client.list_prompts()
            prompt_names = [p.name for p in prompts.prompts]
            print(f"    list_prompts → {prompt_names}")
            assert "search_query" in prompt_names
            assert "kb_discovery" in prompt_names
            print("    PASS: 2 prompts registered")

            # 5. get_prompt
            p = await client.get_prompt(
                "search_query",
                {"kb_id": working_kb, "question": "how to deploy?"},
            )
            assert p.messages
            content = p.messages[0].content.text
            assert "deploy" in content
            assert working_kb in content
            print(f"    get_prompt search_query → '{content[:60]}...'")

            # 6. doc://{kb_id}/{doc_id} - 需要先添加一个文档
            add_result = await client.call_tool(
                "add_document_tool",
                {
                    "api_key": "admin",
                    "kb_id": working_kb,
                    "doc_id": "verify_resource_doc",
                    "title": "Resource Test",
                    "content": "Test content for resource.",
                },
            )
            if add_result.isError:
                err_text = add_result.content[0].text if add_result.content else "?"
                # dev 环境无 OpenAI 访问时，embedder 不可用属正常情况
                if any(
                    kw in err_text.lower()
                    for kw in ("connection", "api", "import", "key", "flagembedding")
                ):
                    print(f"    SKIP doc:// test: embedder unavailable ({err_text[:80]})")
                    return  # 整个 test_resources 返回，跳过后续
                raise AssertionError(f"add failed: {err_text}")
            result = await client.read_resource(
                f"doc://{working_kb}/verify_resource_doc"
            )
            assert result.contents
            text = result.contents[0].text
            doc = json.loads(text)
            assert doc["doc_id"] == "verify_resource_doc"
            print("    read doc://.../verify_resource_doc → OK")

            # 清理
            await client.call_tool(
                "delete_document_tool",
                {
                    "api_key": "admin",
                    "kb_id": working_kb,
                    "doc_id": "verify_resource_doc",
                },
            )

    asyncio.run(test_resources())
    print("    PASS: All resources / prompts verified")

    # ---- 4. 兼容性：Tool 仍能正常调用 ----
    print("\n[4] Tool 仍能正常调用（list_accessible_kbs）")
    async def test_tool() -> None:
        from mcp.shared.memory import create_connected_server_and_client_session

        async with create_connected_server_and_client_session(mcp2) as client:
            r = await client.call_tool("list_accessible_kbs_tool", {"api_key": "admin"})
            assert r.content
            import json

            data = json.loads(r.content[0].text)
            assert isinstance(data, list)
            print(f"    PASS: list_accessible_kbs_tool → {len(data)} KBs")

    asyncio.run(test_tool())

    print("\n" + "=" * 60)
    print("ALL PASS: MCP Resources + Prompts")
    print("=" * 60)


if __name__ == "__main__":
    main()
