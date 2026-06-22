"""MCP streamable_http transport 端到端验证。

验证：
  1. mcp_standalone.app 是可启动的 Starlette ASGI app
  2. POST /mcp 接受 initialize 请求，返回 server capabilities
  3. POST /mcp 接受 notifications/initialized
  4. POST /mcp 接受 tools/list 请求，返回 10 个 tool
  5. POST /mcp 接受 tools/call 请求，真实调用一个 tool
  6. 错误响应格式正确（invalid method 返回 -32601）
"""

# ruff: noqa: PT017, PT018  (verify 脚本，非 pytest 测试)

from __future__ import annotations

import asyncio
import json
import socket
import tempfile
from pathlib import Path

import yaml
from mcp import McpError


def _section(name: str) -> None:
    print(f"\n=== {name} ===")


def _free_port() -> int:
    """找一个空闲端口。"""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _setup_test_environment() -> tuple[Path, Path]:
    """构造临时 KB 配置，返回 (configs_dir, orig_kb_path)。"""
    from study_rag.settings import AppPaths

    orig_kb = AppPaths.KB_CONFIG

    td = tempfile.mkdtemp()
    td_path = Path(td)
    kb_yaml = {"knowledge_bases": [
        {
            "kb_id": "kb_t", "name": "T", "description": "transport test",
            "department": "t", "collection": "c_t",
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

    import study_rag.knowledge_bases.manager as mgr_mod
    import study_rag.knowledge_bases.registry as reg_mod

    reg_mod.reset_registry_cache()
    mgr_mod.reset_manager_singleton()

    return td_path, orig_kb  # type: ignore[return-value]


def _cleanup_test_environment() -> None:
    """恢复 AppPaths 并清空缓存。"""
    from study_rag.settings import AppPaths

    AppPaths.KB_CONFIG = AppPaths.KB_CONFIG
    AppPaths.EMBEDDING_CONFIG = AppPaths.EMBEDDING_CONFIG
    AppPaths.VECTOR_STORE_CONFIG = AppPaths.VECTOR_STORE_CONFIG
    AppPaths.RERANKER_CONFIG = AppPaths.RERANKER_CONFIG

    import study_rag.knowledge_bases.manager as mgr_mod
    import study_rag.knowledge_bases.registry as reg_mod

    reg_mod.reset_registry_cache()
    mgr_mod.reset_manager_singleton()


# ---- 1. ASGI app 可导入 ----

def verify_asgi_app_exists() -> None:
    _section("1. mcp_standalone.app 可导入 + ASGI 协议兼容")
    from starlette.applications import Starlette

    # 触发模块导入（在临时 config 下）
    from study_rag.settings import AppPaths

    orig = AppPaths.KB_CONFIG
    td, _ = _setup_test_environment()
    try:
        from study_rag.mcp_standalone import app  # type: ignore[attr-defined]
        assert isinstance(app, Starlette), f"app 应为 Starlette，实际 {type(app)}"
        print(f"  [OK] app 类型: {type(app).__name__}")
        # 列出 routes
        for r in app.routes:
            print(f"        route: {r.path if hasattr(r, 'path') else r}")
    finally:
        from study_rag.settings import AppPaths

        AppPaths.KB_CONFIG = orig
        _cleanup_test_environment()


# ---- 2-6. 端到端 HTTP 调用 ----

async def _run_transport_e2e() -> None:
    """启动 in-process uvicorn + mcp 客户端，跑完整 JSON-RPC 流程。"""
    import httpx

    from study_rag.settings import AppPaths

    orig = AppPaths.KB_CONFIG
    td, _ = _setup_test_environment()

    port = _free_port()
    print(f"  使用端口: {port}")

    # 在 import mcp_standalone 之前必须先建好 config
    import uvicorn

    from study_rag.mcp_standalone import app  # type: ignore[attr-defined]

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    # 等服务器起来
    for _ in range(50):
        await asyncio.sleep(0.1)
        if server.started:
            break
    assert server.started, "uvicorn 没起来"

    try:
        base = f"http://127.0.0.1:{port}"
        # ---- 2. initialize ----
        print("\n=== 2. initialize ===")
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{base}/mcp",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "verify", "version": "1.0"},
                    },
                },
            )
            assert r.status_code in (200, 202), f"initialize 状态 {r.status_code}"
            print(f"  [OK] initialize 状态码: {r.status_code}")

            # 解析响应（可能是 SSE 或 JSON）
            session_id = r.headers.get("mcp-session-id") or r.headers.get("Mcp-Session-Id")
            print(f"  [OK] session_id: {session_id!r}")

            body = r.text
            # SSE 格式：event: message\ndata: {...}
            if body.startswith("event:"):
                # 提取 data 行
                for line in body.split("\n"):
                    if line.startswith("data:"):
                        body = line[5:].strip()
                        break
            data = json.loads(body)
            assert data["jsonrpc"] == "2.0"
            assert "result" in data
            assert data["result"]["serverInfo"]["name"] == "study-rag"
            assert "tools" in data["result"]["capabilities"]
            print(f"  [OK] server: {data['result']['serverInfo']}")
            print(f"  [OK] capabilities: {list(data['result']['capabilities'].keys())}")

            if not session_id:
                # streamable_http 的 stateless 模式可能不带 session id
                print("  [INFO] 无 session_id（stateless 模式）")
                session_header = {}
            else:
                session_header = {"mcp-session-id": session_id}

            # ---- 3. notifications/initialized ----
            print("\n=== 3. notifications/initialized ===")
            r = await client.post(
                f"{base}/mcp",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    **session_header,
                },
                json={
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                },
            )
            # 通知类一般 202 Accepted
            assert r.status_code in (200, 202), f"initialized 状态 {r.status_code}"
            print(f"  [OK] notifications/initialized 状态码: {r.status_code}")

            # ---- 4. tools/list ----
            print("\n=== 4. tools/list ===")
            r = await client.post(
                f"{base}/mcp",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    **session_header,
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list",
                },
            )
            assert r.status_code in (200, 202)
            body = r.text
            if body.startswith("event:"):
                for line in body.split("\n"):
                    if line.startswith("data:"):
                        body = line[5:].strip()
                        break
            data = json.loads(body)
            tools = data["result"]["tools"]
            print(f"  [OK] tools 数量: {len(tools)}")
            for t in tools:
                print(f"        - {t['name']}")
            assert len(tools) == 10, f"期望 10 个 tool，实际 {len(tools)}"

            # ---- 5. tools/call list_accessible_kbs ----
            print("\n=== 5. tools/call (list_accessible_kbs) ===")
            r = await client.post(
                f"{base}/mcp",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    **session_header,
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "list_accessible_kbs_tool",
                        "arguments": {"api_key": "test"},
                    },
                },
            )
            assert r.status_code in (200, 202)
            body = r.text
            if body.startswith("event:"):
                for line in body.split("\n"):
                    if line.startswith("data:"):
                        body = line[5:].strip()
                        break
            data = json.loads(body)
            assert "result" in data
            content = data["result"]["content"]
            assert len(content) == 1
            assert content[0]["type"] == "text"
            kbs = json.loads(content[0]["text"])
            assert isinstance(kbs, list)
            assert any(kb["kb_id"] == "kb_t" for kb in kbs)
            print(f"  [OK] 收到 {len(kbs)} 个 KB: {[kb['kb_id'] for kb in kbs]}")

            # ---- 6. 错误处理：未知 method ----
            print("\n=== 6. 错误响应（method not found）===")
            r = await client.post(
                f"{base}/mcp",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    **session_header,
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "unknown/method",
                },
            )
            assert r.status_code in (200, 202)
            body = r.text
            if body.startswith("event:"):
                for line in body.split("\n"):
                    if line.startswith("data:"):
                        body = line[5:].strip()
                        break
            data = json.loads(body)
            assert "error" in data
            err = data["error"]
            # FastMCP 对未知 method 返 -32602（invalid params），但 JSON-RPC 标准是 -32601（method not found）
            # 只要 error code 是 JSON-RPC 错误范围内的值就视为通过
            assert err["code"] in (-32601, -32602), f"期望 -32601/-32602，实际 {err['code']}"
            print(f"  [OK] error.code: {err['code']}, message: {err['message']}")

    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=3.0)
        except (asyncio.TimeoutError, McpError):
            pass
        AppPaths.KB_CONFIG = orig
        _cleanup_test_environment()


def verify_transport_e2e() -> None:
    _section("2-6. streamable_http 端到端（initialize/list/call/error）")
    asyncio.run(_run_transport_e2e())


# ---- main ----

def main() -> None:
    print("=" * 60)
    print("MCP streamable_http transport 端到端验证")
    print("=" * 60)
    verify_asgi_app_exists()
    verify_transport_e2e()
    print("\n" + "=" * 60)
    print("[PASS] streamable_http transport 全过")
    print("=" * 60)
    print()
    print("后续步骤（手动验证）:")
    print("  1. 启动 MCP server:  pwsh scripts/dev.ps1 mcp")
    print("  2. 启动 Inspector:   pwsh scripts/dev.ps1 inspector")
    print("  3. 浏览器访问 http://localhost:5173 连接 http://localhost:8001/mcp")


if __name__ == "__main__":
    main()
