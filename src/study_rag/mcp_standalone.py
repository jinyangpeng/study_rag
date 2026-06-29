"""MCP standalone server：独立端口运行 streamable_http transport。

启动方式：
    python -m study_rag.mcp_standalone
    # 或：
    uvicorn study_rag.mcp_standalone:app --host 0.0.0.0 --port 3220

端点：
    POST/GET http://HOST:PORT/mcp      # JSON-RPC over streamable HTTP
    GET   http://HOST:PORT/.well-known/...  # 资源元数据（auth 启用时）
    GET   http://HOST:PORT/metrics     # Prometheus 指标

为什么独立进程：
  - StreamableHTTP 的 session manager 需要显式管理 task group
  - mount 到 FastAPI 子应用时，session manager 的 run() 不会被自动启动
  - 拆成独立进程后，FastMCP 自己管理生命周期，问题消失
  - 关注点分离：admin REST / MCP 可以独立扩缩容

初始化时机（关键设计）：
  - **不**在模块导入时调 asyncio.run() 做 KB 初始化
    （uvicorn --reload 子进程已在事件循环中，asyncio.run() 会报
    "cannot be called from a running event loop"）
  - 改为「懒加载」：首个 HTTP 请求到达时，用 asyncio.Lock 保护一次性 init
  - ASGI lifespan 事件透传给 FastMCP 的 streamable_http_app（它自管 session manager）

MCP Inspector 调试：
    npx @modelcontextprotocol/inspector http://localhost:3220/mcp
"""

from __future__ import annotations

import asyncio
import os

from mcp.server.fastmcp import FastMCP

from .mcp.context import MCPContext
from .mcp.server import create_mcp_server
from .observability.logging import configure_logging, get_logger
from .observability.metrics import get_metrics
from .settings import get_server_settings

logger = get_logger(__name__)

# ---- 模块级单例：MCP server + context ----
# ctx 在 _build_app() 和 lazy init 中共享同一实例
_ctx: MCPContext | None = None
_mcp: FastMCP | None = None


def _get_ctx() -> MCPContext:
    """获取（必要时创建）模块级 MCPContext 单例。"""
    global _ctx
    if _ctx is None:
        _ctx = MCPContext.default()
    return _ctx


def _build_app() -> FastMCP:
    """构造 MCP server（不在此处初始化 KB，延迟到首个请求）。

    之前版本在模块导入时调 asyncio.run(_bootstrap()) 做 KB 初始化，
    但 uvicorn --reload 子进程已在事件循环中，asyncio.run() 会报
    "cannot be called from a running event loop"。
    """
    settings = get_server_settings()
    configure_logging(level=settings.log_level)

    ctx = _get_ctx()
    return create_mcp_server(ctx)


# 模块级实例（供 `uvicorn study_rag.mcp_standalone:app` 使用）
_mcp = _build_app()


# ---- 懒加载初始化 ----
_init_lock = asyncio.Lock()
_initialized = False


async def _ensure_initialized() -> None:
    """懒加载初始化所有 KB（首个请求时触发，asyncio.Lock 保证只执行一次）。

    之所以不在模块导入时做：
      - asyncio.run() 在已有事件循环时会报 RuntimeError
      - uvicorn --reload 子进程已在事件循环中
    之所以不用 ASGI lifespan：
      - FastMCP 的 streamable_http_app 自身需要 lifespan 启动 session manager
      - 我们不拦截 lifespan，让它透传给 inner app
      - KB 初始化放首个 HTTP 请求前，简单且可靠
    """
    global _initialized
    if _initialized:
        return
    async with _init_lock:
        if _initialized:
            return
        ctx = _get_ctx()
        await ctx.manager.init_all()
        summaries = await ctx.manager.list_summaries()
        logger.info("kbs_initialized", count=len(summaries))
        _initialized = True


# 包装一层 ASGI app：lazy init + /metrics 端点 + 透传 lifespan 给 FastMCP
def _build_asgi_app():
    """返回 ASGI app 组合（mcp 主端点 + /metrics + 懒加载 init）。"""
    mcp_app = _mcp.streamable_http_app()

    async def combined_app(scope, receive, send):  # type: ignore[no-untyped-def]
        # ASGI lifespan：透传给 FastMCP 的 inner app（它需要启动 session manager）
        if scope["type"] == "lifespan":
            return await mcp_app(scope, receive, send)

        # 首个 HTTP/WebSocket 请求前做一次 KB 初始化
        await _ensure_initialized()

        # /metrics 端点
        if scope["type"] == "http" and scope.get("path") == "/metrics":
            body = get_metrics().render().encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"text/plain; version=0.0.4"),
                        (b"content-length", str(len(body)).encode("ascii")),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        # /health 端点（轻量存活探针，供 docker / k8s healthcheck 使用）
        if scope["type"] == "http" and scope.get("path") == "/health":
            body = b'{"status":"ok"}'
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode("ascii")),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        # 其他路径交给 mcp
        return await mcp_app(scope, receive, send)

    return combined_app


def main() -> None:
    """CLI 入口。"""
    import uvicorn

    settings = get_server_settings()
    port = int(os.environ.get("MCP_PORT", "3220"))
    host = os.environ.get("MCP_HOST", settings.host)
    logger.info("starting_mcp_standalone", host=host, port=port, endpoint="/mcp")

    # 用 uvicorn 直接跑 streamable_http_app（FastMCP.run() 不支持 host/port）
    # 加上 /metrics 端点 + 懒加载 init
    uvicorn.run(
        _build_asgi_app(),
        host=host,
        port=port,
        log_level=settings.log_level.lower(),
    )


# 兼容 uvicorn 入口（`uvicorn study_rag.mcp_standalone:app`）
app = _build_asgi_app()


if __name__ == "__main__":
    main()
