"""MCP standalone server：独立端口运行 streamable_http transport。

启动方式：
    python -m study_rag.mcp_standalone
    # 或：
    uvicorn study_rag.mcp_standalone:app --host 0.0.0.0 --port 8001

端点：
    POST/GET http://HOST:PORT/mcp      # JSON-RPC over streamable HTTP
    GET   http://HOST:PORT/.well-known/...  # 资源元数据（auth 启用时）
    GET   http://HOST:PORT/metrics     # Prometheus 指标

为什么独立进程：
  - StreamableHTTP 的 session manager 需要显式管理 task group
  - mount 到 FastAPI 子应用时，session manager 的 run() 不会被自动启动
  - 拆成独立进程后，FastMCP 自己管理生命周期，问题消失
  - 关注点分离：admin REST / MCP 可以独立扩缩容

MCP Inspector 调试：
    npx @modelcontextprotocol/inspector http://localhost:8001/mcp
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from .mcp.context import MCPContext
from .mcp.server import create_mcp_server
from .observability.logging import configure_logging, get_logger
from .observability.metrics import get_metrics
from .settings import get_server_settings

logger = get_logger(__name__)


def _build_app() -> FastMCP:
    """构造 MCP server 并初始化 KB。"""
    # 初始化结构化日志
    settings = get_server_settings()
    configure_logging(level=settings.log_level)

    ctx = MCPContext.default()

    # 同步初始化所有 KB（嵌入向量库 / 创建 collection）
    import asyncio

    asyncio.run(ctx.manager.init_all())
    logger.info("kbs_initialized", count=len(ctx.manager.list_summaries()))

    return create_mcp_server(ctx)


# 模块级实例（供 `uvicorn study_rag.mcp_standalone:app` 使用）
mcp: FastMCP = _build_app()


# 包装一层 ASGI app：把 /metrics 接到同一个端口
def _build_asgi_app():
    """返回 ASGI app 组合（mcp 主端点 + /metrics）。"""
    mcp_app = mcp.streamable_http_app()

    async def combined_app(scope, receive, send):  # type: ignore[no-untyped-def]
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
        # 其他路径交给 mcp
        return await mcp_app(scope, receive, send)

    return combined_app


def main() -> None:
    """CLI 入口。"""
    import uvicorn

    settings = get_server_settings()
    port = int(os.environ.get("MCP_PORT", "8001"))
    host = os.environ.get("MCP_HOST", settings.host)
    logger.info("starting_mcp_standalone", host=host, port=port, endpoint="/mcp")

    # 用 uvicorn 直接跑 streamable_http_app（FastMCP.run() 不支持 host/port）
    # 加上 /metrics 端点
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
