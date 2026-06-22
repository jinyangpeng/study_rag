"""FastAPI 应用入口：管理面 REST API。

架构:
  - /admin/*     → 管理面 REST
  - /health/*    → 健康检查
  - /metrics     → Prometheus 指标

MCP Server 独立部署：
  - 不要把 MCP mount 到本应用，详见 mcp_standalone.py
  - StreamableHTTP 的 session_manager 需要显式管理 task group，
    与 FastAPI mount 不兼容
  - 生产推荐两个独立进程：admin REST + MCP streamable_http
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from .mcp.context import MCPContext
from .observability.logging import configure_logging, get_logger
from .observability.metrics import get_metrics
from .observability.middleware import RequestIDMiddleware
from .settings import get_server_settings

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """创建 FastAPI 应用（仅 admin REST）。"""
    settings = get_server_settings()

    # 1. 初始化结构化日志（在 app 创建前）
    configure_logging(level=settings.log_level)

    app = FastAPI(
        title="study_rag admin REST API",
        summary="Enterprise knowledge base management & retrieval debug API",
        version="0.1.0",
        description=(
            "## 简介\n\n"
            "`study_rag` 是基于 LlamaIndex + LangChain 的企业知识库检索服务。\n"
            "知识内容通过 MCP（streamable_http transport）暴露给外部 Agent，\n"
            "本服务（admin REST）用于**管理**知识库和**调试**检索结果。\n\n"
            "## 主要功能\n\n"
            "- **KB 管理**：列出/查看知识库\n"
            "- **文档管理**：添加（整篇 / 切块 / 批量）、查询、删除\n"
            "- **检索调试**：与 MCP `search_kb` 共享后端的检索端点\n"
            "- **可观测性**：`/metrics` 暴露 Prometheus 指标，`X-Request-Id` 跟踪请求\n"
            "- **健康检查**：`/health`、`/health/ready`、`/health/detailed`\n\n"
            "## 鉴权\n\n"
            "通过环境变量 `STUDY_RAG_ADMIN_TOKEN` 配置 Bearer Token：\n"
            "- 未设置：开发模式，所有 `/admin/*` 免鉴权\n"
            "- 设置后：必须传 `Authorization: Bearer <token>`，否则 401\n\n"
            "## 限流\n\n"
            "按 client IP（`X-Forwarded-For` > `X-Real-IP` > `client.host`）限流：\n"
            "- 默认 120 burst / 2 req·s⁻¹ steady（`STUDY_RAG_ADMIN_RATELIMIT_*` 可调）\n"
            "- 触发后返回 429 + `Retry-After` 头\n\n"
            "## 标签\n\n"
            "| Tag | 用途 |\n"
            "| --- | --- |\n"
            "| `admin` | KB / 文档 CRUD + 检索调试 |\n"
            "| `health` | 存活 / 就绪 / 详细探针 |\n\n"
            "## 关联服务\n\n"
            "- **MCP standalone**（生产推荐）：`uvicorn study_rag.mcp_standalone:app`\n"
            "- **MCP Inspector 调试**：`npx @modelcontextprotocol/inspector`\n"
        ),
        contact={
            "name": "study_rag",
            "url": "https://example.com/study_rag",
            "email": "dev@example.com",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        openapi_tags=[
            {
                "name": "admin",
                "description": (
                    "管理面 REST API（KB CRUD、文档管理、检索调试）。\n\n"
                    "**鉴权**：Bearer Token（`STUDY_RAG_ADMIN_TOKEN`）。\n"
                    "**限流**：默认 120 burst / 2 req·s⁻¹。"
                ),
                "externalDocs": {
                    "description": "MCP 工具对照",
                    "url": "https://modelcontextprotocol.io/",
                },
            },
            {
                "name": "health",
                "description": (
                    "健康检查端点：\n"
                    "- `GET /health` — 存活探针（K8s livenessProbe）\n"
                    "- `GET /health/ready` — 就绪探针（K8s readinessProbe）\n"
                    "- `GET /health/detailed` — 详细状态（限流 / 熔断 / KB / 组件）"
                ),
            },
        ],
        servers=[
            {"url": "http://localhost:8765", "description": "本地开发"},
            {"url": "https://rag-admin.internal.example.com", "description": "生产内部"},
        ],
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # 2. 注入 request-id 中间件
    app.add_middleware(RequestIDMiddleware)

    # 3. 准备 MCP 上下文（单例），方便 admin 接口复用 manager
    ctx = MCPContext.default()

    # 4. 启动时初始化所有 KB
    @app.on_event("startup")
    async def _startup() -> None:
        logger.info("initializing_kbs", count=len(ctx.manager.list_summaries()))
        await ctx.manager.init_all()
        logger.info("admin_started", port=settings.port)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        logger.info("admin_shutdown")

    # 5. 注册 REST 路由
    from .api import admin, health
    from .web import mount_admin_ui

    app.include_router(health.router)
    app.include_router(admin.router)

    # 6. 挂载管理控制台 SPA
    mount_admin_ui(app)

    # 7. /metrics 端点
    @app.get(
        "/metrics",
        response_class=PlainTextResponse,
        include_in_schema=False,
        summary="Prometheus 指标",
    )
    async def metrics() -> str:
        """Prometheus metrics 端点（不被 OpenAPI 收录）。"""
        return get_metrics().render()

    return app


# uvicorn 入口
app = create_app()


def main() -> None:
    """CLI 入口。"""
    import uvicorn

    settings = get_server_settings()
    uvicorn.run(
        "study_rag.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        workers=settings.workers,
    )


if __name__ == "__main__":
    main()


# === MCP 独立部署 ===
# 如需将 MCP 跑在独立端口（推荐生产环境），使用：
#
#   python -m study_rag.mcp_standalone
# 或：
#   uvicorn study_rag.mcp_standalone:app --host 0.0.0.0 --port 8001
#
# 用 MCP Inspector 调试：
#   npx @modelcontextprotocol/inspector http://localhost:8001/mcp
