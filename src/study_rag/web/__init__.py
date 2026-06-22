"""管理控制台 UI 路由：挂载 React SPA 构建产物。

行为：
  - GET /admin/ui             → 307 重定向到 /admin/ui/
  - GET /admin/ui/            → 返回 SPA 入口 index.html
  - GET /admin/ui/assets/...  → 返回 Vite 构建的静态资源（js/css/图片/字体）
  - GET /admin/ui/<sub-path>  → SPA 路由 fallback，返回 index.html

构建产物路径：`src/study_rag/web/dist/`，由 `frontend/` 通过 `npm run build` 生成。

未构建时（首次 build 前）：
  - /admin/ui 渲染一个简单的提示页，引导用户运行 `just ui-build`。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from starlette.staticfiles import StaticFiles

from ..observability.logging import get_logger

logger = get_logger(__name__)

# SPA 构建产物根目录
# __file__ = src/study_rag/web/__init__.py → parent = src/study_rag/web/
WEB_DIST_DIR = Path(__file__).resolve().parent / "dist"


def mount_admin_ui(app: FastAPI) -> None:
    """把构建好的前端 SPA 挂到 /admin/ui/。

    如果 dist 目录不存在（首次 build 前），fallback 到一个简单的提示页。
    """
    router = APIRouter(prefix="/admin/ui", include_in_schema=False)
    index_html = WEB_DIST_DIR / "index.html"
    assets_dir = WEB_DIST_DIR / "assets"

    if not index_html.exists():
        logger.warning(
            "admin_ui_not_built",
            dist=str(WEB_DIST_DIR),
            hint="run `just ui-build` or `cd frontend && npm run build`",
        )

        @router.get("", response_class=HTMLResponse)
        @router.get("/", response_class=HTMLResponse)
        @router.get(
            "/{full_path:path}",
            response_class=HTMLResponse,
        )
        async def _not_built(full_path: str = "") -> str:
            return (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<title>study_rag Admin UI - not built</title>"
                "<style>body{font-family:system-ui;max-width:680px;margin:48px auto;"
                "padding:0 16px;line-height:1.6;color:#333}"
                "code{background:#f4f4f4;padding:2px 6px;border-radius:4px;font-size:14px}"
                "pre{background:#1e1e1e;color:#d4d4d4;padding:16px;border-radius:6px;overflow:auto}</style>"
                "</head><body>"
                "<h1>study_rag Admin UI 未构建</h1>"
                "<p>请先构建前端：</p>"
                "<pre>cd frontend\nnpm install\nnpm run build</pre>"
                "<p>或使用 justfile：</p>"
                "<pre>just ui-install\njust ui-build</pre>"
                f"<p>预期产物路径：<code>{WEB_DIST_DIR}</code></p>"
                "<p>OpenAPI 文档：<a href='/docs'>/docs</a></p>"
                "</body></html>"
            )

        app.include_router(router)
        return

    # ----- 已构建：挂载真实 SPA -----

    # 1. /admin/ui/assets/* → 静态资源（js/css/图片/字体）
    #    必须先 mount，否则会被下面的 catch-all 吞掉
    if assets_dir.exists():
        app.mount(
            "/admin/ui/assets",
            StaticFiles(directory=str(assets_dir)),
            name="admin_ui_assets",
        )

    # 2. /admin/ui/ → index.html（也覆盖 /admin/ui，因为 prefix 会被规范化）
    @router.get("", response_class=HTMLResponse)
    @router.get("/", response_class=HTMLResponse)
    async def _index() -> FileResponse:
        return FileResponse(index_html, media_type="text/html")

    # 3. SPA fallback：所有其他子路径（如 /admin/ui/dashboard）都返回 index.html
    @router.get(
        "/{full_path:path}",
        response_class=HTMLResponse,
    )
    async def _spa_fallback(full_path: str) -> FileResponse:
        # 安全：禁止路径穿越
        if ".." in full_path.split("/"):
            return FileResponse(index_html, media_type="text/html")
        return FileResponse(index_html, media_type="text/html")

    app.include_router(router)
    logger.info("admin_ui_mounted", dist=str(WEB_DIST_DIR))
