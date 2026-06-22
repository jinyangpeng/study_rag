"""FastAPI 中间件：request-id 注入、访问日志。

挂在 app 上后：
  - 每次请求生成 X-Request-Id（已有则复用）
  - 写入 contextvar（logger 自动注入到结构化字段）
  - 响应头返回 X-Request-Id
  - 访问日志：method / path / status / duration_ms
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .logging import get_logger, set_request_id

log = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """注入 request-id 上下文 + 访问日志。"""

    HEADER_NAME = "X-Request-Id"

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # 1. 复用或生成 request_id
        request_id = request.headers.get(self.HEADER_NAME) or uuid.uuid4().hex
        set_request_id(request_id)

        # 2. structlog 临时绑定（带 path/method）
        structlog.contextvars.bind_contextvars(
            method=request.method,
            path=request.url.path,
        )

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            log.exception(
                "request_failed",
                status=500,
                duration_ms=round(duration_ms, 2),
                error=str(e),
            )
            structlog.contextvars.clear_contextvars()
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers[self.HEADER_NAME] = request_id
        log.info(
            "request_completed",
            status=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        structlog.contextvars.clear_contextvars()
        return response
