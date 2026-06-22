"""structlog 配置：JSON 格式输出，自动注入时间戳/级别/request_id。

用法:
    from study_rag.observability.logging import configure_logging, get_logger
    configure_logging(level="INFO")
    log = get_logger(__name__)
    log.info("user_query", user_id="u1", kb_id="kb_a", latency_ms=42)
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog

# 全局 request_id 上下文（被 middleware 设置，被 logger 读取）
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def _add_request_id(_: Any, __: str, event_dict: structlog.types.EventDict) -> structlog.types.EventDict:
    """注入当前请求的 request_id（如有）。"""
    rid = request_id_var.get()
    if rid:
        event_dict["request_id"] = rid
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    """初始化结构化日志。

    行为:
      - 替换 stdlib 的 root logger handler → 走 structlog
      - 输出 JSON（生产）/ 彩色 console（开发）
      - 自动注入 timestamp / level / request_id / logger
    """
    # 1. 配置 stdlib logging（很多第三方库用 stdlib）
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
        force=True,
    )

    # 2. 配置 structlog 共享 processor chain
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        _add_request_id,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # 3. 选择 renderer
    is_tty = sys.stdout.isatty()
    if is_tty:
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # 4. 把 stdlib 输出也走 structlog 的 renderer
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """获取一个 structlog logger。"""
    return structlog.get_logger(name)


def set_request_id(request_id: str) -> None:
    """设置当前请求的 request_id（middleware 调用）。"""
    request_id_var.set(request_id)


def get_request_id() -> str | None:
    """获取当前请求的 request_id。"""
    return request_id_var.get()
