"""健康检查（OpenAPI 标签: health）。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ..knowledge_bases.manager import build_default_manager
from ..knowledge_bases.registry import get_registry
from ..observability.circuit_breaker import get_openai_breaker, get_search_breaker
from ..observability.ratelimit import get_admin_limiter, get_search_limiter
from .admin import _admin_token

router = APIRouter(prefix="/health", tags=["health"])


@router.get(
    "",
    summary="存活探针",
    description="极简探针；K8s livenessProbe 用。返回 200 即进程在跑。",
)
async def health() -> dict:
    """基础健康检查（存活探针）。"""
    return {"status": "ok"}


@router.get(
    "/ready",
    summary="就绪探针",
    description="K8s readinessProbe 用；目前只返回 ready，详细状态见 /health/detailed。",
)
async def ready() -> dict:
    """就绪检查（依赖加载完成）。"""
    return {"status": "ready"}


@router.get(
    "/detailed",
    summary="详细健康检查",
    description=(
        "返回：\n"
        "- KB 总数 / 启用数\n"
        "- 已加载 embedder / reranker 数量\n"
        "- 限流器状态（容量 / 稳态 QPS / 当前跟踪的 key 数）\n"
        "- 熔断器状态（CLOSED / OPEN / HALF_OPEN）\n"
        "- 鉴权开关 / Registry 加载状态"
    ),
    responses={
        200: {
            "description": "状态详情",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "kbs_total": 6,
                        "kbs_enabled": 6,
                        "embedders": 2,
                        "rerankers": 1,
                        "ratelimit": {
                            "admin": {"capacity": 120, "refill_rate": 2.0, "tracked_keys": 3},
                            "search": {"capacity": 30, "refill_rate": 5.0, "tracked_keys": 0},
                        },
                        "circuit_breakers": {
                            "openai_embed": {"state": "closed"},
                            "search": {"state": "closed"},
                        },
                        "auth_enabled": False,
                        "registry_loaded": True,
                    }
                }
            },
        }
    },
)
async def detailed() -> dict[str, Any]:
    """详细健康检查：KB / 组件 / 限流 / 熔断 / 鉴权。"""
    manager = build_default_manager()
    kbs = manager.list_summaries()
    return {
        "status": "ok",
        "kbs_total": len(kbs),
        "kbs_enabled": sum(1 for k in kbs if k.enabled),
        "embedders": len(manager._embedders),  # type: ignore[attr-defined]
        "rerankers": len(manager._rerankers),  # type: ignore[attr-defined]
        "ratelimit": {
            "admin": get_admin_limiter().stats(),
            "search": get_search_limiter().stats(),
        },
        "circuit_breakers": {
            "openai_embed": get_openai_breaker().stats(),
            "search": get_search_breaker().stats(),
        },
        "auth_enabled": _admin_token() is not None,
        "registry_loaded": get_registry() is not None,
    }
