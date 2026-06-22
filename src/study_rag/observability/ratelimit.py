"""Token Bucket 限流（in-process，无新依赖）。

算法：每个 key 一个桶，按 rate (tokens/sec) 匀速补 token，容量 cap。
  - 请求时取 1 token，不够则 429
  - Retry-After = 缺 token 数 / rate 秒

适用场景：
  - Admin REST 端点：按 IP（或 api_key）限流
  - MCP Tool：按 api_key 限流（防止 agent 循环调打爆 OpenAI）
  - 外部 API 调用：保护下游 OpenAI/BGE 配额

不适用：
  - 多实例部署需要全局限流（这种情况用 Redis 集中计数）
  - 当前的实现是 per-process，多 worker 各算各的
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .logging import get_logger
from .metrics import get_metrics

logger = get_logger(__name__)


@dataclass
class _Bucket:
    """单个 key 的 token bucket。"""

    capacity: float
    refill_rate: float  # tokens per second
    tokens: float
    last_refill: float = field(default_factory=time.monotonic)


class TokenBucketLimiter:
    """线程安全的 token bucket 限流器。

    用法:
        limiter = TokenBucketLimiter(capacity=60, refill_rate=1.0)  # 60 burst, 1/s steady
        if not limiter.allow("ip:1.2.3.4"):
            raise HTTPException(429, "rate limit exceeded")
    """

    def __init__(
        self,
        capacity: int = 60,
        refill_rate: float = 1.0,
        name: str = "default",
    ) -> None:
        """初始化限流器。

        Args:
            capacity: 桶容量（最大突发请求数）
            refill_rate: 每秒补 token 数（持续 QPS）
            name: 限流器名字（用于 metrics label）
        """
        self._capacity = float(capacity)
        self._refill_rate = float(refill_rate)
        self._name = name
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, cost: float = 1.0) -> bool:
        """尝试消费 1 个 token。

        Args:
            key: 限流维度（IP、api_key、kb_id 等）
            cost: 本次消费多少 token（默认 1）

        Returns:
            True: 允许请求
            False: 被限流
        """
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(
                    capacity=self._capacity,
                    refill_rate=self._refill_rate,
                    tokens=self._capacity,
                )
                self._buckets[key] = bucket
            else:
                # 按时间补 token
                elapsed = now - bucket.last_refill
                if elapsed > 0:
                    bucket.tokens = min(
                        bucket.capacity,
                        bucket.tokens + elapsed * bucket.refill_rate,
                    )
                    bucket.last_refill = now

            if bucket.tokens >= cost:
                bucket.tokens -= cost
                allowed = True
                retry_after = 0.0
            else:
                # 缺多少 token，需要等多久
                deficit = cost - bucket.tokens
                retry_after = deficit / bucket.refill_rate if bucket.refill_rate > 0 else 1.0
                allowed = False

        # 记 metrics（锁外，避免阻塞其他 key）
        get_metrics().inc(
            "study_rag_ratelimit_total",
            {"name": self._name, "outcome": "allowed" if allowed else "rejected"},
        )
        if not allowed:
            logger.warning(
                "rate_limit_exceeded",
                limiter=self._name,
                key=key,
                retry_after_s=round(retry_after, 3),
            )
        return allowed

    def retry_after(self, key: str, cost: float = 1.0) -> float:
        """查询还需要等多久才能消费 cost 个 token（不消费）。"""
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                return 0.0
            elapsed = now - bucket.last_refill
            if elapsed > 0:
                bucket.tokens = min(
                    bucket.capacity,
                    bucket.tokens + elapsed * bucket.refill_rate,
                )
                bucket.last_refill = now
            if bucket.tokens >= cost:
                return 0.0
            deficit = cost - bucket.tokens
            return deficit / bucket.refill_rate if bucket.refill_rate > 0 else 1.0

    def reset(self, key: str | None = None) -> None:
        """重置桶（测试用）。"""
        with self._lock:
            if key is None:
                self._buckets.clear()
            else:
                self._buckets.pop(key, None)

    def stats(self) -> dict[str, Any]:
        """当前状态（用于 /health/detailed）。"""
        with self._lock:
            return {
                "name": self._name,
                "capacity": self._capacity,
                "refill_rate": self._refill_rate,
                "tracked_keys": len(self._buckets),
            }


# ---- 预置限流器（按需在 create_app 时构造并通过依赖注入） ----

_admin_limiter: TokenBucketLimiter | None = None
_search_limiter: TokenBucketLimiter | None = None


def get_admin_limiter() -> TokenBucketLimiter:
    """获取 admin REST 限流器（lazy 单例，参数来自 settings）。"""
    global _admin_limiter
    if _admin_limiter is None:
        from ..settings import get_server_settings

        s = get_server_settings()
        _admin_limiter = TokenBucketLimiter(
            capacity=s.admin_ratelimit_capacity,
            refill_rate=s.admin_ratelimit_per_sec,
            name="admin",
        )
    return _admin_limiter


def get_search_limiter() -> TokenBucketLimiter:
    """获取检索限流器（lazy 单例；防止 agent 循环调）。"""
    global _search_limiter
    if _search_limiter is None:
        from ..settings import get_server_settings

        s = get_server_settings()
        _search_limiter = TokenBucketLimiter(
            capacity=s.search_ratelimit_capacity,
            refill_rate=s.search_ratelimit_per_sec,
            name="search",
        )
    return _search_limiter


def reset_limiters() -> None:
    """测试用：重置所有限流器。"""
    global _admin_limiter, _search_limiter
    _admin_limiter = None
    _search_limiter = None
