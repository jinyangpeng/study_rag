"""Circuit Breaker（熔断器，in-process，无新依赖）。

三态机：
  CLOSED      正常放行；连续失败超过 threshold → OPEN
  OPEN        直接拒绝；open_timeout 秒后进入 HALF_OPEN
  HALF_OPEN   放 1 个试探请求；成功 → CLOSED，失败 → OPEN

适用：
  - 保护外部 API（OpenAI、BGE、Cohere）不被连环 timeout 拖垮
  - 一旦下游不可用，立刻失败，让上层走降级（不重试）

指标：
  - study_rag_circuit_breaker_state{name, state}
  - study_rag_circuit_breaker_calls_total{name, outcome}
  - study_rag_circuit_breaker_open_total{name}
"""

from __future__ import annotations

import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from .logging import get_logger
from .metrics import get_metrics

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """熔断器状态。"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """熔断器 OPEN 时调用被拒绝。"""

    def __init__(self, name: str, retry_after_s: float) -> None:
        self.name = name
        self.retry_after_s = retry_after_s
        super().__init__(f"Circuit '{name}' is OPEN, retry after {retry_after_s:.1f}s")


@dataclass
class _State:
    """熔断器运行时状态。"""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    opened_at: float = 0.0
    half_open_token_held: bool = False  # HALF_OPEN 只放 1 个请求


class CircuitBreaker:
    """线程/异步安全的熔断器。

    用法:
        cb = CircuitBreaker(
            name="openai_embed",
            failure_threshold=5,
            open_timeout_s=30.0,
        )

        result = await cb.call(openai_client.embeddings.create, **kwargs)
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        open_timeout_s: float = 30.0,
        half_open_max_calls: int = 1,
        expected_exceptions: tuple[type[BaseException], ...] = (Exception,),
    ) -> None:
        """
        Args:
            name: 熔断器名（用于 metrics label）
            failure_threshold: 连续失败多少次后 OPEN
            open_timeout_s: OPEN 状态持续多久后进入 HALF_OPEN
            half_open_max_calls: HALF_OPEN 状态允许放行的探测请求数
            expected_exceptions: 视为失败的异常类型（其它异常不计入）
        """
        self._name = name
        self._failure_threshold = failure_threshold
        self._open_timeout_s = open_timeout_s
        self._half_open_max_calls = half_open_max_calls
        self._expected_exceptions = expected_exceptions
        self._state = _State()
        # RLock 因为 call() 内部会调用 retry_after()/stats() 等子方法，
        # 它们也要加锁；普通 Lock 会自死锁
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> CircuitState:
        """当前状态（带自动 OPEN→HALF_OPEN 转换）。"""
        with self._lock:
            self._maybe_transition_open_to_half_open()
            return self._state.state

    def _maybe_transition_open_to_half_open(self) -> None:
        """如果 OPEN 超时，自动转 HALF_OPEN。"""
        if self._state.state != CircuitState.OPEN:
            return
        if time.monotonic() - self._state.opened_at >= self._open_timeout_s:
            self._state.state = CircuitState.HALF_OPEN
            self._state.half_open_token_held = False
            logger.info("circuit_breaker_half_open", name=self._name)
            self._record_state_metric()

    def _record_state_metric(self) -> None:
        """把当前状态写到 metric（用 counter 累加 1 表示该状态出现一次）。"""
        get_metrics().inc(
            "study_rag_circuit_breaker_state",
            {"name": self._name, "state": self._state.state.value},
        )

    def _allow_request(self) -> bool:
        """是否允许本次请求。"""
        self._maybe_transition_open_to_half_open()

        if self._state.state == CircuitState.CLOSED:
            return True
        if self._state.state == CircuitState.OPEN:
            return False
        # HALF_OPEN: 只放 N 个探测
        if not self._state.half_open_token_held:
            self._state.half_open_token_held = True
            return True
        return False

    def _on_success(self) -> None:
        with self._lock:
            if self._state.state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
                logger.info("circuit_breaker_closed", name=self._name)
            self._state = _State(state=CircuitState.CLOSED)
            self._record_state_metric()

    def _on_failure(self) -> None:
        with self._lock:
            self._state.failure_count += 1
            if self._state.state == CircuitState.HALF_OPEN:
                # 探测失败 → 直接回 OPEN
                self._state.state = CircuitState.OPEN
                self._state.opened_at = time.monotonic()
                logger.warning("circuit_breaker_reopened", name=self._name)
                self._record_state_metric()
                get_metrics().inc(
                    "study_rag_circuit_breaker_open_total", {"name": self._name}
                )
            elif (
                self._state.state == CircuitState.CLOSED
                and self._state.failure_count >= self._failure_threshold
            ):
                self._state.state = CircuitState.OPEN
                self._state.opened_at = time.monotonic()
                logger.warning(
                    "circuit_breaker_open",
                    name=self._name,
                    failure_count=self._state.failure_count,
                )
                self._record_state_metric()
                get_metrics().inc(
                    "study_rag_circuit_breaker_open_total", {"name": self._name}
                )

    def retry_after(self) -> float:
        """OPEN 状态下还需等多久才能进入 HALF_OPEN。"""
        with self._lock:
            if self._state.state != CircuitState.OPEN:
                return 0.0
            elapsed = time.monotonic() - self._state.opened_at
            return max(0.0, self._open_timeout_s - elapsed)

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """通过熔断器执行一个 async 函数。

        Raises:
            CircuitOpenError: 熔断器 OPEN 时
            *func* 抛出的异常: 调用本身失败
        """
        with self._lock:
            if not self._allow_request():
                retry_after = self.retry_after()
                get_metrics().inc(
                    "study_rag_circuit_breaker_calls_total",
                    {"name": self._name, "outcome": "rejected"},
                )
                raise CircuitOpenError(self._name, retry_after)

        try:
            result = await func(*args, **kwargs)
        except self._expected_exceptions:
            self._on_failure()
            get_metrics().inc(
                "study_rag_circuit_breaker_calls_total",
                {"name": self._name, "outcome": "error"},
            )
            raise
        else:
            self._on_success()
            get_metrics().inc(
                "study_rag_circuit_breaker_calls_total",
                {"name": self._name, "outcome": "success"},
            )
            return result

    def stats(self) -> dict[str, Any]:
        """当前状态（用于 /health/detailed）。"""
        with self._lock:
            return {
                "name": self._name,
                "state": self.state.value,
                "failure_count": self._state.failure_count,
                "success_count": self._state.success_count,
                "retry_after_s": round(self.retry_after(), 2),
            }

    def reset(self) -> None:
        """重置（测试用）。"""
        with self._lock:
            self._state = _State()
            self._record_state_metric()


# ---- 预置熔断器 ----

_openai_cb: CircuitBreaker | None = None
_search_cb: CircuitBreaker | None = None
_breaker_lock = threading.Lock()


def get_openai_breaker() -> CircuitBreaker:
    """OpenAI 调用的熔断器（lazy 单例，参数来自 settings）。"""
    global _openai_cb
    if _openai_cb is None:
        from ..settings import get_server_settings

        s = get_server_settings()
        with _breaker_lock:
            if _openai_cb is None:
                _openai_cb = CircuitBreaker(
                    name="openai_embed",
                    failure_threshold=s.openai_breaker_threshold,
                    open_timeout_s=s.openai_breaker_timeout_s,
                )
    return _openai_cb


def get_search_breaker() -> CircuitBreaker:
    """搜索链路的总熔断器（lazy 单例）。"""
    global _search_cb
    if _search_cb is None:
        from ..settings import get_server_settings

        s = get_server_settings()
        with _breaker_lock:
            if _search_cb is None:
                _search_cb = CircuitBreaker(
                    name="search",
                    failure_threshold=s.search_breaker_threshold,
                    open_timeout_s=s.search_breaker_timeout_s,
                )
    return _search_cb


def reset_breakers() -> None:
    """测试用：重置所有熔断器。"""
    global _openai_cb, _search_cb
    _openai_cb = None
    _search_cb = None
