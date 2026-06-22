"""可观测性 + 可靠性：结构化日志 + 请求追踪 + Metrics + 限流 + 熔断。

不引入新依赖（structlog 已在 pyproject）。

模块:
  - configure_logging():  初始化 structlog，把 stdlib logging 也桥接过来
  - request_id 中间件:   为每次请求生成 X-Request-Id，写到 contextvar
  - MetricsRegistry:     in-memory 计数器/直方图，/metrics 端点暴露
  - TokenBucketLimiter:  token bucket 限流（per-key）
  - CircuitBreaker:      熔断器（CLOSED / OPEN / HALF_OPEN 三态）
"""
