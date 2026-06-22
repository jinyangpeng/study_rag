# syntax=docker/dockerfile:1.7
# ============================================
#  study_rag base image
#  - 同时支持 admin REST 与 MCP standalone 两种 entrypoint
#  - 运行时用 CONTAINER_ROLE=admin|mcp 选择启动哪个
# ============================================

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CONTAINER_ROLE=admin \
    ADMIN_PORT=8765 \
    MCP_PORT=8001

WORKDIR /app

# 系统依赖（curl 仅用于 healthcheck，build-essential 留给 torch/sentence-transformers）
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# 先装依赖（利用 layer cache）
COPY pyproject.toml ./
COPY src ./src
RUN pip install --upgrade pip \
 && pip install -e ".[dev,llamaindex]"

# 复制配置和文档
COPY configs ./configs
COPY README.md ./

# 配置目录：生产环境可挂载 /app/configs 覆盖
ENV CONFIG_DIR=/app/configs

# 容器启动入口（由 docker-compose 选择 ROLE）
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8765 8001

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD sh -c 'case "$CONTAINER_ROLE" in \
        admin) curl -fsS http://127.0.0.1:${ADMIN_PORT}/health/live || exit 1 ;; \
        mcp)   curl -fsS -X POST http://127.0.0.1:${MCP_PORT}/mcp -H "Content-Type: application/json" -d "{}" || exit 1 ;; \
        *) exit 1 ;; \
        esac'

ENTRYPOINT ["/entrypoint.sh"]
