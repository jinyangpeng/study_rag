# syntax=docker/dockerfile:1.7
# ============================================
#  study_rag 镜像（多阶段构建）
#
#  阶段：
#    1. frontend-builder : 构建 React SPA → /app/src/study_rag/web/dist
#    2. base             : Python 运行时 + 依赖 + 前端产物
#
#  同一镜像支持 admin REST 与 MCP standalone 两种入口，
#  运行时用 CONTAINER_ROLE=admin|mcp 选择启动哪个。
#
#  构建参数：
#    EXTRAS       : pip install -e 的可选依赖组（逗号分隔），默认 llamaindex,vector-milvus
#                   例：--build-arg EXTRAS=llamaindex,vector-milvus,embedding-openai,reranker-bge
#    NODE_VERSION : 前端构建用 Node 版本，默认 20
# ============================================

# ---- Stage 1: 前端构建 ----
# WORKDIR=/app 镜像项目结构，使 vite outDir(../src/study_rag/web/dist) 输出到 /app/src/study_rag/web/dist
FROM node:${NODE_VERSION:-20}-bookworm-slim AS frontend-builder
WORKDIR /app

# 先装依赖（利用 layer cache）
COPY frontend/package.json frontend/package-lock.json* ./frontend/
RUN cd frontend && (npm ci || npm install)

# 复制前端源码并构建（production mode → base="/admin/ui/"）
COPY frontend/ ./frontend/
RUN cd frontend && npm run build

# ---- Stage 2: Python 运行时 ----
FROM python:3.11-slim AS base

ARG EXTRAS=llamaindex,vector-milvus

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CONTAINER_ROLE=admin \
    ADMIN_PORT=8765 \
    MCP_PORT=8001 \
    CONFIG_DIR=/app/configs

WORKDIR /app

# 系统依赖
#   curl           : healthcheck
#   build-essential: torch / sentence-transformers 等需要编译
#   tini           : PID 1 信号转发（uvicorn 优雅退出）
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl tini build-essential \
 && rm -rf /var/lib/apt/lists/*

# 先装 Python 依赖（利用 layer cache）
COPY pyproject.toml ./
COPY src ./src
# 拷贝前端构建产物到包内（让 admin UI 静态服务可用）
COPY --from=frontend-builder /app/src/study_rag/web/dist ./src/study_rag/web/dist
RUN pip install --upgrade pip \
 && pip install -e ".[${EXTRAS}]"

# 复制配置和文档
COPY configs ./configs
COPY README.md ./

# 创建非 root 用户（uid 1000）并赋权
RUN groupadd -r appuser -g 1000 \
 && useradd -r -g appuser -u 1000 -d /app -s /sbin/nologin appuser \
 && mkdir -p /app/data \
 && chown -R appuser:appuser /app

USER appuser

# 容器启动入口（由 docker-compose 选择 ROLE）
COPY --chown=appuser:appuser docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8765 8001

# HEALTHCHECK 根据 CONTAINER_ROLE 切换探针端点
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD sh -c 'case "$CONTAINER_ROLE" in \
        admin) curl -fsS http://127.0.0.1:${ADMIN_PORT}/health || exit 1 ;; \
        mcp)   curl -fsS http://127.0.0.1:${MCP_PORT}/health || exit 1 ;; \
        *) exit 1 ;; \
        esac'

ENTRYPOINT ["tini", "--", "/entrypoint.sh"]
