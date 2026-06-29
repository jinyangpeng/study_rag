#!/bin/sh
# ============================================
#  entrypoint: 根据 CONTAINER_ROLE 选择启动 admin 或 mcp
#
#  环境变量：
#    CONTAINER_ROLE  : admin | mcp
#    STUDY_RAG_HOST  : admin 监听地址（默认 0.0.0.0）
#    STUDY_RAG_PORT  : admin 监听端口（默认 3200）
#    MCP_HOST        : mcp 监听地址（默认 0.0.0.0）
#    MCP_PORT        : mcp 监听端口（默认 3220）
#    STUDY_RAG_LOG_LEVEL : uvicorn 日志级别（默认 info）
# ============================================
set -e

ADMIN_HOST="${STUDY_RAG_HOST:-0.0.0.0}"
ADMIN_PORT="${STUDY_RAG_PORT:-3200}"
_MCP_HOST="${MCP_HOST:-0.0.0.0}"
_MCP_PORT="${MCP_PORT:-3220}"
LOG_LEVEL="${STUDY_RAG_LOG_LEVEL:-info}"
# 转小写（uvicorn 要求 lowercase）
LOG_LEVEL=$(echo "$LOG_LEVEL" | tr '[:upper:]' '[:lower:]')

case "${CONTAINER_ROLE}" in
    admin)
        echo "[entrypoint] starting admin REST on ${ADMIN_HOST}:${ADMIN_PORT}"
        exec python -m uvicorn study_rag.app:app \
            --host "${ADMIN_HOST}" \
            --port "${ADMIN_PORT}" \
            --log-level "${LOG_LEVEL}"
        ;;
    mcp)
        echo "[entrypoint] starting MCP standalone on ${_MCP_HOST}:${_MCP_PORT}"
        exec python -m uvicorn study_rag.mcp_standalone:app \
            --host "${_MCP_HOST}" \
            --port "${_MCP_PORT}" \
            --log-level "${LOG_LEVEL}"
        ;;
    *)
        echo "[entrypoint] unknown CONTAINER_ROLE='${CONTAINER_ROLE}', expected admin|mcp" >&2
        exit 1
        ;;
esac
