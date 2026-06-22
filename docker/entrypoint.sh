#!/bin/sh
# ============================================
#  entrypoint: 根据 CONTAINER_ROLE 选择启动 admin 或 mcp
# ============================================
set -e

case "${CONTAINER_ROLE}" in
    admin)
        echo "[entrypoint] starting admin REST on port ${ADMIN_PORT}"
        exec python -m uvicorn study_rag.app:app \
            --host 0.0.0.0 \
            --port "${ADMIN_PORT}" \
            --log-level info
        ;;
    mcp)
        echo "[entrypoint] starting MCP standalone on port ${MCP_PORT}"
        exec python -m uvicorn study_rag.mcp_standalone:app \
            --host 0.0.0.0 \
            --port "${MCP_PORT}" \
            --log-level info
        ;;
    *)
        echo "[entrypoint] unknown CONTAINER_ROLE='${CONTAINER_ROLE}', expected admin|mcp" >&2
        exit 1
        ;;
esac
