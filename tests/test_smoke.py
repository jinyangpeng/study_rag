"""Smoke tests: 最基本的导入和配置加载。"""

from __future__ import annotations

import importlib


def test_settings_import():
    from study_rag import settings
    assert settings.get_server_settings() is not None


def test_knowledge_bases_module():
    from study_rag.knowledge_bases import manager, registry
    assert manager is not None
    assert registry is not None


def test_capabilities_packages():
    for pkg in [
        "study_rag.capabilities.embedding",
        "study_rag.capabilities.vector_store",
        "study_rag.capabilities.reranker",
    ]:
        mod = importlib.import_module(pkg)
        assert mod is not None


def test_llamaindex_optional():
    """LlamaIndex 是可选的，缺包不应崩。"""
    from study_rag import capabilities

    # 顶层 import 不应该因缺 llama-index 而崩
    assert capabilities is not None


def test_mcp_module_import():
    from study_rag.mcp import context, server
    assert context.MCPContext is not None
    assert server.create_mcp_server is not None
