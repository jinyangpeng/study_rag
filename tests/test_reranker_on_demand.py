"""get_reranker 按需实例化测试。

验证检索调试时，即使 reranker 未被任何 KB 引用（启动时未加载），
get_reranker 也能从 reranker.yaml 即时实例化并缓存。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from study_rag.capabilities.reranker.base import RerankerConfig
from study_rag.knowledge_bases.manager import (
    ComponentUnavailableError,
    KnowledgeBaseManager,
)


@pytest.fixture
def manager_with_empty_rerankers() -> KnowledgeBaseManager:
    """构造一个 rerankers 字典为空的 manager（模拟启动时无 KB 引用 reranker）。"""
    return KnowledgeBaseManager(
        registry=MagicMock(),
        vector_store=MagicMock(),
        embedders={},
        rerankers={},
        docs_index_path=MagicMock(),
    )


def test_get_reranker_loaded_directly(manager_with_empty_rerankers):
    """已加载的 reranker 直接返回，不触发按需加载。"""
    fake_reranker = MagicMock()
    manager_with_empty_rerankers._rerankers["already_loaded"] = fake_reranker
    assert manager_with_empty_rerankers.get_reranker("already_loaded") is fake_reranker


def test_get_reranker_on_demand_load(manager_with_empty_rerankers):
    """未加载的 reranker 从 reranker.yaml 按需实例化并缓存。"""
    cfg = RerankerConfig(provider="none", top_k=5)
    fake_instance = MagicMock()

    with (
        patch(
            "study_rag.knowledge_bases.manager._load_single_reranker_config",
            return_value=cfg,
        ) as mock_load,
        patch(
            "study_rag.knowledge_bases.manager.create_reranker",
            return_value=fake_instance,
        ) as mock_create,
    ):
        result = manager_with_empty_rerankers.get_reranker("local_bge_base_zh")

    mock_load.assert_called_once_with("local_bge_base_zh")
    mock_create.assert_called_once_with(cfg)
    assert result is fake_instance
    # 缓存：再次获取不再触发加载/创建
    result2 = manager_with_empty_rerankers.get_reranker("local_bge_base_zh")
    assert result2 is fake_instance
    assert mock_load.call_count == 1
    assert mock_create.call_count == 1


def test_get_reranker_config_not_found_raises(manager_with_empty_rerankers):
    """配置文件中不存在的 reranker 名 → ComponentUnavailableError。"""
    with patch(
        "study_rag.knowledge_bases.manager._load_single_reranker_config",
        return_value=None,
    ):
        with pytest.raises(ComponentUnavailableError, match="not found in reranker.yaml"):
            manager_with_empty_rerankers.get_reranker("nonexistent")


def test_get_reranker_init_failure_raises(manager_with_empty_rerankers):
    """实例化失败（依赖缺失等）→ ComponentUnavailableError（而非 ImportError）。"""
    cfg = RerankerConfig(provider="http", protocol="tei", extra={"base_url": "http://x:8080"})

    with (
        patch(
            "study_rag.knowledge_bases.manager._load_single_reranker_config",
            return_value=cfg,
        ),
        patch(
            "study_rag.knowledge_bases.manager.create_reranker",
            side_effect=ImportError("missing httpx"),
        ),
    ):
        with pytest.raises(ComponentUnavailableError, match="failed to initialize"):
            manager_with_empty_rerankers.get_reranker("broken")
