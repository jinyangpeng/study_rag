"""模型配置管理（embedder / reranker CRUD）测试。

验证 config_store 的原子读写 + 引用保护逻辑：
  - 创建 / 列出 / 更新 / 删除
  - 重复创建报 409
  - 删除被 KB 引用的配置被拒绝
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from study_rag.knowledge_bases import config_store
from study_rag.knowledge_bases.config_store import ConfigNotFoundError


@pytest.fixture
def isolated_configs(tmp_path: Path, monkeypatch):
    """把配置文件路径指向临时目录，隔离测试。"""
    emb_path = tmp_path / "embeddings.yaml"
    rerank_path = tmp_path / "reranker.yaml"
    li_path = tmp_path / "llamaindex.yaml"
    emb_path.write_text("embeddings: {}\n", encoding="utf-8")
    rerank_path.write_text("rerankers: {}\n", encoding="utf-8")
    # llamaindex.yaml 含 settings 部分，验证 CRUD 不破坏它
    li_path.write_text(
        "parsers: {}\nsettings:\n  enabled: true\n  over_fetch: 4\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("study_rag.settings.AppPaths.EMBEDDING_CONFIG", emb_path)
    monkeypatch.setattr("study_rag.settings.AppPaths.RERANKER_CONFIG", rerank_path)
    monkeypatch.setattr("study_rag.settings.AppPaths.LLAMAINDEX_CONFIG", li_path)
    return emb_path, rerank_path, li_path


# ===== Embedder =====


class TestEmbedderConfigStore:
    def test_create_and_list(self, isolated_configs):
        cfg = {
            "provider": "openai",
            "model_name": "bge-m3",
            "dimension": 1024,
            "batch_size": 16,
            "description": "test embedder",
            "extra": {"base_url": "http://x:8080/v1"},
        }
        config_store.create_embedder_config("my_emb", cfg)
        all_cfgs = config_store.list_embedder_configs_raw()
        assert "my_emb" in all_cfgs
        assert all_cfgs["my_emb"]["provider"] == "openai"
        assert all_cfgs["my_emb"]["extra"]["base_url"] == "http://x:8080/v1"

    def test_duplicate_create_raises(self, isolated_configs):
        config_store.create_embedder_config("dup", {"provider": "mock"})
        with pytest.raises(ValueError, match="already exists"):
            config_store.create_embedder_config("dup", {"provider": "mock"})

    def test_update_merges(self, isolated_configs):
        config_store.create_embedder_config(
            "u1", {"provider": "openai", "dimension": 1024, "extra": {"a": 1}}
        )
        merged = config_store.update_embedder_config("u1", {"dimension": 768})
        assert merged["dimension"] == 768
        assert merged["provider"] == "openai"  # 未传的字段保留
        assert merged["extra"]["a"] == 1  # extra 保留

    def test_update_not_found(self, isolated_configs):
        with pytest.raises(ConfigNotFoundError):
            config_store.update_embedder_config("nope", {"dimension": 1})

    def test_delete(self, isolated_configs):
        config_store.create_embedder_config("del", {"provider": "mock"})
        removed = config_store.delete_embedder_config("del")
        assert removed["provider"] == "mock"
        assert "del" not in config_store.list_embedder_configs_raw()

    def test_delete_not_found(self, isolated_configs):
        with pytest.raises(ConfigNotFoundError):
            config_store.delete_embedder_config("nope")

    def test_write_preserves_yaml_structure(self, isolated_configs):
        emb_path = isolated_configs[0]
        config_store.create_embedder_config(
            "struct", {"provider": "bge", "model_name": "bge-m3", "extra": {}}
        )
        with emb_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "embeddings" in data
        assert "struct" in data["embeddings"]


# ===== Reranker =====


class TestRerankerConfigStore:
    def test_create_with_protocol(self, isolated_configs):
        cfg = {
            "provider": "http",
            "protocol": "tei",
            "model_name": "BAAI/bge-reranker-base",
            "top_k": 5,
            "extra": {"base_url": "http://127.0.0.1:8080"},
        }
        config_store.create_reranker_config("local_rr", cfg)
        raw = config_store.get_reranker_config_raw("local_rr")
        assert raw["protocol"] == "tei"
        assert raw["extra"]["base_url"] == "http://127.0.0.1:8080"

    def test_update_extra_replaces(self, isolated_configs):
        config_store.create_reranker_config(
            "e1", {"provider": "http", "extra": {"base_url": "old"}}
        )
        merged = config_store.update_reranker_config(
            "e1", {"extra": {"base_url": "new", "timeout": 10}}
        )
        assert merged["extra"] == {"base_url": "new", "timeout": 10}

    def test_delete_reranker(self, isolated_configs):
        config_store.create_reranker_config("d1", {"provider": "none"})
        config_store.delete_reranker_config("d1")
        assert "d1" not in config_store.list_reranker_configs_raw()


# ===== Parser =====


class TestParserConfigStore:
    def test_create_and_list(self, isolated_configs):
        cfg = {
            "strategy": "sentence",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "paragraph_separator": "\n\n",
        }
        config_store.create_parser_config("my_parser", cfg)
        all_cfgs = config_store.list_parser_configs_raw()
        assert "my_parser" in all_cfgs
        assert all_cfgs["my_parser"]["strategy"] == "sentence"

    def test_duplicate_create_raises(self, isolated_configs):
        config_store.create_parser_config("dup", {"strategy": "whole"})
        with pytest.raises(ValueError, match="already exists"):
            config_store.create_parser_config("dup", {"strategy": "whole"})

    def test_update_merges(self, isolated_configs):
        config_store.create_parser_config(
            "u1", {"strategy": "sentence", "chunk_size": 512}
        )
        merged = config_store.update_parser_config("u1", {"chunk_size": 1024})
        assert merged["chunk_size"] == 1024
        assert merged["strategy"] == "sentence"  # 未传字段保留

    def test_delete(self, isolated_configs):
        config_store.create_parser_config("del", {"strategy": "token"})
        config_store.delete_parser_config("del")
        assert "del" not in config_store.list_parser_configs_raw()

    def test_preserves_settings_section(self, isolated_configs):
        """关键：parser CRUD 不能破坏 llamaindex.yaml 的 settings 部分。"""
        _, _, li_path = isolated_configs
        config_store.create_parser_config(
            "s1", {"strategy": "sentence", "chunk_size": 256}
        )
        config_store.update_parser_config("s1", {"chunk_overlap": 30})
        with li_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # settings 完整保留
        assert data["settings"]["enabled"] is True
        assert data["settings"]["over_fetch"] == 4
        # parser 写入成功
        assert data["parsers"]["s1"]["chunk_size"] == 256
        assert data["parsers"]["s1"]["chunk_overlap"] == 30
