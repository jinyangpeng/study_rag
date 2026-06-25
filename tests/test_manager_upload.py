"""Manager 层的 upload / preview 测试。"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml

from study_rag.capabilities.embedding import EmbeddingConfig, create_embedder
from study_rag.capabilities.vector_store import (
    VectorStoreConfig,
    create_vector_store,
)
from study_rag.knowledge_bases.manager import KnowledgeBaseManager
from study_rag.knowledge_bases.registry import (
    get_registry,
    reset_registry_cache,
)
from study_rag.settings import AppPaths


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")


@pytest.fixture
def manager_with_kb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """建一个用 mock embedder + mock vector store 的最小 manager + KB 'kb_up'。"""
    kb_yaml = {
        "knowledge_bases": [
            {
                "kb_id": "kb_up",
                "name": "upload",
                "description": "test",
                "department": "d",
                "collection": "c_up",
                "embedding": "mock_up",
                "reranker": None,
                "enabled": True,
            }
        ]
    }
    emb_yaml: dict[str, Any] = {
        "embeddings": {
            "mock_up": {"provider": "mock", "model_name": "m", "dimension": 8},
        }
    }
    vs_yaml: dict[str, Any] = {"vector_store": {"provider": "mock", "uri": ""}}
    rerank_yaml: dict[str, Any] = {"rerankers": {}}

    _write_yaml(tmp_path / "kb.yaml", kb_yaml)
    _write_yaml(tmp_path / "emb.yaml", emb_yaml)
    _write_yaml(tmp_path / "vs.yaml", vs_yaml)
    _write_yaml(tmp_path / "rerank.yaml", rerank_yaml)

    monkeypatch.setattr(AppPaths, "KB_CONFIG", tmp_path / "kb.yaml")
    monkeypatch.setattr(AppPaths, "EMBEDDING_CONFIG", tmp_path / "emb.yaml")
    monkeypatch.setattr(AppPaths, "VECTOR_STORE_CONFIG", tmp_path / "vs.yaml")
    monkeypatch.setattr(AppPaths, "RERANKER_CONFIG", tmp_path / "rerank.yaml")

    reset_registry_cache()

    registry = get_registry()
    embedders = {
        "mock_up": create_embedder(
            EmbeddingConfig(provider="mock", model_name="m", dimension=8)
        )
    }
    vs = create_vector_store(VectorStoreConfig(provider="mock"))
    manager = KnowledgeBaseManager(
        registry=registry,
        vector_store=vs,
        embedders=embedders,
    )

    async def _setup() -> None:
        await manager.init_all()

    asyncio.run(_setup())
    return manager


@pytest.mark.asyncio
async def test_add_document_from_upload_persists_doc_meta(manager_with_kb):
    """上传入库：document meta 写入 + chunk 写入向量库。"""
    manager = manager_with_kb
    chunks = await manager.add_document_from_upload(
        kb_id="kb_up",
        doc_id="up-1",
        title="t",
        content="Sentence 1. Sentence 2.\n\nSentence 3. Sentence 4.",
        source="",
        metadata={"filename": "note.txt"},
        parser_name="sentence_512",
    )
    assert chunks >= 1

    # DocumentMeta 写入
    meta = manager.get_document("kb_up", "up-1")
    assert meta is not None
    assert meta.title == "t"
    assert meta.metadata.get("filename") == "note.txt"


@pytest.mark.asyncio
async def test_add_document_from_upload_uses_default_parser(manager_with_kb):
    """parser_name=None 时回退到默认 sentence 策略。"""
    manager = manager_with_kb
    chunks = await manager.add_document_from_upload(
        kb_id="kb_up",
        doc_id="up-default",
        title="t",
        content="A. B. C. D.",
        parser_name=None,
    )
    assert chunks >= 1
    assert manager.get_document("kb_up", "up-default") is not None


@pytest.mark.asyncio
async def test_add_document_from_upload_unknown_parser_raises(manager_with_kb):
    """parser_name 不存在抛 KeyError。"""
    manager = manager_with_kb
    with pytest.raises(KeyError):
        await manager.add_document_from_upload(
            kb_id="kb_up",
            doc_id="up-bad",
            title="t",
            content="hello",
            parser_name="not_a_parser",
        )


@pytest.mark.asyncio
async def test_add_document_from_upload_empty_content_returns_zero(manager_with_kb):
    """空 content 返回 0，不抛错，不写入。"""
    manager = manager_with_kb
    chunks = await manager.add_document_from_upload(
        kb_id="kb_up",
        doc_id="up-empty",
        title="t",
        content="   \n\n  ",
        parser_name="sentence_512",
    )
    assert chunks == 0
    assert manager.get_document("kb_up", "up-empty") is None


@pytest.mark.asyncio
async def test_add_document_chunked_still_works(manager_with_kb):
    """确认 add_document_chunked 旧接口未被破坏。"""
    manager = manager_with_kb
    chunks = await manager.add_document_chunked(
        kb_id="kb_up",
        doc_id="old-style",
        title="legacy",
        content="A. B. C.\n\nD. E. F.",
        parser_config={"strategy": "sentence", "chunk_size": 16, "chunk_overlap": 4},
    )
    assert chunks >= 1


@pytest.mark.asyncio
async def test_add_document_from_upload_parser_label_is_named(manager_with_kb):
    """回归：上传时 DocumentMeta.parser 应该是命名 parser（如 'sentence_512'），
    而不是 strategy 名（如 'sentence'）。

    历史 bug：把 factory._config.strategy 当作 parser 存到 DocumentMeta，
    导致列表/详情显示 'sentence' 而不是用户选的 'sentence_512'。
    """
    manager = manager_with_kb
    await manager.add_document_from_upload(
        kb_id="kb_up",
        doc_id="up-named",
        title="t",
        content="alpha. beta. gamma. delta.",
        parser_name="sentence_512",
    )
    meta = manager.get_document("kb_up", "up-named")
    assert meta is not None
    # 关键不变量：parser 应该是命名 parser（人类可读）
    assert meta.parser == "sentence_512", (
        f"DocumentMeta.parser 应该是命名 parser 'sentence_512'，"
        f"实际是 {meta.parser!r}"
    )


@pytest.mark.asyncio
async def test_add_document_chunked_parser_label_falls_back_to_strategy(manager_with_kb):
    """add_document_chunked 没传 parser_name 时，parser 字段回退到 strategy。"""
    manager = manager_with_kb
    await manager.add_document_chunked(
        kb_id="kb_up",
        doc_id="up-cb",
        title="t",
        content="hello. world.",
        parser_config={"strategy": "sentence", "chunk_size": 8, "chunk_overlap": 0},
    )
    meta = manager.get_document("kb_up", "up-cb")
    assert meta is not None
    # 没传 parser_name 时回退到 strategy
    assert meta.parser in ("sentence", "token", "semantic", "whole")
