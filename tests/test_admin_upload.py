"""端到端测试：/admin/parsers + /upload + /preview-chunk。

跑法：
    pytest tests/test_admin_upload.py -v

依赖：
    - httpx (ASGITransport + AsyncClient)
    - llama-index-core（reader / parser 需要；缺包时用 skip 兜底）
"""
from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

# ---- fixtures ----


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")


def _has_llama_index() -> bool:
    try:
        import llama_index.core  # noqa: F401
    except ImportError:
        return False
    return True


needs_li = pytest.mark.skipif(
    not _has_llama_index(), reason="llama-index-core 未装"
)


async def _wait_job_done(
    client: AsyncClient, job_id: str, timeout_s: float = 10.0
) -> dict:
    """轮询 /admin/jobs/{id} 直到 done / error / cancelled。"""
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        r = await client.get(f"/admin/jobs/{job_id}")
        if r.status_code == 200:
            info = r.json()
            if info["status"] in ("done", "error", "cancelled"):
                return info
        await asyncio.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout_s}s")


@pytest.fixture
async def app_with_kb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """建一个用 in-memory vector store + mock embedder 的最小 app。

    端到端覆盖 /admin/parsers / preview-chunk / upload 三个端点。
    """
    from study_rag.capabilities.embedding import EmbeddingConfig, create_embedder
    from study_rag.capabilities.vector_store import (
        VectorStoreConfig,
        create_vector_store,
    )
    from study_rag.knowledge_bases import manager as mgr_mod
    from study_rag.knowledge_bases import registry as reg_mod
    from study_rag.knowledge_bases.manager import KnowledgeBaseManager
    from study_rag.settings import AppPaths

    # 临时 yaml
    kb_yaml = {
        "knowledge_bases": [
            {
                "kb_id": "kb_up_admin",
                "name": "upload-admin",
                "description": "test",
                "department": "d",
                "collection": "c_up_admin",
                "embedding": "mock_admin",
                "reranker": None,
                "enabled": True,
            },
            {
                "kb_id": "hr_policies",
                "name": "hr-policies",
                "description": "KB with no embedder (negative test)",
                "department": "hr",
                "collection": "c_hr_policies",
                "embedding": "missing_embedder",
                "reranker": None,
                "enabled": True,
            },
        ]
    }
    emb_yaml: dict[str, Any] = {
        "embeddings": {
            "mock_admin": {
                "provider": "mock",
                "model_name": "m",
                "dimension": 8,
            },
        }
    }
    vs_yaml: dict[str, Any] = {"vector_store": {"provider": "mock", "uri": ""}}
    rerank_yaml: dict[str, Any] = {"rerankers": {}}

    kb_path = tmp_path / "kb.yaml"
    emb_path = tmp_path / "emb.yaml"
    vs_path = tmp_path / "vs.yaml"
    rerank_path = tmp_path / "rerank.yaml"
    docs_path = tmp_path / "docs.json"
    _write_yaml(kb_path, kb_yaml)
    _write_yaml(emb_path, emb_yaml)
    _write_yaml(vs_path, vs_yaml)
    _write_yaml(rerank_path, rerank_yaml)

    monkeypatch.setattr(AppPaths, "KB_CONFIG", kb_path)
    monkeypatch.setattr(AppPaths, "EMBEDDING_CONFIG", emb_path)
    monkeypatch.setattr(AppPaths, "VECTOR_STORE_CONFIG", vs_path)
    monkeypatch.setattr(AppPaths, "RERANKER_CONFIG", rerank_path)
    monkeypatch.setattr(AppPaths, "DOCS_INDEX", docs_path)

    # 清空单例
    reg_mod.reset_registry_cache()
    mgr_mod.reset_manager_singleton()

    # 直接构造一个 manager 并塞进单例，避免 create_app 里的 startup 走远端
    registry = reg_mod.get_registry()
    embedders = {
        "mock_admin": create_embedder(
            EmbeddingConfig(provider="mock", model_name="m", dimension=8)
        )
    }
    vs = create_vector_store(VectorStoreConfig(provider="mock"))
    manager = KnowledgeBaseManager(
        registry=registry, vector_store=vs, embedders=embedders
    )
    await manager.init_all()
    mgr_mod._manager_singleton = manager  # type: ignore[attr-defined]

    # 强制 reload parser registry（指向 AppPaths.LLAMAINDEX_CONFIG）
    from study_rag.capabilities.llamaindex import registry as li_reg_mod

    li_reg_mod._registry_singleton = None  # type: ignore[attr-defined]

    # 显式调用 create_app（不跑 startup）
    from study_rag.app import create_app

    app = create_app()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


# ---- tests ----


@needs_li
@pytest.mark.asyncio
async def test_list_parsers(app_with_kb: AsyncClient) -> None:
    r = await app_with_kb.get("/admin/parsers")
    assert r.status_code == 200, r.text
    names = [p["name"] for p in r.json()]
    assert "sentence_512" in names
    assert "whole" in names


@needs_li
@pytest.mark.asyncio
async def test_preview_chunk_no_insert(app_with_kb: AsyncClient) -> None:
    payload = {
        "content": "First sentence. Second sentence.\n\nThird sentence.",
        "parser": "sentence_512",
    }
    r = await app_with_kb.post(
        "/admin/kbs/kb_up_admin/documents/preview-chunk", json=payload
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "chunks" in data
    assert len(data["chunks"]) >= 1
    for c in data["chunks"]:
        assert "chunk_index" in c
        assert "text" in c
        assert "char_count" in c
    assert data["parser"] == "sentence_512"
    assert data["total_chars"] == len(payload["content"])


@needs_li
@pytest.mark.asyncio
async def test_preview_chunk_unknown_parser_400(app_with_kb: AsyncClient) -> None:
    payload = {
        "content": "Hello world.",
        "parser": "not_a_parser",
    }
    r = await app_with_kb.post(
        "/admin/kbs/kb_up_admin/documents/preview-chunk", json=payload
    )
    assert r.status_code == 400
    assert "parser" in r.json()["detail"].lower()


@needs_li
@pytest.mark.asyncio
async def test_preview_chunk_empty_content_400(app_with_kb: AsyncClient) -> None:
    payload = {"content": "   \n\n  ", "parser": "sentence_512"}
    r = await app_with_kb.post(
        "/admin/kbs/kb_up_admin/documents/preview-chunk", json=payload
    )
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


@needs_li
@pytest.mark.asyncio
async def test_upload_txt_file(app_with_kb: AsyncClient) -> None:
    files = {"file": ("note.txt", io.BytesIO(b"Hello world."), "text/plain")}
    data = {"doc_id": "upload-1", "title": "Test", "parser": "sentence_512"}
    r = await app_with_kb.post(
        "/admin/kbs/kb_up_admin/documents/upload", files=files, data=data
    )
    # Phase 7: 异步上传 → 202 Accepted + job_id
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["doc_id"] == "upload-1"
    assert body["format"] == "txt"
    assert body["size_bytes"] == len(b"Hello world.")
    assert body["parser"] == "sentence_512"
    assert "job_id" in body
    # 等任务完成
    info = await _wait_job_done(app_with_kb, body["job_id"])
    assert info["status"] == "done", info
    # 验证 chunk 已写入
    r2 = await app_with_kb.get(
        "/admin/kbs/kb_up_admin/documents/upload-1/chunks"
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["total"] >= 1


@needs_li
@pytest.mark.asyncio
async def test_upload_unsupported_format_400(app_with_kb: AsyncClient) -> None:
    files = {
        "file": ("data.exe", io.BytesIO(b"binary"), "application/octet-stream")
    }
    data = {"doc_id": "x", "title": "x", "parser": "whole"}
    r = await app_with_kb.post(
        "/admin/kbs/kb_up_admin/documents/upload", files=files, data=data
    )
    assert r.status_code == 400
    assert "format" in r.json()["detail"].lower()


@needs_li
@pytest.mark.asyncio
async def test_upload_duplicate_doc_409(app_with_kb: AsyncClient) -> None:
    files = {"file": ("a.txt", io.BytesIO(b"abc"), "text/plain")}
    data = {"doc_id": "dup-1", "title": "T", "parser": "whole"}
    r1 = await app_with_kb.post(
        "/admin/kbs/kb_up_admin/documents/upload", files=files, data=data
    )
    assert r1.status_code == 202, r1.text
    # 第一次 job 完成
    await _wait_job_done(app_with_kb, r1.json()["job_id"])

    # 第二次同 doc_id，无 overwrite → 409
    files2 = {"file": ("b.txt", io.BytesIO(b"def"), "text/plain")}
    r2 = await app_with_kb.post(
        "/admin/kbs/kb_up_admin/documents/upload", files=files2, data=data
    )
    assert r2.status_code == 409

    # overwrite=true → 202
    data3 = {**data, "overwrite": "true"}
    r3 = await app_with_kb.post(
        "/admin/kbs/kb_up_admin/documents/upload", files=files2, data=data3
    )
    assert r3.status_code == 202, r3.text
    await _wait_job_done(app_with_kb, r3.json()["job_id"])


@needs_li
@pytest.mark.asyncio
async def test_upload_missing_doc_id_400(app_with_kb: AsyncClient) -> None:
    files = {"file": ("a.txt", io.BytesIO(b"x"), "text/plain")}
    data = {"title": "T", "parser": "whole"}  # 没传 doc_id
    r = await app_with_kb.post(
        "/admin/kbs/kb_up_admin/documents/upload", files=files, data=data
    )
    assert r.status_code == 400


@needs_li
@pytest.mark.asyncio
async def test_upload_unknown_kb_404(app_with_kb: AsyncClient) -> None:
    files = {"file": ("a.txt", io.BytesIO(b"x"), "text/plain")}
    data = {"doc_id": "d1", "title": "T", "parser": "whole"}
    r = await app_with_kb.post(
        "/admin/kbs/no_such_kb/documents/upload", files=files, data=data
    )
    # KB 不存在时，manager.add_document_from_upload 会让 registry.get_required 抛
    # 已在 endpoint 处转 4xx。这里期望非 200（4xx）。
    assert r.status_code in (400, 404)


# ---- Phase 5: semantic 策略走 KB embedder ----


@needs_li
@pytest.mark.asyncio
async def test_preview_chunk_semantic_uses_kb_embedder(
    app_with_kb: AsyncClient,
) -> None:
    """semantic 策略 + KB 已有 embedder → preview 应该 200。"""
    # 内容 > 1500 chars 以便 SemanticSplitter 有足够句子
    payload = {
        "content": (
            "Apple is a fruit that grows on trees. "
            "Bananas are yellow and rich in potassium. "
            "The sky appears blue due to Rayleigh scattering. "
            "Python is a popular programming language. "
            "Mountains are formed by tectonic plate movement. "
        ) * 30,
        "parser": "semantic",
    }
    assert len(payload["content"]) > 1500
    r = await app_with_kb.post(
        "/admin/kbs/kb_up_admin/documents/preview-chunk", json=payload
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["parser"] == "semantic"
    assert "chunks" in data
    # semantic 至少切出 1 块（mock embedder 也能切）
    assert data["total_chunks"] >= 1
    for c in data["chunks"]:
        assert "chunk_index" in c
        assert "text" in c
        assert "char_count" in c


@needs_li
@pytest.mark.asyncio
async def test_upload_with_semantic_parser(app_with_kb: AsyncClient) -> None:
    """上传 + semantic 策略能成功入库（自动注入 KB 的 embedder）。"""
    files = {
        "file": (
            "note.txt",
            io.BytesIO(
                b"Apple is a fruit. Banana is yellow. Sky is blue. "
                b"Python is a language. Mountains are tall. " * 40
            ),
            "text/plain",
        )
    }
    data = {
        "doc_id": "sem-1",
        "title": "Semantic Test",
        "parser": "semantic",
    }
    r = await app_with_kb.post(
        "/admin/kbs/kb_up_admin/documents/upload", files=files, data=data
    )
    # Phase 7: 异步 → 202
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["parser"] == "semantic"
    assert body["doc_id"] == "sem-1"
    # 等 job 完成
    info = await _wait_job_done(app_with_kb, body["job_id"])
    assert info["status"] == "done", info
    # 至少 1 块
    r2 = await app_with_kb.get(
        "/admin/kbs/kb_up_admin/documents/sem-1/chunks"
    )
    assert r2.status_code == 200
    assert r2.json()["total"] >= 1


@needs_li
@pytest.mark.asyncio
async def test_preview_chunk_semantic_kb_without_embedder_400(
    app_with_kb: AsyncClient,
) -> None:
    """KB 缺 embedder 时 semantic 策略应返回 4xx（不 5xx）。"""
    payload = {
        "content": "A. B. C. " * 200,
        "parser": "semantic",
    }
    r = await app_with_kb.post(
        "/admin/kbs/hr_policies/documents/preview-chunk", json=payload
    )
    # semantic + KB 缺 embedder → 4xx（不应该是 5xx，也不应该 200）
    assert r.status_code in (400, 503), r.text
    detail = r.json().get("detail", "")
    # 错误信息应当提示 embedder / semantic 相关
    assert (
        "embed" in detail.lower() or "semantic" in detail.lower()
    ), f"unexpected error detail: {detail}"


# ---- Phase 6.2: 文档分块查看端点 ----


@needs_li
@pytest.mark.asyncio
async def test_list_document_chunks(app_with_kb: AsyncClient) -> None:
    """上传后能列出该文档的所有 chunks。"""
    files = {
        "file": (
            "note.txt",
            io.BytesIO(b"Sentence 1. Sentence 2. Sentence 3. " * 100),
            "text/plain",
        )
    }
    data = {"doc_id": "list-chunks-1", "title": "List Test", "parser": "sentence_512"}
    r = await app_with_kb.post(
        "/admin/kbs/kb_up_admin/documents/upload", files=files, data=data
    )
    # Phase 7: 异步 → 202
    assert r.status_code == 202, r.text
    await _wait_job_done(app_with_kb, r.json()["job_id"])

    # 列出 chunks
    r = await app_with_kb.get(
        "/admin/kbs/kb_up_admin/documents/list-chunks-1/chunks?limit=50"
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kb_id"] == "kb_up_admin"
    assert body["doc_id"] == "list-chunks-1"
    assert body["total"] >= 1
    assert len(body["chunks"]) >= 1
    # 验证每个 chunk 有完整字段
    for c in body["chunks"]:
        assert "chunk_id" in c
        assert "chunk_index" in c
        assert "text" in c
        assert "char_count" in c
        assert c["char_count"] == len(c["text"])
    # chunks 按 chunk_index 升序
    indices = [c["chunk_index"] for c in body["chunks"]]
    assert indices == sorted(indices)


@needs_li
@pytest.mark.asyncio
async def test_list_chunks_pagination(app_with_kb: AsyncClient) -> None:
    """分页：limit + offset 正确。"""
    files = {
        "file": ("big.txt", io.BytesIO(b"X. " * 5000), "text/plain")
    }  # 10000 chars
    data = {"doc_id": "page-1", "title": "Page Test", "parser": "sentence_512"}
    r = await app_with_kb.post(
        "/admin/kbs/kb_up_admin/documents/upload", files=files, data=data
    )
    # Phase 7: 异步 → 202
    assert r.status_code == 202, r.text
    await _wait_job_done(app_with_kb, r.json()["job_id"])

    # 取前 2 个
    r1 = await app_with_kb.get(
        "/admin/kbs/kb_up_admin/documents/page-1/chunks?limit=2&offset=0"
    )
    assert r1.status_code == 200
    # 取下 2 个
    r2 = await app_with_kb.get(
        "/admin/kbs/kb_up_admin/documents/page-1/chunks?limit=2&offset=2"
    )
    assert r2.status_code == 200
    assert r1.json()["chunks"][0]["chunk_index"] == 0
    assert r1.json()["chunks"][1]["chunk_index"] == 1
    if len(r2.json()["chunks"]) >= 1:
        assert r2.json()["chunks"][0]["chunk_index"] == 2


@needs_li
@pytest.mark.asyncio
async def test_list_chunks_unknown_kb_404(app_with_kb: AsyncClient) -> None:
    """KB 不存在 → 404。"""
    r = await app_with_kb.get(
        "/admin/kbs/nonexistent_kb/documents/x/chunks"
    )
    assert r.status_code == 404


@needs_li
@pytest.mark.asyncio
async def test_list_chunks_unknown_doc_returns_empty(
    app_with_kb: AsyncClient,
) -> None:
    """doc 不存在 → 200 + 空 list（不是 404，因为 doc_id 在 metadata 里查不到是合法状态）。"""
    r = await app_with_kb.get(
        "/admin/kbs/kb_up_admin/documents/nonexistent_doc/chunks"
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert body["chunks"] == []
