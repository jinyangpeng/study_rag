"""端到端测试：/admin/jobs 三个端点（list / get / cancel）。

覆盖：
  - GET    /admin/jobs              list（挂载验证 + 空列表 + kb_id 过滤）
  - GET    /admin/jobs/{job_id}     get（已知 / 未知 job_id）
  - DELETE /admin/jobs/{job_id}     cancel（已完成 / 未知 / 运行中）

注意：这些测试验证 router 已经被 ``app.include_router(jobs_router)`` 挂载。
如果未来谁误删了那行，``test_jobs_router_list_mounted`` 立刻会从 200 变 404。
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

# ---- fixtures（最小 app：mock embedder + mock vector store） ----


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")


@pytest.fixture
async def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """最小 app 用来测 jobs router 端点本身（不依赖完整 KB 上传链路）。"""
    from study_rag.capabilities.embedding import EmbeddingConfig, create_embedder
    from study_rag.capabilities.vector_store import (
        VectorStoreConfig,
        create_vector_store,
    )
    from study_rag.knowledge_bases import manager as mgr_mod
    from study_rag.knowledge_bases import registry as reg_mod
    from study_rag.knowledge_bases.manager import KnowledgeBaseManager
    from study_rag.settings import AppPaths

    kb_yaml = {
        "knowledge_bases": [
            {
                "kb_id": "kb_jobs_test",
                "name": "jobs-test",
                "description": "test",
                "department": "d",
                "collection": "c_jobs_test",
                "embedding": "mock_jobs",
                "reranker": None,
                "enabled": True,
            }
        ]
    }
    emb_yaml: dict[str, Any] = {
        "embeddings": {
            "mock_jobs": {
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

    reg_mod.reset_registry_cache()
    mgr_mod.reset_manager_singleton()

    registry = reg_mod.get_registry()
    embedders = {
        "mock_jobs": create_embedder(
            EmbeddingConfig(provider="mock", model_name="m", dimension=8)
        )
    }
    vs = create_vector_store(VectorStoreConfig(provider="mock"))
    manager = KnowledgeBaseManager(
        registry=registry, vector_store=vs, embedders=embedders
    )
    await manager.init_all()
    mgr_mod._manager_singleton = manager  # type: ignore[attr-defined]

    from study_rag.app import create_app

    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


# ---- 端点存在性（router 已挂载） ----


@pytest.mark.asyncio
async def test_jobs_router_list_mounted(app_client: AsyncClient) -> None:
    """GET /admin/jobs 必须返 200 + list（不是 404）。

    这条测试是 router 挂载的「保险丝」：
    谁误删 ``app.include_router(jobs_router)``，这条立刻挂。
    """
    r = await app_client.get("/admin/jobs")
    assert r.status_code == 200, f"router not mounted? got {r.status_code} body={r.text}"
    assert isinstance(r.json(), list)


@pytest.mark.asyncio
async def test_jobs_router_list_empty_when_no_jobs(app_client: AsyncClient) -> None:
    """没有任务时 list 应返空 list（不是 404 / 不是 None）。"""
    r = await app_client.get("/admin/jobs")
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_jobs_router_list_filter_by_kb_id(app_client: AsyncClient) -> None:
    """?kb_id=xx query 参数应能过滤。"""
    # 直接用 app.state.jobs 注入两个 job（一个 kb_jobs_test，一个 null）
    jobs = app_client._transport.app.state.jobs  # type: ignore[attr-defined]
    from study_rag.jobs.manager import JobManager

    assert isinstance(jobs, JobManager)

    async def noop(job_id, on_progress, is_cancelled):
        await asyncio.sleep(0.01)

    await jobs.submit("upload_doc", noop, kb_id="kb_jobs_test", doc_id="d1")
    await jobs.submit("upload_doc", noop, kb_id="kb_jobs_test", doc_id="d2")
    await jobs.submit("upload_doc", noop, kb_id="other_kb", doc_id="d3")
    await asyncio.sleep(0.05)

    r1 = await app_client.get("/admin/jobs")
    assert r1.status_code == 200
    assert len(r1.json()) == 3

    r2 = await app_client.get("/admin/jobs?kb_id=kb_jobs_test")
    assert r2.status_code == 200
    body = r2.json()
    assert len(body) == 2
    assert all(j["kb_id"] == "kb_jobs_test" for j in body)


# ---- GET /admin/jobs/{job_id} ----


@pytest.mark.asyncio
async def test_jobs_router_get_unknown_returns_404(
    app_client: AsyncClient,
) -> None:
    """不存在的 job_id 必须 404（不是 200 + None）。"""
    r = await app_client.get("/admin/jobs/nonexistent_id_abc")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_jobs_router_get_known_job(app_client: AsyncClient) -> None:
    """已知 job_id 应返 200 + JobInfo。"""
    jobs = app_client._transport.app.state.jobs  # type: ignore[attr-defined]
    from study_rag.jobs.models import JobStage

    async def fast(job_id, on_progress, is_cancelled):
        await on_progress(JobStage.CHUNKING, 1, 1, "done")

    jid = await jobs.submit("upload_doc", fast, kb_id="kb_jobs_test", doc_id="d1")
    await asyncio.sleep(0.05)

    r = await app_client.get(f"/admin/jobs/{jid}")
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == jid
    assert body["type"] == "upload_doc"
    assert body["status"] in ("done", "running")


# ---- DELETE /admin/jobs/{job_id} ----


@pytest.mark.asyncio
async def test_jobs_router_cancel_unknown_returns_not_found(
    app_client: AsyncClient,
) -> None:
    """取消未知 job → 200 + status="not_found"（不返 500）。"""
    r = await app_client.delete("/admin/jobs/nonexistent_id_abc")
    assert r.status_code == 200
    assert r.json() == {"status": "not_found"}


@pytest.mark.asyncio
async def test_jobs_router_cancel_completed_returns_not_cancellable(
    app_client: AsyncClient,
) -> None:
    """已完成 job → 200 + status="not_cancellable"。"""
    jobs = app_client._transport.app.state.jobs  # type: ignore[attr-defined]

    async def fast(job_id, on_progress, is_cancelled):
        pass

    jid = await jobs.submit("upload_doc", fast, kb_id="kb_jobs_test", doc_id="d1")
    await asyncio.sleep(0.05)  # 等任务完成

    r = await app_client.delete(f"/admin/jobs/{jid}")
    assert r.status_code == 200
    assert r.json() == {"status": "not_cancellable"}


@pytest.mark.asyncio
async def test_jobs_router_cancel_running_returns_cancelling(
    app_client: AsyncClient,
) -> None:
    """正在运行的 job → 200 + status="cancelling"（task 在后台会被中断）。"""
    jobs = app_client._transport.app.state.jobs  # type: ignore[attr-defined]

    async def slow(job_id, on_progress, is_cancelled):
        for _i in range(20):
            if is_cancelled():
                raise asyncio.CancelledError()
            await asyncio.sleep(0.05)

    jid = await jobs.submit("upload_doc", slow, kb_id="kb_jobs_test", doc_id="d1")
    await asyncio.sleep(0.02)  # 让它进入 RUNNING

    r = await app_client.delete(f"/admin/jobs/{jid}")
    assert r.status_code == 200
    assert r.json() == {"status": "cancelling"}

    # 等一会，job 状态应变成 CANCELLED
    await asyncio.sleep(0.2)
    r2 = await app_client.get(f"/admin/jobs/{jid}")
    assert r2.json()["status"] == "cancelled"
