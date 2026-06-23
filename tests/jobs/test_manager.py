"""JobManager 单元测试。"""
from __future__ import annotations

import asyncio

import pytest

from study_rag.jobs.manager import InMemoryJobStore, JobManager
from study_rag.jobs.models import JobInfo, JobStage, JobStatus

# ---- 基础行为 ----


@pytest.mark.asyncio
async def test_submit_returns_job_id():
    mgr = JobManager()

    async def noop_runner(job_id, on_progress, is_cancelled):
        pass

    jid = await mgr.submit("upload_doc", noop_runner)
    assert jid is not None
    assert isinstance(jid, str)
    assert len(jid) >= 8


@pytest.mark.asyncio
async def test_submit_unique_ids():
    """多次 submit 应返回不同 job_id。"""
    mgr = JobManager()

    async def noop_runner(job_id, on_progress, is_cancelled):
        pass

    ids = set()
    for _ in range(5):
        jid = await mgr.submit("upload_doc", noop_runner)
        ids.add(jid)
    assert len(ids) == 5


@pytest.mark.asyncio
async def test_submit_creates_pending_job():
    """submit 后立即 get 应能拿到 PENDING 状态（同步部分先于后台 task）。"""
    mgr = JobManager()

    started = asyncio.Event()

    async def slow_runner(job_id, on_progress, is_cancelled):
        started.set()
        await asyncio.sleep(0.5)

    jid = await mgr.submit("upload_doc", slow_runner)
    info = await mgr.get(jid)
    assert info is not None
    assert info.status in (JobStatus.PENDING, JobStatus.RUNNING)
    assert info.type == "upload_doc"
    # 等 runner 真正启动
    await started.wait()
    await asyncio.sleep(0.05)
    info = await mgr.get(jid)
    assert info.status == JobStatus.RUNNING


@pytest.mark.asyncio
async def test_get_returns_none_for_unknown():
    mgr = JobManager()
    info = await mgr.get("nonexistent")
    assert info is None


@pytest.mark.asyncio
async def test_submit_with_metadata():
    """submit 应能把 kb_id / doc_id / filename 存到 JobInfo。"""
    mgr = JobManager()

    async def noop_runner(job_id, on_progress, is_cancelled):
        pass

    jid = await mgr.submit(
        "upload_doc",
        noop_runner,
        kb_id="kb1",
        doc_id="d1",
        filename="test.txt",
    )
    info = await mgr.get(jid)
    assert info is not None
    assert info.kb_id == "kb1"
    assert info.doc_id == "d1"
    assert info.filename == "test.txt"


# ---- 状态转换 ----


@pytest.mark.asyncio
async def test_done_after_runner_completes():
    mgr = JobManager()

    async def ok_runner(job_id, on_progress, is_cancelled):
        await on_progress(JobStage.CHUNKING, 0, 1, "start")
        await on_progress(JobStage.CHUNKING, 1, 1, "end")

    jid = await mgr.submit("upload_doc", ok_runner)
    # 等后台 task 完成
    await asyncio.sleep(0.1)
    info = await mgr.get(jid)
    assert info is not None
    assert info.status == JobStatus.DONE
    assert info.progress == 1.0
    assert info.stage == JobStage.DONE


@pytest.mark.asyncio
async def test_error_when_runner_raises():
    mgr = JobManager()

    async def fail_runner(job_id, on_progress, is_cancelled):
        raise RuntimeError("boom")

    jid = await mgr.submit("upload_doc", fail_runner)
    await asyncio.sleep(0.1)
    info = await mgr.get(jid)
    assert info is not None
    assert info.status == JobStatus.ERROR
    assert info.error is not None
    assert "boom" in info.error


@pytest.mark.asyncio
async def test_cancel_marks_cancelled():
    mgr = JobManager()

    async def slow_runner(job_id, on_progress, is_cancelled):
        for i in range(10):
            if is_cancelled():
                raise asyncio.CancelledError()
            await asyncio.sleep(0.05)
            await on_progress(JobStage.CHUNKING, i + 1, 10, f"step {i + 1}")

    jid = await mgr.submit("upload_doc", slow_runner)
    await asyncio.sleep(0.05)
    cancelled = await mgr.cancel(jid)
    assert cancelled is True
    await asyncio.sleep(0.2)
    info = await mgr.get(jid)
    assert info is not None
    assert info.status == JobStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_returns_false_for_completed():
    """已完成的任务 cancel 应返回 False。"""
    mgr = JobManager()

    async def ok_runner(job_id, on_progress, is_cancelled):
        pass

    jid = await mgr.submit("upload_doc", ok_runner)
    await asyncio.sleep(0.1)
    cancelled = await mgr.cancel(jid)
    assert cancelled is False


@pytest.mark.asyncio
async def test_cancel_returns_false_for_unknown():
    mgr = JobManager()
    cancelled = await mgr.cancel("nonexistent")
    assert cancelled is False


# ---- 列表 ----


@pytest.mark.asyncio
async def test_list_returns_all_jobs():
    mgr = JobManager()

    async def noop_runner(job_id, on_progress, is_cancelled):
        await asyncio.sleep(0.01)

    for i in range(3):
        await mgr.submit("upload_doc", noop_runner, doc_id=f"d{i}")
    await asyncio.sleep(0.05)
    jobs = await mgr.list()
    assert len(jobs) == 3


@pytest.mark.asyncio
async def test_list_filter_by_kb_id():
    mgr = JobManager()

    async def noop_runner(job_id, on_progress, is_cancelled):
        await asyncio.sleep(0.01)

    await mgr.submit("upload_doc", noop_runner, kb_id="kb1")
    await mgr.submit("upload_doc", noop_runner, kb_id="kb1")
    await mgr.submit("upload_doc", noop_runner, kb_id="kb2")
    await asyncio.sleep(0.05)
    jobs = await mgr.list(kb_id="kb1")
    assert len(jobs) == 2
    assert all(j.kb_id == "kb1" for j in jobs)


@pytest.mark.asyncio
async def test_list_sorted_by_created_at_desc():
    """列表按 created_at 倒序（最新的在前）。"""
    mgr = JobManager()

    async def noop_runner(job_id, on_progress, is_cancelled):
        await asyncio.sleep(0.01)

    await mgr.submit("upload_doc", noop_runner)
    await asyncio.sleep(0.01)
    await mgr.submit("upload_doc", noop_runner)
    await asyncio.sleep(0.01)
    await mgr.submit("upload_doc", noop_runner)
    await asyncio.sleep(0.05)
    jobs = await mgr.list()
    # 最新（最后 submit）的应在最前
    for a, b in zip(jobs, jobs[1:], strict=False):
        assert a.created_at >= b.created_at


# ---- 进度回调 ----


@pytest.mark.asyncio
async def test_progress_callback_updates_info():
    """on_progress 应更新 JobInfo 的 stage/current/total/progress。"""
    mgr = JobManager()
    captured = []

    async def runner(job_id, on_progress, is_cancelled):
        await on_progress(JobStage.EMBEDDING, 0, 10, "start")
        info = await mgr.get(job_id)
        captured.append(("after_start", info.progress, info.current))
        await on_progress(JobStage.EMBEDDING, 5, 10, "half")
        info = await mgr.get(job_id)
        captured.append(("after_half", info.progress, info.current))
        await on_progress(JobStage.EMBEDDING, 10, 10, "end")
        info = await mgr.get(job_id)
        captured.append(("after_end", info.progress, info.current))

    jid = await mgr.submit("upload_doc", runner)
    await asyncio.sleep(0.1)
    # 最后一次 on_progress 后 progress 应该是 1.0
    info = await mgr.get(jid)
    assert info is not None
    assert info.progress == 1.0
    # runner 结束后 manager 会把 stage 推到 DONE
    assert info.stage == JobStage.DONE
    # 验证 capture 序列
    assert captured[0][1] == 0.0
    assert captured[1][1] == 0.5
    assert captured[2][1] == 1.0
    # 中间的 capture 里 stage 应该是 EMBEDDING（runner 内的状态）
    assert captured[0][2] == 0
    assert captured[1][2] == 5
    assert captured[2][2] == 10


@pytest.mark.asyncio
async def test_progress_zero_total_yields_zero_progress():
    """total=0 时 progress 应为 0（避免除零）。"""
    mgr = JobManager()

    async def runner(job_id, on_progress, is_cancelled):
        await on_progress(JobStage.PARSING, 0, 0, "empty")
        info = await mgr.get(job_id)
        assert info.progress == 0.0

    jid = await mgr.submit("upload_doc", runner)
    await asyncio.sleep(0.1)
    info = await mgr.get(jid)
    assert info is not None
    assert info.status == JobStatus.DONE


# ---- JobStore 抽象 ----


@pytest.mark.asyncio
async def test_in_memory_store_basic():
    store = InMemoryJobStore()
    info = JobInfo(job_id="x1", type="upload_doc")
    await store.save(info)
    loaded = await store.load("x1")
    assert loaded is not None
    assert loaded.job_id == "x1"


@pytest.mark.asyncio
async def test_custom_store_implementation():
    """JobStore Protocol 应能接受自定义实现（未来可换 Redis）。"""

    class CustomStore:
        def __init__(self) -> None:
            self._data: dict[str, JobInfo] = {}

        async def save(self, info: JobInfo) -> None:
            self._data[info.job_id] = info

        async def load(self, job_id: str) -> JobInfo | None:
            return self._data.get(job_id)

        async def list_all(self) -> list[JobInfo]:
            return list(self._data.values())

    custom = CustomStore()
    mgr = JobManager(store=custom)  # type: ignore[arg-type]

    async def noop_runner(job_id, on_progress, is_cancelled):
        pass

    jid = await mgr.submit("upload_doc", noop_runner)
    await asyncio.sleep(0.1)
    info = await mgr.get(jid)
    assert info is not None
    assert info.status == JobStatus.DONE
    # 自定义 store 真的存了
    assert jid in custom._data


# ---- 边界 ----


@pytest.mark.asyncio
async def test_concurrent_submits():
    """并发 submit 多个任务，应都能完成。"""
    mgr = JobManager()

    async def ok_runner(job_id, on_progress, is_cancelled):
        await on_progress(JobStage.CHUNKING, 1, 1, "done")
        await asyncio.sleep(0.01)

    jids = await asyncio.gather(
        *[mgr.submit("upload_doc", ok_runner) for _ in range(10)]
    )
    assert len(set(jids)) == 10
    # 等待所有完成
    await asyncio.sleep(0.3)
    for jid in jids:
        info = await mgr.get(jid)
        assert info is not None
        assert info.status == JobStatus.DONE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
