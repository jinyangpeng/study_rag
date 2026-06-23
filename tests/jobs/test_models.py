"""Job 状态数据模型测试。"""
from __future__ import annotations

import pytest

from study_rag.jobs.models import JobInfo, JobStage, JobStatus


def test_job_info_default_status_is_pending():
    info = JobInfo(job_id="abc123", type="upload_doc")
    assert info.status == JobStatus.PENDING
    assert info.stage == JobStage.QUEUED
    assert info.progress == 0.0
    assert info.error is None
    assert info.result is None
    assert info.current == 0
    assert info.total == 0
    assert info.message == ""


def test_job_info_progress_clamped_above_one():
    info = JobInfo(job_id="abc", type="upload_doc", progress=1.5)
    assert info.progress == 1.0


def test_job_info_progress_clamped_below_zero():
    info = JobInfo(job_id="abc", type="upload_doc", progress=-0.5)
    assert info.progress == 0.0


def test_job_info_progress_valid_range_kept():
    info = JobInfo(job_id="abc", type="upload_doc", progress=0.42)
    assert info.progress == 0.42


def test_job_info_optional_metadata():
    info = JobInfo(
        job_id="abc",
        type="upload_doc",
        kb_id="kb1",
        doc_id="d1",
        filename="test.txt",
    )
    assert info.kb_id == "kb1"
    assert info.doc_id == "d1"
    assert info.filename == "test.txt"


def test_job_info_timestamps_set():
    info = JobInfo(job_id="abc", type="upload_doc")
    assert info.created_at is not None
    assert info.updated_at is not None


def test_job_status_values():
    assert JobStatus.PENDING.value == "pending"
    assert JobStatus.RUNNING.value == "running"
    assert JobStatus.DONE.value == "done"
    assert JobStatus.ERROR.value == "error"
    assert JobStatus.CANCELLED.value == "cancelled"


def test_job_stage_values():
    assert JobStage.QUEUED.value == "queued"
    assert JobStage.PARSING.value == "parsing"
    assert JobStage.CHUNKING.value == "chunking"
    assert JobStage.EMBEDDING.value == "embedding"
    assert JobStage.SAVING.value == "saving"
    assert JobStage.DONE.value == "done"


def test_job_info_model_copy_creates_new_instance():
    info = JobInfo(job_id="abc", type="upload_doc")
    updated = info.model_copy(update={"status": JobStatus.RUNNING, "progress": 0.5})
    assert updated is not info
    assert updated.status == JobStatus.RUNNING
    assert info.status == JobStatus.PENDING  # 原对象不变


def test_job_info_serializes_to_json():
    info = JobInfo(job_id="abc", type="upload_doc")
    data = info.model_dump(mode="json")
    assert data["job_id"] == "abc"
    assert data["status"] == "pending"
    assert data["stage"] == "queued"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
