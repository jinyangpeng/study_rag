"""Job 状态数据模型。

设计原则：
- JobInfo 是不可变的 Pydantic 模型；更新通过 manager.update() 创建新实例
  （model_copy）。
- status / stage 用 Enum 字符串值序列化（前端友好）。
- progress 在写入时 clamp 到 [0.0, 1.0]，避免除零和前端错乱。
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class JobStatus(str, Enum):
    """Job 生命周期状态。"""

    PENDING = "pending"  # 已注册，尚未开始
    RUNNING = "running"  # 正在跑
    DONE = "done"  # 正常结束
    ERROR = "error"  # 抛异常
    CANCELLED = "cancelled"  # 被用户取消


class JobStage(str, Enum):
    """Job 进度阶段（用于 UI 显示当前在干啥）。"""

    QUEUED = "queued"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    SAVING = "saving"
    DONE = "done"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobInfo(BaseModel):
    """Job 的完整状态快照。"""

    job_id: str
    type: str
    status: JobStatus = JobStatus.PENDING
    stage: JobStage = JobStage.QUEUED
    current: int = 0
    total: int = 0
    progress: float = 0.0
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None

    # 关联元数据
    kb_id: str | None = None
    doc_id: str | None = None
    filename: str | None = None

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @field_validator("progress")
    @classmethod
    def _clamp_progress(cls, v: float) -> float:
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v


__all__ = ["JobInfo", "JobStatus", "JobStage"]
