"""Job 调度 + 状态管理。

设计原则：
1. 进程内调度（asyncio.create_task）。
2. ``JobStore`` Protocol 抽象（未来可换 Redis / SQLite 实现）。
3. ``runner`` 是纯回调函数，不知道 JobManager 存在。
4. 失败 / 完成 / 取消都有完整状态记录。

为什么进程内 dict 就够：
- 单进程 FastAPI，asyncio 单线程；JobStore 不会跨进程被读。
- 上传任务跑几秒到几分钟，量小。
- 需要扩展时（多副本 / 跨进程）只换 JobStore 实现，runner 不变。

未来升级路径：
- 换 Redis：实现 RedisJobStore 替代 InMemoryJobStore。
- 跨实例：把 _tasks 移到独立 worker 进程（Celery / Arq / Dramatiq）。
- 持久化：在 _update 时写 SQLite / PostgreSQL。
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, Protocol

from .models import JobInfo, JobStage, JobStatus

logger = logging.getLogger(__name__)


# ---- 类型定义 ----


ProgressCallback = Callable[[JobStage, int, int, str], Awaitable[None]]
CancelCheck = Callable[[], bool]
Runner = Callable[[str, ProgressCallback, CancelCheck], Awaitable[None]]


# ---- JobStore 抽象 ----


class JobStore(Protocol):
    """Job 存储抽象（未来可换 Redis / SQLite）。"""

    async def save(self, info: JobInfo) -> None: ...
    async def load(self, job_id: str) -> JobInfo | None: ...
    async def list_all(self) -> list[JobInfo]: ...


class InMemoryJobStore:
    """默认 JobStore：进程内 dict + asyncio.Lock 保护并发。"""

    def __init__(self) -> None:
        self._jobs: dict[str, JobInfo] = {}
        self._lock = asyncio.Lock()

    async def save(self, info: JobInfo) -> None:
        async with self._lock:
            self._jobs[info.job_id] = info

    async def load(self, job_id: str) -> JobInfo | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def list_all(self) -> list[JobInfo]:
        async with self._lock:
            return list(self._jobs.values())


# ---- JobManager ----


class JobManager:
    """异步任务调度 + 状态管理。

    用法：
        mgr = JobManager()
        jid = await mgr.submit("upload_doc", runner, kb_id="kb1", doc_id="d1")
        info = await mgr.get(jid)        # 查询状态
        jobs = await mgr.list()          # 列出所有
        await mgr.cancel(jid)            # 取消
    """

    def __init__(self, store: JobStore | None = None) -> None:
        self._store: JobStore = store or InMemoryJobStore()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._cancelled: set[str] = set()

    async def submit(
        self,
        type_: str,
        runner: Runner,
        *,
        kb_id: str | None = None,
        doc_id: str | None = None,
        filename: str | None = None,
    ) -> str:
        """注册任务 + 启动后台 asyncio.Task。返回 job_id。"""
        job_id = uuid.uuid4().hex[:12]
        info = JobInfo(
            job_id=job_id,
            type=type_,
            kb_id=kb_id,
            doc_id=doc_id,
            filename=filename,
        )
        await self._store.save(info)
        task = asyncio.create_task(self._run(job_id, runner))
        self._tasks[job_id] = task
        logger.info("job submitted: %s (type=%s)", job_id, type_)
        return job_id

    async def _run(self, job_id: str, runner: Runner) -> None:
        """后台执行 runner，更新状态。"""
        await self._update(job_id, status=JobStatus.RUNNING)

        async def on_progress(
            stage: JobStage,
            current: int,
            total: int,
            message: str = "",
        ) -> None:
            progress = current / total if total > 0 else 0.0
            await self._update(
                job_id,
                stage=stage,
                current=current,
                total=total,
                progress=progress,
                message=message,
            )

        try:
            await runner(job_id, on_progress, lambda: self._is_cancelled(job_id))
            await self._update(
                job_id,
                status=JobStatus.DONE,
                stage=JobStage.DONE,
                progress=1.0,
                message="完成",
            )
            logger.info("job done: %s", job_id)
        except asyncio.CancelledError:
            await self._update(
                job_id,
                status=JobStatus.CANCELLED,
                message="已取消",
            )
            logger.info("job cancelled: %s", job_id)
        except Exception as e:  # noqa: BLE001
            await self._update(
                job_id,
                status=JobStatus.ERROR,
                error=f"{type(e).__name__}: {e}",
            )
            logger.exception("job error: %s", job_id)
        finally:
            # 清理 task 引用（避免内存泄漏）
            self._tasks.pop(job_id, None)
            self._cancelled.discard(job_id)

    def _is_cancelled(self, job_id: str) -> bool:
        return job_id in self._cancelled

    async def _update(self, job_id: str, **kwargs: Any) -> None:
        info = await self._store.load(job_id)
        if info is None:
            return
        new_info = info.model_copy(
            update={**kwargs, "updated_at": datetime.now(timezone.utc)}
        )
        await self._store.save(new_info)

    async def get(self, job_id: str) -> JobInfo | None:
        return await self._store.load(job_id)

    async def list(self, kb_id: str | None = None) -> list[JobInfo]:
        all_jobs = await self._store.list_all()
        if kb_id is not None:
            all_jobs = [j for j in all_jobs if j.kb_id == kb_id]
        return sorted(all_jobs, key=lambda j: j.created_at, reverse=True)

    async def cancel(self, job_id: str) -> bool:
        """请求取消任务。返回是否成功标记。

        - 未知 job_id：返回 False。
        - 已完成 / 失败 / 已取消：返回 False。
        - 正在跑：标记取消位 + 调 task.cancel()（runner 通常在
          下一次 ``is_cancelled()`` 检查时退出，抛 ``CancelledError``）。
        """
        info = await self._store.load(job_id)
        if info is None:
            return False
        if info.status in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED):
            return False
        self._cancelled.add(job_id)
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
        await self._update(job_id, message="正在取消...")
        return True


__all__ = ["JobManager", "InMemoryJobStore", "JobStore", "Runner", "ProgressCallback", "CancelCheck"]
