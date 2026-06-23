"""Jobs 模块：异步任务调度 + 状态管理。

设计动机：
- 上传 / 重建索引 / 批量 embedding 等操作耗时数秒到数分钟，
  同步 HTTP 请求会触发前端 30s 超时。
- 通过 ``JobManager.submit()`` 把任务挂到后台 asyncio.Task，
  立即返回 job_id；前端通过 ``GET /admin/jobs/{id}`` 轮询状态。

模块结构：
- ``models.py``    JobInfo / JobStatus / JobStage  数据模型
- ``manager.py``   JobManager + JobStore 协议 + InMemoryJobStore
- ``pipeline.py``  run_chunking_pipeline  上传分块 → embedding → 入库
- ``api.py``       FastAPI 路由（list / get / cancel）

升级路径：
- 换 Redis：实现 JobStore 协议替代 InMemoryJobStore。
- 多副本：JobManager 跑在独立 worker 进程，状态写到共享存储。
"""
from .manager import (
    CancelCheck,
    InMemoryJobStore,
    JobManager,
    JobStore,
    ProgressCallback,
    Runner,
)
from .models import JobInfo, JobStage, JobStatus
from .pipeline import run_chunking_pipeline

__all__ = [
    "CancelCheck",
    "InMemoryJobStore",
    "JobInfo",
    "JobManager",
    "JobStage",
    "JobStatus",
    "JobStore",
    "ProgressCallback",
    "Runner",
    "run_chunking_pipeline",
]
