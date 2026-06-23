"""Jobs 的 FastAPI 路由。

端点：
  GET    /admin/jobs              列出所有 job（可选 kb_id 过滤）
  GET    /admin/jobs/{job_id}     获取单个 job 状态
  DELETE /admin/jobs/{job_id}     取消 job
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from .manager import JobManager
from .models import JobInfo

router = APIRouter(prefix="/admin/jobs", tags=["jobs"])


def _get_job_manager(request: Request) -> JobManager:
    return request.app.state.jobs


@router.get(
    "",
    response_model=list[JobInfo],
    summary="列出 jobs",
    description=(
        "返回所有 job 状态（按 created_at 倒序）。\n\n"
        "可选 query 参数：\n"
        "- `kb_id`: 只返回该 KB 下的 jobs"
    ),
)
async def list_jobs(request: Request, kb_id: str | None = None) -> list[JobInfo]:
    """列出 jobs。"""
    mgr = _get_job_manager(request)
    return await mgr.list(kb_id=kb_id)


@router.get(
    "/{job_id}",
    response_model=JobInfo,
    summary="获取 job 状态",
    description=(
        "返回单个 job 的最新状态。\n\n"
        "前端轮询这个端点（推荐间隔 1s）来显示进度条 / 当前阶段 / 错误信息。"
    ),
    responses={404: {"description": "job 不存在"}},
)
async def get_job(request: Request, job_id: str) -> JobInfo:
    """获取 job 状态。"""
    mgr = _get_job_manager(request)
    info = await mgr.get(job_id)
    if info is None:
        raise HTTPException(404, f"job {job_id} not found")
    return info


@router.delete(
    "/{job_id}",
    summary="取消 job",
    description=(
        "请求取消一个正在运行 / 等待中的 job。\n\n"
        '- 未知 job_id：返回 ``{"status": "not_found"}``\n'
        '- 已完成 / 失败 / 已取消：返回 ``{"status": "not_cancellable"}``\n'
        '- 取消成功：返回 ``{"status": "cancelling"}``\n'
    ),
)
async def cancel_job(request: Request, job_id: str) -> dict[str, str]:
    """取消 job。"""
    mgr = _get_job_manager(request)
    info = await mgr.get(job_id)
    if info is None:
        return {"status": "not_found"}
    ok = await mgr.cancel(job_id)
    return {"status": "cancelling" if ok else "not_cancellable"}


__all__ = ["router"]
