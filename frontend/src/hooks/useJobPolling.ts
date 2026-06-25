/**
 * useJobPolling — 后台任务轮询 hook
 *
 * 用途：
 *   上传文档等异步任务提交后，希望弹窗立即关闭，但后端 job 仍在
 *   异步处理（parsing → chunking → embedding → saving）。
 *
 * 用法：
 *   1. 数据列表页（Documents / KnowledgeBases）顶层挂载 useJobPolling，
 *      注册 onTerminal / onProgress 回调
 *   2. 弹窗在提交后调用 registerJob({ job_id, status: 'pending', ... })
 *      即可把任务"移交"给页面层继续轮询，无需关心弹窗何时关闭
 *
 * 简化设计：
 *   - 模块级 jobs Map<jobId, JobInfo> 持有当前所有"已注册但未完成"的 job
 *   - 每个调用 useJobPolling 的 hook 实例独立跑一个 setInterval（轮询
 *     同一组 jobs），多实例并发由 inflight 标志避免重复请求
 *   - job 终态（done / error / cancelled）自动从 Map 移除，并触发
 *     所有 hook 实例的 onTerminal 回调
 */
import { useEffect, useRef } from "react";
import { useApi } from "@/api/client";
import type { JobInfo, JobStatus } from "@/api/types";

export type JobTerminalStatus = Extract<JobStatus, "done" | "error" | "cancelled">;

export interface JobTerminalEvent {
  jobId: string;
  status: JobTerminalStatus;
  job: JobInfo;
}

export interface JobPollingOptions {
  /** job 进入终态时回调（每个 job 触发一次） */
  onTerminal?: (e: JobTerminalEvent) => void;
  /** 每次轮询拿到进行中 job 时回调（用于 UI 顶栏进度指示） */
  onProgress?: (jobs: JobInfo[]) => void;
  /** 轮询间隔（ms），默认 1000 */
  intervalMs?: number;
}

// ============ 模块级：pending job 集合 ============

const _jobs = new Map<string, JobInfo>();

/** 注册一个 job 到全局集合；重复 id 会被覆盖。 */
export function registerJob(job: JobInfo): void {
  _jobs.set(job.job_id, job);
}

/** 主动从集合移除一个 job（一般用于已知完成，手动停止） */
export function unregisterJob(jobId: string): void {
  _jobs.delete(jobId);
}

/** 当前所有 pending + 进行中的 job（只读快照） */
export function getActiveJobs(): JobInfo[] {
  return Array.from(_jobs.values());
}

// ============ React Hook ============

/**
 * 在数据列表页顶层挂载一次。
 * 注意：本 hook 不会停止已经在跑的 job（即使卸载了），
 * 只是不再监听进度/终态。如果需要"全局永驻"，
 * 可以把 useJobPolling 放在 App.tsx 这种根位置。
 */
export function useJobPolling(options: JobPollingOptions = {}): void {
  const { client } = useApi();
  const optsRef = useRef(options);
  optsRef.current = options;
  const intervalMs = options.intervalMs ?? 1000;

  useEffect(() => {
    let cancelled = false;
    let timer: number | null = null;
    let inflight = false;

    async function tick() {
      if (cancelled || inflight) return;
      if (_jobs.size === 0) return;
      inflight = true;
      try {
        const ids = Array.from(_jobs.keys());
        const results = await Promise.allSettled(
          ids.map((id) => client.getJob(id))
        );
        if (cancelled) return;

        const progressJobs: JobInfo[] = [];
        for (let i = 0; i < ids.length; i++) {
          const id = ids[i];
          const r = results[i];
          if (r.status !== "fulfilled") continue;
          const info = r.value;
          _jobs.set(id, info);
          const status = info.status as JobStatus;
          if (status === "done" || status === "error" || status === "cancelled") {
            _jobs.delete(id);
            optsRef.current.onTerminal?.({
              jobId: id,
              status: status as JobTerminalStatus,
              job: info,
            });
          } else {
            progressJobs.push(info);
          }
        }
        if (progressJobs.length > 0) {
          optsRef.current.onProgress?.(progressJobs);
        }
      } catch (e) {
        // eslint-disable-next-line no-console
        console.warn("job polling tick error:", e);
      } finally {
        inflight = false;
      }
    }

    timer = window.setInterval(tick, intervalMs);
    // 立即跑一次
    void tick();

    return () => {
      cancelled = true;
      if (timer !== null) window.clearInterval(timer);
    };
  }, [client, intervalMs]);
}
