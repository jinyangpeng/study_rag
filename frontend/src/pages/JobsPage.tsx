/**
 * JobsPage — 异步任务管理页
 *
 * 布局：
 *  - 4 个 Tabs：运行中（pending+running）/ 已完成 / 失败 / 已取消
 *  - 每个 Tab 一个高密度 Table
 *  - 运行中 / 待处理 的行有实时 Progress + 取消按钮
 *  - 每秒轮询一次；空闲 Tab 暂缓轮询
 */
import { useEffect, useRef, useState, useCallback } from "react";
import {
  Activity,
  CheckCircle2,
  XCircle,
  StopCircle,
  RotateCw,
  StopCircle as StopIcon,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { ErrorState } from "@/components/shared/ErrorState";
import { useApi } from "@/api/client";
import type { JobInfo, JobStage, JobStatus } from "@/api/types";
import { formatRelativeTime } from "@/lib/utils";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

const STAGE_LABEL: Record<JobStage, string> = {
  queued: "排队中",
  parsing: "解析文件",
  chunking: "切分文本",
  embedding: "生成向量",
  saving: "写入数据库",
  done: "完成",
};

const STATUS_VARIANT: Record<
  JobStatus,
  "muted" | "default" | "success" | "danger" | "warning"
> = {
  pending: "muted",
  running: "default",
  done: "success",
  error: "danger",
  cancelled: "warning",
};

const STATUS_LABEL: Record<JobStatus, string> = {
  pending: "等待",
  running: "运行中",
  done: "完成",
  error: "失败",
  cancelled: "已取消",
};

const POLL_INTERVAL_MS = 1500;

const TAB_DEFS = [
  { key: "active", label: "运行中", statuses: ["pending", "running"] as JobStatus[] },
  { key: "done", label: "已完成", statuses: ["done"] as JobStatus[] },
  { key: "error", label: "失败", statuses: ["error"] as JobStatus[] },
  { key: "cancelled", label: "已取消", statuses: ["cancelled"] as JobStatus[] },
] as const;

type TabKey = (typeof TAB_DEFS)[number]["key"];

export default function JobsPage() {
  const { client } = useApi();
  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<TabKey>("active");
  const pollTimerRef = useRef<number | null>(null);

  const load = useCallback(
    async (silent = false) => {
      if (!silent) setLoading(true);
      setError(null);
      try {
        const r = await client.listJobs();
        // 按 updated_at 倒序
        const sorted = [...r].sort(
          (a, b) =>
            new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
        );
        setJobs(sorted);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        if (!silent) setLoading(false);
      }
    },
    [client]
  );

  // 初次加载
  useEffect(() => {
    void load();
  }, [load]);

  // 实时轮询：只有运行中 tab 才高频拉
  useEffect(() => {
    if (tab !== "active") {
      if (pollTimerRef.current !== null) {
        window.clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
      return;
    }
    pollTimerRef.current = window.setInterval(() => {
      void load(true);
    }, POLL_INTERVAL_MS);
    return () => {
      if (pollTimerRef.current !== null) {
        window.clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [tab, load]);

  async function handleCancel(jobId: string) {
    try {
      await client.cancelJob(jobId);
      toast.info("已请求取消");
      void load(true);
    } catch (e) {
      toast.error((e as Error).message);
    }
  }

  const counts = TAB_DEFS.reduce<Record<TabKey, number>>(
    (acc, t) => {
      acc[t.key] = jobs.filter((j) => t.statuses.includes(j.status)).length;
      return acc;
    },
    { active: 0, done: 0, error: 0, cancelled: 0 }
  );

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-end justify-between">
        <div>
          <h2 className="text-base font-semibold">异步任务</h2>
          <p className="text-xs text-fg-muted">
            实时显示文档处理、批量导入等后台任务
            {tab === "active" && jobs.length > 0 && (
              <span className="ml-1 inline-flex items-center gap-1">
                <span className="inline-block size-1.5 animate-pulse-dot rounded-full bg-accent" />
                自动轮询中
              </span>
            )}
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => void load()}
          disabled={loading}
        >
          <RotateCw className={cn("size-3.5", loading && "animate-spin")} />
          刷新
        </Button>
      </div>

      {error && !loading ? (
        <ErrorState message={error} onRetry={() => void load()} />
      ) : (
        <Card>
          <CardContent className="p-0">
            <Tabs value={tab} onValueChange={(v) => setTab(v as TabKey)}>
              <div className="flex items-center justify-between border-b border-border px-3 pt-3">
                <TabsList>
                  {TAB_DEFS.map((t) => (
                    <TabsTrigger key={t.key} value={t.key}>
                      {t.label}
                      <Badge variant="muted" className="ml-1.5 font-mono">
                        {counts[t.key]}
                      </Badge>
                    </TabsTrigger>
                  ))}
                </TabsList>
              </div>

              {TAB_DEFS.map((t) => {
                const filtered = jobs.filter((j) =>
                  t.statuses.includes(j.status)
                );
                return (
                  <TabsContent key={t.key} value={t.key} className="m-0">
                    {loading ? (
                      <div className="space-y-2 p-4">
                        {Array.from({ length: 4 }).map((_, i) => (
                          <Skeleton key={i} className="h-12" />
                        ))}
                      </div>
                    ) : filtered.length === 0 ? (
                      <EmptyState
                        title={`暂无「${t.label}」任务`}
                        icon={
                          t.key === "active"
                            ? Activity
                            : t.key === "done"
                            ? CheckCircle2
                            : t.key === "error"
                            ? XCircle
                            : StopCircle
                        }
                      />
                    ) : (
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>状态</TableHead>
                            <TableHead>阶段</TableHead>
                            <TableHead>任务 ID</TableHead>
                            <TableHead>KB / 文档</TableHead>
                            <TableHead className="w-48">进度</TableHead>
                            <TableHead>消息</TableHead>
                            <TableHead className="text-right">更新</TableHead>
                            <TableHead className="w-20 text-right">操作</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {filtered.map((j) => (
                            <JobRow
                              key={j.job_id}
                              job={j}
                              onCancel={handleCancel}
                            />
                          ))}
                        </TableBody>
                      </Table>
                    )}
                  </TabsContent>
                );
              })}
            </Tabs>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function JobRow({
  job,
  onCancel,
}: {
  job: JobInfo;
  onCancel: (jobId: string) => void;
}) {
  const pct = Math.round((job.progress ?? 0) * 100);
  const cancellable = job.status === "pending" || job.status === "running";
  const stageLabel = STAGE_LABEL[job.stage] ?? job.stage;

  return (
    <TableRow>
      <TableCell>
        <Badge variant={STATUS_VARIANT[job.status]}>
          {STATUS_LABEL[job.status]}
        </Badge>
      </TableCell>
      <TableCell>
        <Badge variant="outline" className="font-normal">
          {stageLabel}
        </Badge>
      </TableCell>
      <TableCell className="font-mono text-[10px] text-fg-secondary">
        {job.job_id.slice(0, 16)}
      </TableCell>
      <TableCell className="font-mono text-[10px] text-fg-secondary">
        {job.kb_id ? (
          <span>
            {job.kb_id}
            {job.doc_id ? ` / ${job.doc_id}` : ""}
          </span>
        ) : job.filename ? (
          <span>{job.filename}</span>
        ) : (
          <span className="text-fg-muted">—</span>
        )}
      </TableCell>
      <TableCell>
        <div className="flex items-center gap-2">
          <Progress value={pct} className="h-1.5" />
          <span className="w-12 text-right font-mono text-[10px] text-fg-muted">
            {pct}%
            {job.current > 0 && job.total > 0 && (
              <span className="ml-1">
                {job.current}/{job.total}
              </span>
            )}
          </span>
        </div>
      </TableCell>
      <TableCell className="max-w-[16rem] truncate text-[10px] text-fg-secondary">
        {job.message || stageLabel}
      </TableCell>
      <TableCell className="text-right text-[10px] text-fg-muted">
        {formatRelativeTime(job.updated_at)}
      </TableCell>
      <TableCell className="text-right">
        {cancellable ? (
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-warning hover:text-warning"
            onClick={() => onCancel(job.job_id)}
            title="取消"
          >
            <StopIcon className="size-3" />
          </Button>
        ) : (
          <span className="text-fg-muted">—</span>
        )}
      </TableCell>
    </TableRow>
  );
}
