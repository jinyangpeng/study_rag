/**
 * Dashboard — 系统状态首页
 *
 * 布局：
 *  - 顶部 4 个 StatCard（KBs / Documents / Chunks / Active Jobs）
 *  - 中部 Activity feed（最近 10 个 job）
 *  - 右侧 Quick actions（跳转链接）
 *  - 底部 系统健康（circuit breaker 状态）
 */
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Database,
  FileText,
  Layers,
  Activity,
  ArrowRight,
  ShieldCheck,
  AlertTriangle,
  Plus,
  ListTodo,
  Search as SearchIcon,
  Gauge,
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { StatCard } from "@/components/shared/StatCard";
import { ErrorState } from "@/components/shared/ErrorState";
import { EmptyState } from "@/components/shared/EmptyState";
import { useApi } from "@/api/client";
import type { HealthDetailed, JobInfo, JobStatus } from "@/api/types";
import { formatRelativeTime } from "@/lib/utils";
import { toast } from "sonner";

const STATUS_VARIANT: Record<
  JobStatus,
  "default" | "secondary" | "success" | "warning" | "danger" | "muted"
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

export default function Dashboard() {
  const { client } = useApi();
  const [health, setHealth] = useState<HealthDetailed | null>(null);
  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [kbs, setKbs] = useState<{ document_count: number; chunk_count: number } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const [h, j, k] = await Promise.all([
        client.getHealthDetailed(),
        client.listJobs(),
        client.listKBs(),
      ]);
      setHealth(h);
      // 按更新时间倒序
      const sorted = [...j].sort(
        (a, b) =>
          new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
      );
      setJobs(sorted.slice(0, 10));
      const totalDocs = k.reduce((s, x) => s + (x.document_count ?? 0), 0);
      const totalChunks = k.reduce((s, x) => s + (x.chunk_count ?? 0), 0);
      setKbs({ document_count: totalDocs, chunk_count: totalChunks });
    } catch (e) {
      const msg = (e as Error).message;
      setError(msg);
      toast.error(`加载失败: ${msg}`);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const activeJobs = jobs.filter(
    (j) => j.status === "running" || j.status === "pending"
  );
  const kbCount = health?.kbs_total ?? 0;
  const kbEnabled = health?.kbs_enabled ?? 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-end justify-between">
        <div>
          <h2 className="text-base font-semibold">系统状态</h2>
          <p className="text-xs text-fg-muted">
            {health
              ? `服务运行正常 · ${kbEnabled} / ${kbCount} 个 KB 启用`
              : "探测服务端健康中..."}
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => void load()}>
          刷新
        </Button>
      </div>

      {error && !health ? (
        <ErrorState message={error} onRetry={() => void load()} />
      ) : (
        <>
          {/* StatCards */}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {loading ? (
              <>
                <Skeleton className="h-24" />
                <Skeleton className="h-24" />
                <Skeleton className="h-24" />
                <Skeleton className="h-24" />
              </>
            ) : (
              <>
                <StatCard
                  label="知识库"
                  value={kbCount}
                  hint={`已启用 ${kbEnabled}`}
                  icon={Database}
                />
                <StatCard
                  label="文档"
                  value={kbs?.document_count ?? 0}
                  hint="跨所有 KB"
                  icon={FileText}
                />
                <StatCard
                  label="Chunks"
                  value={kbs?.chunk_count ?? 0}
                  hint="向量化的文本块"
                  icon={Layers}
                />
                <StatCard
                  label="活跃任务"
                  value={activeJobs.length}
                  hint={`${jobs.length} 个最近任务`}
                  icon={Activity}
                  accent={activeJobs.length > 0 ? "default" : "default"}
                />
              </>
            )}
          </div>

          {/* Activity + Quick actions */}
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
            {/* Activity feed */}
            <Card className="lg:col-span-2">
              <CardHeader className="flex flex-row items-center justify-between space-y-0">
                <CardTitle className="flex items-center gap-2">
                  <Activity className="size-3.5 text-fg-muted" />
                  最近活动
                </CardTitle>
                <Button asChild variant="ghost" size="sm">
                  <Link to="/jobs" className="flex items-center gap-1 text-xs">
                    全部任务
                    <ArrowRight className="size-3" />
                  </Link>
                </Button>
              </CardHeader>
              <CardContent className="pt-0">
                {loading ? (
                  <div className="space-y-2">
                    {Array.from({ length: 5 }).map((_, i) => (
                      <Skeleton key={i} className="h-8" />
                    ))}
                  </div>
                ) : jobs.length === 0 ? (
                  <EmptyState
                    title="暂无活动"
                    description="上传文档后会显示在这里"
                    icon={ListTodo}
                  />
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>状态</TableHead>
                        <TableHead>任务</TableHead>
                        <TableHead>KB / 文档</TableHead>
                        <TableHead className="w-32">进度</TableHead>
                        <TableHead className="text-right">更新</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {jobs.map((j) => (
                        <TableRow key={j.job_id}>
                          <TableCell>
                            <Badge variant={STATUS_VARIANT[j.status]}>
                              {STATUS_LABEL[j.status]}
                            </Badge>
                          </TableCell>
                          <TableCell className="font-mono text-[11px] text-fg-secondary">
                            {j.job_id.slice(0, 12)}
                          </TableCell>
                          <TableCell className="text-fg-secondary">
                            {j.kb_id ?? "—"}
                            {j.doc_id ? ` / ${j.doc_id}` : ""}
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center gap-2">
                              <Progress
                                value={Math.round((j.progress ?? 0) * 100)}
                                className="h-1"
                              />
                              <span className="w-8 text-right text-[10px] text-fg-muted">
                                {Math.round((j.progress ?? 0) * 100)}%
                              </span>
                            </div>
                          </TableCell>
                          <TableCell className="text-right text-[10px] text-fg-muted">
                            {formatRelativeTime(j.updated_at)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>

            {/* Quick actions */}
            <div className="space-y-3">
              <Card>
                <CardHeader>
                  <CardTitle>快速操作</CardTitle>
                </CardHeader>
                <CardContent className="space-y-1 pt-0">
                  <QuickAction
                    to="/kbs"
                    icon={Plus}
                    label="新建知识库"
                    hint="配置 embedding + collection"
                  />
                  <Separator />
                  <QuickAction
                    to="/kbs"
                    icon={Database}
                    label="管理知识库"
                    hint="查看 / 编辑 / 删除"
                  />
                  <Separator />
                  <QuickAction
                    to="/search"
                    icon={SearchIcon}
                    label="检索测试"
                    hint="快速验证 KB 召回质量"
                  />
                  <Separator />
                  <QuickAction
                    to="/metrics"
                    icon={Gauge}
                    label="查看指标"
                    hint="Prometheus 原始文本"
                  />
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Health / Circuit Breakers */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <ShieldCheck className="size-3.5 text-fg-muted" />
                系统健康
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              {loading ? (
                <div className="space-y-2">
                  <Skeleton className="h-6" />
                  <Skeleton className="h-6" />
                </div>
              ) : !health ? (
                <EmptyState title="无健康数据" icon={ShieldCheck} />
              ) : (
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
                  <HealthItem
                    label="Embedders"
                    value={health.embedders}
                    ok
                  />
                  <HealthItem
                    label="Rerankers"
                    value={health.rerankers}
                    ok
                  />
                  <HealthItem
                    label="Admin 限流"
                    value={`${health.ratelimit.admin.tracked_keys} keys`}
                    ok
                  />
                  <HealthItem
                    label="Search 限流"
                    value={`${health.ratelimit.search.tracked_keys} keys`}
                    ok
                  />
                  {Object.entries(health.circuit_breakers ?? {}).map(
                    ([name, cb]) => (
                      <HealthItem
                        key={name}
                        label={`CB: ${name}`}
                        value={`${cb.state} (${cb.failure_count})`}
                        ok={cb.state === "closed"}
                        warn={cb.state !== "closed"}
                      />
                    )
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

function QuickAction({
  to,
  icon: Icon,
  label,
  hint,
}: {
  to: string;
  icon: typeof Plus;
  label: string;
  hint: string;
}) {
  return (
    <Link
      to={to}
      className="group flex items-center justify-between rounded px-2 py-1.5 transition-colors hover:bg-bg-tertiary"
    >
      <div className="flex items-center gap-2.5">
        <Icon className="size-3.5 text-fg-muted" />
        <div>
          <div className="text-xs font-medium text-fg">{label}</div>
          <div className="text-[10px] text-fg-muted">{hint}</div>
        </div>
      </div>
      <ArrowRight className="size-3 text-fg-muted opacity-0 transition-opacity group-hover:opacity-100" />
    </Link>
  );
}

function HealthItem({
  label,
  value,
  warn,
}: {
  label: string;
  value: string | number;
  ok?: boolean;
  warn?: boolean;
}) {
  return (
    <div className="flex items-center justify-between rounded border border-border bg-bg-tertiary px-3 py-2">
      <div className="flex items-center gap-2">
        {warn ? (
          <AlertTriangle className="size-3 text-warning" />
        ) : (
          <ShieldCheck className="size-3 text-success" />
        )}
        <span className="text-xs text-fg-secondary">{label}</span>
      </div>
      <span className="text-xs font-medium text-fg">{value}</span>
    </div>
  );
}
