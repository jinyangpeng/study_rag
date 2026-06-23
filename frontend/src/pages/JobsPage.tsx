/**
 * JobsPage：异步任务总览页。
 *
 * 功能：
 *   - 列出所有 job（按 created_at 倒序）
 *   - 可按 KB / 状态过滤
 *   - 自动轮询「运行中 / 等待中」的任务（1s 一次）
 *   - 显示进度条 / 阶段 / 错误信息
 *   - 支持取消运行中的任务
 *
 * 路径：/jobs
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Table,
  Tabs,
  Button,
  Progress,
  Space,
  Empty,
  Tag,
  Select,
  Card,
  Typography,
  App as AntdApp,
} from "antd";
import type { TableColumnsType } from "antd";
import { ReloadOutlined, SyncOutlined } from "@ant-design/icons";
import { useApi } from "../api/client";
import type { JobInfo, JobStatus, KnowledgeBaseSummary } from "../api/types";
import JobProgressRow from "../components/JobProgressRow";

const POLL_INTERVAL_MS = 1000;

type StatusFilter = "all" | "active" | "done" | "error" | "cancelled";

const STATUS_TABS: { key: StatusFilter; label: string }[] = [
  { key: "all", label: "全部" },
  { key: "active", label: "进行中" },
  { key: "done", label: "已完成" },
  { key: "error", label: "失败" },
  { key: "cancelled", label: "已取消" },
];

function statusMatches(s: JobStatus, f: StatusFilter): boolean {
  if (f === "all") return true;
  if (f === "active") return s === "pending" || s === "running";
  if (f === "done") return s === "done";
  if (f === "error") return s === "error";
  if (f === "cancelled") return s === "cancelled";
  return true;
}

export default function JobsPage() {
  const { client } = useApi();
  const { message } = AntdApp.useApp();
  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [kbs, setKBs] = useState<KnowledgeBaseSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState<StatusFilter>("all");
  const [kbFilter, setKBFilter] = useState<string | undefined>(undefined);
  const pollTimerRef = useRef<number | null>(null);

  const loadKBs = useCallback(async () => {
    try {
      const list = await client.listKBs();
      setKBs(list);
    } catch (e) {
      // 拉 KB 失败不影响主流程
      // eslint-disable-next-line no-console
      console.warn("loadKBs failed:", e);
    }
  }, [client]);

  const loadJobs = useCallback(async () => {
    setLoading(true);
    try {
      const list = await client.listJobs(kbFilter);
      setJobs(list);
    } catch (e) {
      message.error((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, [client, kbFilter, message]);

  // mount: 拉一次 + 启轮询
  useEffect(() => {
    void loadKBs();
    void loadJobs();
  }, [loadKBs, loadJobs]);

  // 轮询：有进行中任务时每 1s 拉一次；全停时降频到 5s
  useEffect(() => {
    function tick() {
      void loadJobs();
    }
    pollTimerRef.current = window.setInterval(() => {
      const hasActive = jobs.some(
        (j) => j.status === "pending" || j.status === "running"
      );
      if (hasActive) {
        tick();
      }
    }, POLL_INTERVAL_MS);
    return () => {
      if (pollTimerRef.current !== null) {
        window.clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [loadJobs, jobs]);

  // 单独启一个 5s 兜底刷新（避免"全 done"后永远不刷新了）
  useEffect(() => {
    const id = window.setInterval(() => {
      void loadJobs();
    }, 5000);
    return () => window.clearInterval(id);
  }, [loadJobs]);

  const handleCancel = useCallback(
    async (jobId: string) => {
      try {
        const r = await client.cancelJob(jobId);
        if (r.status === "cancelling") {
          message.info("已请求取消");
        } else if (r.status === "not_cancellable") {
          message.warning("该任务已结束，无法取消");
        } else {
          message.warning("任务不存在");
        }
        void loadJobs();
      } catch (e) {
        message.error((e as Error).message);
      }
    },
    [client, message, loadJobs]
  );

  const filtered = useMemo(
    () => jobs.filter((j) => statusMatches(j.status, filter)),
    [jobs, filter]
  );

  const counts = useMemo(() => {
    const c: Record<StatusFilter, number> = {
      all: jobs.length,
      active: 0,
      done: 0,
      error: 0,
      cancelled: 0,
    };
    for (const j of jobs) {
      if (j.status === "pending" || j.status === "running") c.active++;
      else if (j.status === "done") c.done++;
      else if (j.status === "error") c.error++;
      else if (j.status === "cancelled") c.cancelled++;
    }
    return c;
  }, [jobs]);

  const columns: TableColumnsType<JobInfo> = [
    {
      title: "进度",
      key: "progress",
      width: 360,
      render: (_v, job) => <JobProgressRow job={job} onCancel={handleCancel} />,
    },
    {
      title: "类型",
      dataIndex: "type",
      key: "type",
      width: 100,
      render: (t: string) => <Tag>{t}</Tag>,
    },
    {
      title: "进度%",
      key: "pct",
      width: 100,
      render: (_v, job) => {
        const pct = Math.round((job.progress ?? 0) * 100);
        return (
          <Progress
            percent={pct}
            size="small"
            showInfo
            status={
              job.status === "done"
                ? "success"
                : job.status === "error"
                  ? "exception"
                  : "active"
            }
          />
        );
      },
    },
    {
      title: "创建时间",
      key: "created_at",
      width: 180,
      render: (_v, job) => (
        <Typography.Text type="secondary" style={{ fontSize: 12 }}>
          {new Date(job.created_at).toLocaleString()}
        </Typography.Text>
      ),
    },
    {
      title: "耗时",
      key: "duration",
      width: 100,
      render: (_v, job) => {
        const start = new Date(job.created_at).getTime();
        const end =
          job.status === "done" ||
          job.status === "error" ||
          job.status === "cancelled"
            ? new Date(job.updated_at).getTime()
            : Date.now();
        const ms = end - start;
        if (ms < 1000) return `${ms}ms`;
        if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
        return `${(ms / 60_000).toFixed(1)}m`;
      },
    },
  ];

  return (
    <div>
      <Card
        title={
          <Space>
            <SyncOutlined spin={loading} />
            <span>异步任务</span>
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              共 {jobs.length} 个，进行中 {counts.active}
            </Typography.Text>
          </Space>
        }
        extra={
          <Space>
            <Select
              allowClear
              placeholder="按 KB 过滤"
              style={{ width: 200 }}
              value={kbFilter}
              onChange={(v) => setKBFilter(v)}
              options={kbs.map((k) => ({ value: k.kb_id, label: k.name }))}
            />
            <Button
              icon={<ReloadOutlined />}
              onClick={() => void loadJobs()}
              loading={loading}
            >
              刷新
            </Button>
          </Space>
        }
      >
        <Tabs
          activeKey={filter}
          onChange={(k) => setFilter(k as StatusFilter)}
          items={STATUS_TABS.map((t) => ({
            key: t.key,
            label: (
              <Space size={4}>
                {t.label}
                <Tag>{counts[t.key]}</Tag>
              </Space>
            ),
          }))}
        />
        <Table<JobInfo>
          rowKey="job_id"
          dataSource={filtered}
          columns={columns}
          loading={loading}
          pagination={{ pageSize: 20, showSizeChanger: false }}
          size="middle"
          locale={{
            emptyText: (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description={
                  jobs.length === 0 ? "暂无任务" : "当前过滤条件下无任务"
                }
              />
            ),
          }}
        />
      </Card>
    </div>
  );
}
