/**
 * JobProgressRow：单个 job 的进度行（表格行内用）。
 *
 * 显示：
 *   - 状态图标 + 状态 Tag
 *   - 阶段 Tag（如 "切分文本"）
 *   - Progress 条 + 百分比
 *   - 当前 message
 *   - error（如果有）
 *   - current / total
 *   - 取消按钮（pending / running 时显示）
 *
 * 取消由 JobsPage 注入的 onCancel 处理（避免这里直接耦合 client）。
 */

import { Space, Progress, Tag, Button, Alert, Typography, Tooltip } from "antd";
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  StopOutlined,
  LoadingOutlined,
  ClockCircleOutlined,
} from "@ant-design/icons";
import type { JobInfo, JobStatus, JobStage } from "../api/types";

const STAGE_LABEL: Record<JobStage, string> = {
  queued: "排队中",
  parsing: "解析文件",
  chunking: "切分文本",
  embedding: "生成向量",
  saving: "写入数据库",
  done: "完成",
};

const STATUS_COLOR: Record<JobStatus, string> = {
  pending: "default",
  running: "processing",
  done: "success",
  error: "error",
  cancelled: "warning",
};

const STATUS_LABEL: Record<JobStatus, string> = {
  pending: "等待",
  running: "运行中",
  done: "完成",
  error: "失败",
  cancelled: "已取消",
};

interface Props {
  job: JobInfo;
  onCancel?: (jobId: string) => void;
}

export default function JobProgressRow({ job, onCancel }: Props) {
  const pct = Math.round((job.progress ?? 0) * 100);
  const stageLabel = STAGE_LABEL[job.stage] ?? job.stage;

  let progressStatus: "active" | "success" | "exception" | "normal" = "active";
  let icon: React.ReactNode = <LoadingOutlined spin />;
  if (job.status === "done") {
    progressStatus = "success";
    icon = <CheckCircleOutlined style={{ color: "#52c41a" }} />;
  } else if (job.status === "error") {
    progressStatus = "exception";
    icon = <CloseCircleOutlined style={{ color: "#ff4d4f" }} />;
  } else if (job.status === "cancelled") {
    progressStatus = "normal";
    icon = <StopOutlined style={{ color: "#faad14" }} />;
  } else if (job.status === "pending") {
    progressStatus = "normal";
    icon = <ClockCircleOutlined style={{ color: "#999" }} />;
  }

  const cancellable = job.status === "pending" || job.status === "running";

  return (
    <div style={{ width: "100%" }}>
      <Space style={{ marginBottom: 6 }} wrap>
        {icon}
        <Tag color={STATUS_COLOR[job.status]}>{STATUS_LABEL[job.status]}</Tag>
        <Tag color="blue">{stageLabel}</Tag>
        {job.kb_id && <Tag>KB: {job.kb_id}</Tag>}
        {job.doc_id && <Tag>doc: {job.doc_id}</Tag>}
        {job.filename && (
          <Tooltip title={job.filename}>
            <Tag>{job.filename}</Tag>
          </Tooltip>
        )}
        {cancellable && onCancel && (
          <Button
            size="small"
            danger
            icon={<StopOutlined />}
            onClick={() => onCancel(job.job_id)}
          >
            取消
          </Button>
        )}
      </Space>
      <Progress
        percent={pct}
        status={progressStatus}
        strokeWidth={8}
        size="small"
      />
      <div
        style={{
          marginTop: 4,
          color: "#666",
          fontSize: 12,
          display: "flex",
          gap: 12,
        }}
      >
        <Typography.Text type="secondary" style={{ fontSize: 12 }}>
          {job.message || stageLabel}
        </Typography.Text>
        {job.current > 0 && job.total > 0 && (
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            {job.current} / {job.total}
          </Typography.Text>
        )}
      </div>
      {job.status === "error" && job.error && (
        <Alert
          type="error"
          showIcon
          style={{ marginTop: 8 }}
          message={job.error}
        />
      )}
    </div>
  );
}
