/**
 * AddDocumentDrawer：替代原 Modal 的文档添加抽屉。
 *
 * 三种模式：
 *   1. 文本输入：手动粘贴大段文本
 *   2. 文件上传：拖拽或选 txt/md/html/pdf/docx，自动解析
 *   3. 选 parser + 实时预览切块（提交前看到每块内容/大小）
 *
 * 提交流程（Phase 7 异步化）：
 *   - 文本：调 preview → 用户确认 → 同步 addDocumentChunked（保持同步，短文档够用）
 *   - 文件：调 upload → 拿到 job_id → 后台跑
 *     → 轮询 /admin/jobs/{id} 显示进度
 *     → 完成后调 onSuccess + 关闭
 */

import { useEffect, useRef, useState } from "react";
import {
  Drawer,
  Tabs,
  Form,
  Input,
  Select,
  Button,
  Upload,
  Alert,
  Space,
  Tag,
  Typography,
  App as AntdApp,
  Spin,
  Progress,
} from "antd";
import type { UploadProps } from "antd";
import {
  InboxOutlined,
  FileTextOutlined,
  CodeOutlined,
  EyeOutlined,
  LoadingOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  StopOutlined,
} from "@ant-design/icons";
import { useApi } from "../api/client";
import type {
  JobInfo,
  JobStatus,
  JobStage,
  ParserSpec,
  ChunkPreviewItem,
  UploadDocumentResponse,
} from "../api/types";
import ChunkPreviewPanel from "./ChunkPreviewPanel";

const { TextArea } = Input;
const { Dragger } = Upload;

interface Props {
  open: boolean;
  kbId: string;
  onCancel: () => void;
  onSuccess: () => void;
}

interface FormValues {
  doc_id: string;
  title: string;
  content?: string;
  source?: string;
  parser?: string;
}

type TabKey = "text" | "file";

// 阶段中英文映射（UI 显示）
const STAGE_LABEL: Record<JobStage, string> = {
  queued: "排队中",
  parsing: "解析文件",
  chunking: "切分文本",
  embedding: "生成向量",
  saving: "写入数据库",
  done: "完成",
};

const POLL_INTERVAL_MS = 1000;

export default function AddDocumentDrawer({
  open,
  kbId,
  onCancel,
  onSuccess,
}: Props) {
  const { client } = useApi();
  const { message } = AntdApp.useApp();
  const [form] = Form.useForm<FormValues>();
  const [parsers, setParsers] = useState<ParserSpec[]>([]);
  const [tab, setTab] = useState<TabKey>("text");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<ChunkPreviewItem[] | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  // Phase 7: 异步上传 + 轮询
  const [job, setJob] = useState<JobInfo | null>(null);
  const pollTimerRef = useRef<number | null>(null);

  // 拉 parser 列表
  useEffect(() => {
    if (!open) return;
    (async () => {
      try {
        const ps = await client.listParsers();
        setParsers(ps);
        const defaultName =
          ps.find((p) => p.name === "sentence_512")?.name ??
          ps[0]?.name ??
          undefined;
        if (defaultName) {
          form.setFieldsValue({ parser: defaultName });
        }
      } catch (e) {
        message.error((e as Error).message);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // 关闭时清状态
  useEffect(() => {
    if (!open) {
      form.resetFields();
      setFile(null);
      setPreview(null);
      stopPolling();
      setJob(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, form]);

  // 卸载时清理 timer
  useEffect(() => {
    return () => stopPolling();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function stopPolling() {
    if (pollTimerRef.current !== null) {
      window.clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }

  function startPolling(jobId: string) {
    stopPolling();
    pollTimerRef.current = window.setInterval(async () => {
      try {
        const info = await client.getJob(jobId);
        setJob(info);
        const status: JobStatus = info.status;
        if (
          status === "done" ||
          status === "error" ||
          status === "cancelled"
        ) {
          stopPolling();
          if (status === "done") {
            message.success(`${info.doc_id ?? "文档"} 上传完成`);
            onSuccess();
            onCancel();
          } else if (status === "error") {
            message.error(`上传失败: ${info.error ?? "未知错误"}`);
          } else if (status === "cancelled") {
            message.warning("任务已取消");
          }
        }
      } catch (e) {
        // 轮询出错 → 静默重试（不打扰用户）
        // eslint-disable-next-line no-console
        console.warn("job poll failed:", e);
      }
    }, POLL_INTERVAL_MS);
  }

  async function handleCancelJob() {
    if (!job) return;
    try {
      await client.cancelJob(job.job_id);
      message.info("已请求取消");
    } catch (e) {
      message.error((e as Error).message);
    }
  }

  const onPreview = async () => {
    try {
      const v = await form.validateFields(["content", "parser"]);
      if (!v.content) {
        message.warning("请先输入正文");
        return;
      }
      setPreviewLoading(true);
      const r = await client.previewChunk(
        kbId,
        v.content,
        v.parser as string,
        v.title || "preview"
      );
      setPreview(r.chunks);
    } catch (e) {
      if (!(e as { errorFields?: unknown }).errorFields) {
        message.error((e as Error).message);
      }
    } finally {
      setPreviewLoading(false);
    }
  };

  const onSubmit = async () => {
    try {
      const v = await form.validateFields();
      setSubmitting(true);
      if (tab === "text") {
        if (!v.content) {
          message.warning("请输入正文");
          return;
        }
        const parserSpec = parsers.find((p) => p.name === v.parser);
        await client.addDocumentChunked({
          kb_id: kbId,
          doc_id: v.doc_id,
          title: v.title,
          content: v.content,
          source: v.source || null,
          metadata: {},
          chunk_size: parserSpec?.chunk_size ?? 512,
          chunk_overlap: parserSpec?.chunk_overlap ?? 50,
        });
        message.success(`文档 ${v.doc_id} 添加成功`);
        onSuccess();
        onCancel();
      } else {
        if (!file) {
          message.error("请先选择文件");
          return;
        }
        const fd = new FormData();
        fd.append("file", file);
        fd.append("doc_id", v.doc_id);
        fd.append("title", v.title);
        fd.append("parser", v.parser ?? "");
        if (v.source) fd.append("source", v.source);
        // Phase 7: 异步上传，立即拿到 job_id
        const r: UploadDocumentResponse = await client.uploadDocument(
          kbId,
          fd
        );
        // 显示初始 job（pending 状态）
        setJob({
          job_id: r.job_id,
          type: "upload_doc",
          status: r.status as JobStatus,
          stage: "queued",
          current: 0,
          total: 0,
          progress: 0,
          message: "已提交，等待处理",
          error: null,
          kb_id: r.kb_id,
          doc_id: r.doc_id,
          filename: r.format,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        });
        // 启动轮询
        startPolling(r.job_id);
      }
    } catch (e) {
      if (!(e as { errorFields?: unknown }).errorFields) {
        message.error((e as Error).message);
      }
    } finally {
      setSubmitting(false);
    }
  };

  const draggerProps: UploadProps = {
    beforeUpload: (f) => {
      setFile(f);
      // 自动填 title（如果空）
      if (!form.getFieldValue("title")) {
        const nameOnly = f.name.replace(/\.[^.]+$/, "");
        form.setFieldValue("title", nameOnly);
      }
      return false; // 阻止自动 upload
    },
    onRemove: () => {
      setFile(null);
    },
    maxCount: 1,
    accept: ".txt,.md,.markdown,.html,.htm,.pdf,.docx",
    showUploadList: false,
  };

  // 渲染 job 进度面板
  const renderJobPanel = () => {
    if (!job) return null;
    const pct = Math.round((job.progress ?? 0) * 100);
    const status = job.status;
    const stageLabel = STAGE_LABEL[job.stage] ?? job.stage;
    let progressStatus: "active" | "success" | "exception" | "normal" =
      "active";
    let icon: React.ReactNode = <LoadingOutlined spin />;
    if (status === "done") {
      progressStatus = "success";
      icon = <CheckCircleOutlined style={{ color: "#52c41a" }} />;
    } else if (status === "error") {
      progressStatus = "exception";
      icon = <CloseCircleOutlined style={{ color: "#ff4d4f" }} />;
    } else if (status === "cancelled") {
      progressStatus = "normal";
      icon = <StopOutlined style={{ color: "#faad14" }} />;
    }

    return (
      <div
        style={{
          padding: 16,
          background: "#fafafa",
          borderRadius: 6,
          marginTop: 16,
        }}
      >
        <Space style={{ marginBottom: 8 }}>
          {icon}
          <Typography.Text strong>异步上传进度</Typography.Text>
          <Tag color="blue">{stageLabel}</Tag>
        </Space>
        <Progress
          percent={pct}
          status={progressStatus}
          strokeWidth={12}
          format={(p) => `${p}%`}
        />
        <div style={{ marginTop: 8, color: "#666", fontSize: 13 }}>
          {job.message || stageLabel}
          {job.current > 0 && job.total > 0 && (
            <Tag style={{ marginLeft: 8 }}>
              {job.current} / {job.total}
            </Tag>
          )}
        </div>
        {status === "error" && job.error && (
          <Alert
            type="error"
            showIcon
            style={{ marginTop: 12 }}
            message={job.error}
          />
        )}
        {(status === "pending" || status === "running") && (
          <Button
            danger
            size="small"
            style={{ marginTop: 12 }}
            icon={<StopOutlined />}
            onClick={handleCancelJob}
          >
            取消任务
          </Button>
        )}
      </div>
    );
  };

  return (
    <Drawer
      title="添加文档"
      open={open}
      onClose={onCancel}
      width={720}
      destroyOnClose
      extra={
        <Space>
          <Button
            onClick={onPreview}
            loading={previewLoading}
            icon={<EyeOutlined />}
            disabled={!!job && (job.status === "running" || job.status === "pending")}
          >
            预览分块
          </Button>
          <Button
            type="primary"
            onClick={onSubmit}
            loading={submitting}
            disabled={!!job && (job.status === "running" || job.status === "pending")}
          >
            添加
          </Button>
        </Space>
      }
    >
      <Form<FormValues> form={form} layout="vertical">
        <Tabs
          activeKey={tab}
          onChange={(k) => setTab(k as TabKey)}
          items={[
            {
              key: "text",
              label: (
                <span>
                  <FileTextOutlined /> 文本输入
                </span>
              ),
              children: (
                <Form.Item
                  label="正文"
                  name="content"
                  rules={[
                    { required: tab === "text", message: "请输入正文" },
                  ]}
                >
                  <TextArea
                    rows={8}
                    placeholder="粘贴文本内容（100-10000 字）"
                    showCount
                  />
                </Form.Item>
              ),
            },
            {
              key: "file",
              label: (
                <span>
                  <InboxOutlined /> 文件上传
                </span>
              ),
              children: (
                <Form.Item label="文件" required>
                  <Dragger {...draggerProps}>
                    <p className="ant-upload-drag-icon">
                      <InboxOutlined />
                    </p>
                    <p className="ant-upload-text">点击或拖拽文件到此处</p>
                    <p className="ant-upload-hint">
                      支持 txt / md / html / pdf / docx，单文件最大 50MB
                    </p>
                  </Dragger>
                  {file && (
                    <Alert
                      type="info"
                      showIcon
                      style={{ marginTop: 8 }}
                      message={
                        <Space>
                          <CodeOutlined />
                          {file.name}
                          <Tag>{(file.size / 1024).toFixed(1)} KB</Tag>
                        </Space>
                      }
                    />
                  )}
                </Form.Item>
              ),
            },
          ]}
        />

        <Form.Item
          label="doc_id"
          name="doc_id"
          rules={[{ required: true, message: "请输入 doc_id" }]}
        >
          <Input placeholder="KB 内唯一，如 react_perf_001" />
        </Form.Item>
        <Form.Item
          label="标题"
          name="title"
          rules={[{ required: true, message: "请输入标题" }]}
        >
          <Input placeholder="React 性能优化指南" />
        </Form.Item>
        <Form.Item label="source" name="source">
          <Input placeholder="可选：来源标识" />
        </Form.Item>
        <Form.Item
          label={
            <Space>
              切块策略
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                （configs/llamaindex.yaml 命名实体）
              </Typography.Text>
            </Space>
          }
          name="parser"
          rules={[{ required: true, message: "请选择切块策略" }]}
        >
          <Select
            options={parsers.map((p) => ({
              value: p.name,
              label: (
                <Space>
                  <Tag color="blue">{p.strategy}</Tag>
                  <span>{p.name}</span>
                  <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                    size={p.chunk_size}, overlap={p.chunk_overlap}
                  </Typography.Text>
                </Space>
              ),
            }))}
          />
        </Form.Item>
      </Form>

      {previewLoading && (
        <div style={{ textAlign: "center", padding: 24 }}>
          <Spin />
        </div>
      )}
      {preview && !previewLoading && <ChunkPreviewPanel chunks={preview} />}

      {renderJobPanel()}
    </Drawer>
  );
}
