/**
 * AddDocumentDrawer：替代原 Modal 的文档添加抽屉。
 *
 * 三种模式：
 *   1. 文本输入：手动粘贴大段文本
 *   2. 文件上传：拖拽或选 txt/md/html/pdf/docx，自动解析
 *   3. 选 parser + 实时预览切块（提交前看到每块内容/大小）
 *
 * 提交流程：
 *   - 文本：调 preview → 用户确认 → 调 chunked 接口
 *   - 文件：调 upload 接口（preview 在后端 upload 路径上做）
 */

import { useEffect, useState } from "react";
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
} from "antd";
import type { UploadProps } from "antd";
import {
  InboxOutlined,
  FileTextOutlined,
  CodeOutlined,
  EyeOutlined,
} from "@ant-design/icons";
import { useApi } from "../api/client";
import type {
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
    }
  }, [open, form]);

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
        const r: UploadDocumentResponse = await client.uploadDocument(
          kbId,
          fd
        );
        message.success(
          `${r.doc_id} 上传成功（${r.format}, ${r.chunks} chunks, ${(
            r.size_bytes / 1024
          ).toFixed(1)} KB）`
        );
      }
      onSuccess();
      onCancel();
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
          >
            预览分块
          </Button>
          <Button type="primary" onClick={onSubmit} loading={submitting}>
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
    </Drawer>
  );
}
