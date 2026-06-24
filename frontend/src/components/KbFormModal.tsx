/**
 * KbFormModal：新建/编辑知识库的表单弹窗。
 *
 * 字段：
 *   - kb_id（创建时必填，编辑时禁用）
 *   - name
 *   - description
 *   - department
 *   - collection（可选；创建时默认 kb_<kb_id>）
 *   - embedding（下拉选；选中后只读显示 dimension）
 *   - reranker（下拉选；可空 = 不用重排）
 *   - enabled
 *
 * 设计要点：
 *   - embedding / reranker 从 GET /admin/embedders /rerankers 拉
 *   - 「未加载」的 provider 也可选，但标红 + warning，提示用户「选了也 init 失败」
 *   - 提交时 kb_id 格式校验、dimension 锁定、enabled 默认 true
 *   - 提交后调 onSuccess() 通知父组件刷新列表
 */

import { useEffect, useState } from "react";
import {
  Form,
  Input,
  Select,
  Switch,
  Tag,
  Tooltip,
  Alert,
  Space,
  Typography,
} from "antd";
import { App as AntdApp, Modal } from "antd";
import {
  ApiOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  InfoCircleOutlined,
} from "@ant-design/icons";
import { useApi } from "../api/client";
import type {
  EmbedderInfo,
  KnowledgeBaseConfig,
  KnowledgeBaseCreate,
  KnowledgeBaseUpdate,
  RerankerInfo,
} from "../api/types";

interface KbFormModalProps {
  open: boolean;
  mode: "create" | "edit";
  initial?: KnowledgeBaseConfig | null;
  onCancel: () => void;
  onSuccess: () => void;
}

interface FormValues {
  kb_id: string;
  name: string;
  description: string;
  department: string;
  collection: string;
  embedding: string;
  reranker: string | undefined;
  enabled: boolean;
}

export default function KbFormModal({
  open,
  mode,
  initial,
  onCancel,
  onSuccess,
}: KbFormModalProps) {
  const { client } = useApi();
  const { message } = AntdApp.useApp();
  const [form] = Form.useForm<FormValues>();
  const [embedders, setEmbedders] = useState<EmbedderInfo[]>([]);
  const [rerankers, setRerankers] = useState<RerankerInfo[]>([]);
  const [loadingOptions, setLoadingOptions] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const watchedEmbedding = Form.useWatch("embedding", form);
  const watchedKbId = Form.useWatch("kb_id", form);

  // 拉 embedders / rerankers
  useEffect(() => {
    if (!open) return;
    (async () => {
      setLoadingOptions(true);
      try {
        const [e, r] = await Promise.all([
          client.listEmbedders(),
          client.listRerankers(),
        ]);
        setEmbedders(e);
        setRerankers(r);
      } catch (err) {
        message.error((err as Error).message);
      } finally {
        setLoadingOptions(false);
      }
    })();
  }, [open, client, message]);

  // 弹窗打开时填表单
  useEffect(() => {
    if (!open) return;
    if (mode === "edit" && initial) {
      form.setFieldsValue({
        kb_id: initial.kb_id,
        name: initial.name,
        description: initial.description,
        department: initial.department,
        collection: initial.collection,
        embedding: initial.embedding,
        reranker: initial.reranker ?? undefined,
        enabled: initial.enabled,
      });
    } else {
      form.setFieldsValue({
        kb_id: "",
        name: "",
        description: "",
        department: "",
        collection: "",
        embedding: embedders.find((e) => e.loaded)?.name ?? embedders[0]?.name,
        reranker: undefined,
        enabled: true,
      });
    }
  }, [open, mode, initial, embedders, form]);

  // 选中 embedder 自动填 collection
  useEffect(() => {
    if (mode !== "create") return;
    const kbId = form.getFieldValue("kb_id");
    if (kbId) {
      form.setFieldValue("collection", `kb_${kbId}`);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [watchedKbId, mode]);

  const selectedEmbedder = embedders.find((e) => e.name === watchedEmbedding);

  const handleSubmit = async () => {
    const v = await form.validateFields();
    setSubmitting(true);
    try {
      if (mode === "create") {
        const payload: KnowledgeBaseCreate = {
          kb_id: v.kb_id,
          name: v.name,
          description: v.description,
          department: v.department,
          collection: v.collection || undefined,
          embedding: v.embedding,
          reranker: v.reranker ?? null,
          enabled: v.enabled,
        };
        const cfg = await client.createKB(payload);
        message.success(
          `KB '${cfg.kb_id}' 创建成功${
            selectedEmbedder && !selectedEmbedder.loaded
              ? "（注意：embedder 未加载，KB 会被 skip）"
              : ""
          }`
        );
      } else {
        const patch: KnowledgeBaseUpdate = {
          name: v.name,
          description: v.description,
          department: v.department,
          reranker: v.reranker ?? null,
          enabled: v.enabled,
        };
        const cfg = await client.updateKB(initial!.kb_id, patch);
        message.success(`KB '${cfg.kb_id}' 已更新`);
      }
      onSuccess();
    } catch (err) {
      message.error((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal
      title={
        <Space>
          {mode === "create" ? (
            <>
              <DatabaseOutlined /> 新建知识库
            </>
          ) : (
            <>
              <DatabaseOutlined /> 编辑知识库 · {initial?.kb_id}
            </>
          )}
        </Space>
      }
      open={open}
      onCancel={onCancel}
      onOk={handleSubmit}
      confirmLoading={submitting}
      width={680}
      okText={mode === "create" ? "创建" : "保存"}
      cancelText="取消"
      destroyOnClose
    >
      {mode === "edit" && (
        <Alert
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
          message="不支持修改"
          description="kb_id / collection / embedding 改完需要重建 collection（破坏数据），请删了重建。"
        />
      )}

      <Form<FormValues> form={form} layout="vertical" preserve={false}>
        <Form.Item
          label="KB ID"
          name="kb_id"
          rules={[
            { required: true, message: "请输入 KB ID" },
            {
              pattern: /^[a-z][a-z0-9_]*$/,
              message: "必须以小写字母开头，只能包含小写字母/数字/下划线",
            },
            { min: 2, max: 64, message: "长度 2-64" },
          ]}
        >
          <Input
            prefix={<DatabaseOutlined />}
            placeholder="rd_frontend"
            disabled={mode === "edit"}
          />
        </Form.Item>

        <Form.Item
          label="名称"
          name="name"
          rules={[{ required: true, message: "请输入名称" }]}
        >
          <Input placeholder="前端研发知识库" />
        </Form.Item>

        <Form.Item
          label="描述"
          name="description"
          tooltip="Agent 选 KB 的依据，写清楚有什么内容、解决什么问题"
          rules={[{ required: true, message: "请输入描述" }]}
        >
          <Input.TextArea
            rows={2}
            placeholder="React/Vue/TypeScript/性能优化等前端开发相关的内部技术文档"
          />
        </Form.Item>

        <Form.Item
          label="部门"
          name="department"
          rules={[{ required: true, message: "请输入部门" }]}
        >
          <Input placeholder="RD" />
        </Form.Item>

        <Form.Item
          label="Collection 名称（向量库）"
          name="collection"
          tooltip="不填则用 'kb_<kb_id>'"
        >
          <Input
            placeholder={`kb_${watchedKbId || "<kb_id>"}`}
            disabled={mode === "edit"}
          />
        </Form.Item>

        <Form.Item
          label={
            <Space>
              Embedding 模型
              <Tooltip title="向量化的模型；选错会导致 KB init 失败">
                <InfoCircleOutlined />
              </Tooltip>
            </Space>
          }
          name="embedding"
          rules={[{ required: true, message: "请选 embedding" }]}
        >
          <Select
            placeholder="选择 embedding 配置"
            loading={loadingOptions}
            disabled={mode === "edit"}
            options={embedders.map((e) => ({
              value: e.name,
              label: (
                <Space>
                  <ExperimentOutlined />
                  <span>{e.name}</span>
                  <Tag color="blue" style={{ marginLeft: 4 }}>
                    {e.provider}
                  </Tag>
                  <span style={{ color: "#888" }}>dim={e.dimension}</span>
                  {!e.loaded && (
                    <Tag color="red" style={{ marginLeft: 4 }}>
                      未加载
                    </Tag>
                  )}
                </Space>
              ),
              disabled: false,
            }))}
            optionFilterProp="label"
            showSearch
          />
        </Form.Item>

        {selectedEmbedder && (
          <Alert
            type={selectedEmbedder.loaded ? "info" : "warning"}
            showIcon
            style={{ marginBottom: 16, marginTop: -8 }}
            message={
              <Space size="small">
                <span>模型：{selectedEmbedder.model_name}</span>
                <span>·</span>
                <span>维度：{selectedEmbedder.dimension}</span>
                <span>·</span>
                <span>batch：{selectedEmbedder.batch_size}</span>
                {selectedEmbedder.loaded ? (
                  <Tag color="success" icon={<ApiOutlined />}>
                    已加载
                  </Tag>
                ) : (
                  <Tag color="red">未加载（KB 会被 skip）</Tag>
                )}
              </Space>
            }
            description={
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                {selectedEmbedder.description ||
                  "（无描述）"}
              </Typography.Text>
            }
          />
        )}

        <Form.Item
          label={
            <Space>
              Reranker 模型
              <Tooltip title="可选；不选 = 不重排（影响精度但速度更快）">
                <InfoCircleOutlined />
              </Tooltip>
            </Space>
          }
          name="reranker"
        >
          <Select
            placeholder="不选 = 不用重排"
            allowClear
            loading={loadingOptions}
            options={rerankers.map((r) => ({
              value: r.name,
              label: (
                <Space>
                  <ExperimentOutlined />
                  <span>{r.name}</span>
                  <Tag color="cyan">{r.provider}</Tag>
                  <span style={{ color: "#888" }}>top_k={r.top_k}</span>
                  {!r.loaded && (
                    <Tag color="red" style={{ marginLeft: 4 }}>
                      未加载
                    </Tag>
                  )}
                </Space>
              ),
            }))}
          />
        </Form.Item>

        <Form.Item
          label="启用"
          name="enabled"
          valuePropName="checked"
          tooltip="关闭后 KB 在 MCP 端不可见、检索时跳过"
        >
          <Switch />
        </Form.Item>
      </Form>
    </Modal>
  );
}
