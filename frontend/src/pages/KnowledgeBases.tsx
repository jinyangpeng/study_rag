/**
 * KnowledgeBases：KB 列表 + 详情侧栏。
 * 数据：GET /admin/kbs, GET /admin/kbs/{kb_id}
 */

import { useEffect, useState } from "react";
import {
  Table,
  Tag,
  Space,
  Button,
  Input,
  Typography,
  Empty,
  Spin,
  Drawer,
  Descriptions,
  Popconfirm,
  App as AntdApp,
} from "antd";
import {
  ReloadOutlined,
  EyeOutlined,
  FileTextOutlined,
  SearchOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
} from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
import { useApi } from "../api/client";
import type {
  KnowledgeBaseConfig,
  KnowledgeBaseSummary,
} from "../api/types";
import KbFormModal from "../components/KbFormModal";

const { Text } = Typography;
const { Search } = Input;

export default function KnowledgeBases() {
  const { client } = useApi();
  const { message, modal } = AntdApp.useApp();
  const navigate = useNavigate();
  const [kbs, setKbs] = useState<KnowledgeBaseSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [keyword, setKeyword] = useState("");
  const [detail, setDetail] = useState<KnowledgeBaseSummary | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  // KB 表单弹窗状态
  const [formOpen, setFormOpen] = useState(false);
  const [formMode, setFormMode] = useState<"create" | "edit">("create");
  const [editTarget, setEditTarget] = useState<KnowledgeBaseConfig | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const d = await client.listKBs();
      setKbs(d);
    } catch (e) {
      message.error((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const openDetail = async (kb: KnowledgeBaseSummary) => {
    setDetail(kb);
    setDetailLoading(true);
    try {
      const d = await client.getKB(kb.kb_id);
      setDetail(d);
    } catch (e) {
      message.error((e as Error).message);
    } finally {
      setDetailLoading(false);
    }
  };

  const openEdit = (kb: KnowledgeBaseSummary) => {
    // summary 已经带 collection / embedding / reranker，直接转
    const cfg: KnowledgeBaseConfig = {
      kb_id: kb.kb_id,
      name: kb.name,
      description: kb.description ?? "",
      department: kb.department ?? "",
      collection: kb.collection ?? `kb_${kb.kb_id}`,
      embedding: kb.embedder ?? "",
      reranker: kb.reranker ?? null,
      enabled: kb.enabled,
    };
    setEditTarget(cfg);
    setFormMode("edit");
    setFormOpen(true);
  };

  const handleDelete = async (kb: KnowledgeBaseSummary) => {
    try {
      await client.deleteKB(kb.kb_id);
      message.success(`KB '${kb.kb_id}' 已删除`);
      void load();
    } catch (e) {
      message.error((e as Error).message);
    }
  };

  const confirmDelete = (kb: KnowledgeBaseSummary) => {
    modal.confirm({
      title: `删除 KB '${kb.kb_id}'？`,
      content: (
        <div>
          <p>该操作<strong>不可恢复</strong>：</p>
          <ul style={{ paddingLeft: 20 }}>
            <li>删除 KB 配置</li>
            <li>drop vector store collection（所有向量丢失）</li>
            <li>清空 in-memory 文档</li>
          </ul>
          <p>
            如果只想临时停用，编辑 KB 把 <code>enabled</code> 设为 false。
          </p>
        </div>
      ),
      okText: "确认删除",
      okButtonProps: { danger: true },
      cancelText: "取消",
      onOk: () => handleDelete(kb),
    });
  };

  const filtered = kbs.filter((k) => {
    if (!keyword) return true;
    const kw = keyword.toLowerCase();
    return (
      k.kb_id.toLowerCase().includes(kw) ||
      (k.name || "").toLowerCase().includes(kw) ||
      (k.description || "").toLowerCase().includes(kw)
    );
  });

  return (
    <div>
      <Space style={{ marginBottom: 16 }} wrap>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => {
            setEditTarget(null);
            setFormMode("create");
            setFormOpen(true);
          }}
        >
          新建知识库
        </Button>
        <Button
          icon={<ReloadOutlined />}
          onClick={load}
          loading={loading}
        >
          刷新
        </Button>
        <Search
          placeholder="搜索 kb_id / name / description"
          allowClear
          onChange={(e) => setKeyword(e.target.value)}
          style={{ width: 320 }}
        />
        <Text type="secondary">
          共 {filtered.length} / {kbs.length} 个 KB
        </Text>
      </Space>

      {loading && kbs.length === 0 ? (
        <div style={{ textAlign: "center", padding: 64 }}>
          <Spin size="large" />
        </div>
      ) : filtered.length === 0 ? (
        <Empty description={kbs.length === 0 ? "暂无 KB" : "无匹配结果"} />
      ) : (
        <Table
          rowKey="kb_id"
          dataSource={filtered}
          pagination={{ pageSize: 20, showSizeChanger: false }}
          columns={[
            {
              title: "KB ID",
              dataIndex: "kb_id",
              width: 200,
              render: (v: string) => <Text code>{v}</Text>,
            },
            {
              title: "名称",
              dataIndex: "name",
              render: (v: string | null | undefined, r) => (
                <Space direction="vertical" size={0}>
                  <Text strong>{v || r.kb_id}</Text>
                  {r.description && (
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {r.description}
                    </Text>
                  )}
                </Space>
              ),
            },
            {
              title: "状态",
              dataIndex: "enabled",
              width: 100,
              render: (v: boolean) =>
                v ? <Tag color="success">enabled</Tag> : <Tag>disabled</Tag>,
            },
            {
              title: "文档 / Chunk",
              width: 160,
              render: (_, r) => (
                <Space size="small">
                  <Tag>{r.document_count} docs</Tag>
                  <Tag color="blue">{r.chunk_count} chunks</Tag>
                </Space>
              ),
            },
            {
              title: "组件",
              render: (_, r) => (
                <Space size={4} wrap>
                  {r.embedder && <Tag color="purple">embed: {r.embedder}</Tag>}
                  {r.reranker && <Tag color="cyan">rerank: {r.reranker}</Tag>}
                  {r.vector_store && <Tag>vs: {r.vector_store}</Tag>}
                </Space>
              ),
            },
            {
              title: "操作",
              width: 360,
              render: (_, r) => (
                <Space size="small" wrap>
                  <Button
                    size="small"
                    icon={<EyeOutlined />}
                    onClick={() => openDetail(r)}
                  >
                    详情
                  </Button>
                  <Button
                    size="small"
                    icon={<EditOutlined />}
                    onClick={() => openEdit(r)}
                  >
                    编辑
                  </Button>
                  <Button
                    size="small"
                    icon={<FileTextOutlined />}
                    type="primary"
                    ghost
                    onClick={() =>
                      navigate(
                        `/kbs/${encodeURIComponent(r.kb_id)}/documents`
                      )
                    }
                  >
                    文档
                  </Button>
                  <Button
                    size="small"
                    icon={<SearchOutlined />}
                    onClick={() =>
                      navigate(`/search?kb=${encodeURIComponent(r.kb_id)}`)
                    }
                  >
                    检索
                  </Button>
                  <Popconfirm
                    title="删除 KB？"
                    description="向量数据不可恢复"
                    onConfirm={() => confirmDelete(r)}
                    okText="删除"
                    okButtonProps={{ danger: true }}
                    cancelText="取消"
                  >
                    <Button
                      size="small"
                      danger
                      icon={<DeleteOutlined />}
                    />
                  </Popconfirm>
                </Space>
              ),
            },
          ]}
        />
      )}

      <Drawer
        title={detail ? `KB 详情: ${detail.kb_id}` : "KB 详情"}
        width={520}
        open={!!detail}
        onClose={() => setDetail(null)}
        loading={detailLoading}
      >
        {detail && (
          <Descriptions
            column={1}
            bordered
            size="small"
            items={[
              { key: "id", label: "KB ID", children: <Text code>{detail.kb_id}</Text> },
              { key: "name", label: "名称", children: detail.name || "-" },
              { key: "desc", label: "描述", children: detail.description || "-" },
              {
                key: "en",
                label: "状态",
                children: detail.enabled ? (
                  <Tag color="success">enabled</Tag>
                ) : (
                  <Tag>disabled</Tag>
                ),
              },
              {
                key: "doc",
                label: "文档 / Chunk",
                children: `${detail.document_count} docs / ${detail.chunk_count} chunks`,
              },
              { key: "embed", label: "Embedder", children: detail.embedder || "-" },
              { key: "rerank", label: "Reranker", children: detail.reranker || "-" },
              { key: "vs", label: "Vector Store", children: detail.vector_store || "-" },
            ]}
          />
        )}
      </Drawer>

      <KbFormModal
        open={formOpen}
        mode={formMode}
        initial={editTarget}
        onCancel={() => setFormOpen(false)}
        onSuccess={() => {
          setFormOpen(false);
          void load();
        }}
      />
    </div>
  );
}
