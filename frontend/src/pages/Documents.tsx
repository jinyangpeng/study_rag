/**
 * Documents：KB 下的文档管理（列表 / 添加 / 删除 / 详情）。
 * 添加方式：拖拽 / 文本输入 → AddDocumentDrawer（支持策略选择 + 分块预览）。
 */

import { useEffect, useState } from "react";
import {
  Table,
  Tag,
  Space,
  Button,
  Typography,
  Empty,
  Spin,
  Drawer,
  Descriptions,
  App as AntdApp,
  Popconfirm,
} from "antd";
import {
  ReloadOutlined,
  PlusOutlined,
  DeleteOutlined,
  EyeOutlined,
  ArrowLeftOutlined,
  BlockOutlined,
} from "@ant-design/icons";
import { useNavigate, useParams } from "react-router-dom";
import { useApi } from "../api/client";
import type { DocumentMeta } from "../api/types";
import AddDocumentDrawer from "../components/AddDocumentDrawer";
import ChunksDrawer from "../components/ChunksDrawer";

const { Title, Text, Paragraph } = Typography;

export default function Documents() {
  const { kbId = "" } = useParams<{ kbId: string }>();
  const { client } = useApi();
  const { message } = AntdApp.useApp();
  const navigate = useNavigate();
  const [docs, setDocs] = useState<DocumentMeta[]>([]);
  const [loading, setLoading] = useState(false);
  const [addOpen, setAddOpen] = useState(false);
  const [detail, setDetail] = useState<DocumentMeta | null>(null);
  const [chunksOpen, setChunksOpen] = useState(false);
  const [chunksDocId, setChunksDocId] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const d = await client.listDocuments(kbId);
      setDocs(d);
    } catch (e) {
      message.error((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (kbId) void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kbId]);

  const onDelete = async (doc: DocumentMeta) => {
    try {
      await client.deleteDocument(doc.kb_id, doc.doc_id);
      message.success(`已删除 ${doc.doc_id}`);
      void load();
    } catch (e) {
      message.error((e as Error).message);
    }
  };

  const openDetail = async (doc: DocumentMeta) => {
    setDetail(doc);
    try {
      const d = await client.getDocument(doc.kb_id, doc.doc_id);
      setDetail(d);
    } catch (e) {
      message.error((e as Error).message);
    }
  };

  return (
    <div>
      <Space style={{ marginBottom: 16 }}>
        <Button
          icon={<ArrowLeftOutlined />}
          onClick={() => navigate("/kbs")}
        >
          返回 KB 列表
        </Button>
        <Title level={5} style={{ margin: 0 }}>
          KB: <Text code>{kbId}</Text>
        </Title>
      </Space>

      <Space style={{ marginBottom: 16 }}>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setAddOpen(true)}
        >
          添加文档
        </Button>
        <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
          刷新
        </Button>
        <Text type="secondary">共 {docs.length} 个文档</Text>
      </Space>

      {loading && docs.length === 0 ? (
        <div style={{ textAlign: "center", padding: 64 }}>
          <Spin size="large" />
        </div>
      ) : docs.length === 0 ? (
        <Empty description="该 KB 暂无文档" />
      ) : (
        <Table
          rowKey="doc_id"
          dataSource={docs}
          pagination={{ pageSize: 20, showSizeChanger: false }}
          columns={[
            {
              title: "doc_id",
              dataIndex: "doc_id",
              width: 220,
              render: (v: string) => <Text code>{v}</Text>,
            },
            {
              title: "标题",
              dataIndex: "title",
              render: (v: string, r) => (
                <Space direction="vertical" size={0}>
                  <Text strong>{v}</Text>
                  {r.source && (
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      source: {r.source}
                    </Text>
                  )}
                </Space>
              ),
            },
            {
              title: "metadata",
              dataIndex: "metadata",
              render: (m: Record<string, unknown>) =>
                m && Object.keys(m).length > 0 ? (
                  <Space size={4} wrap>
                    {Object.entries(m)
                      .slice(0, 4)
                      .map(([k, v]) => (
                        <Tag key={k}>{`${k}: ${String(v)}`}</Tag>
                      ))}
                    {Object.keys(m).length > 4 && (
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        +{Object.keys(m).length - 4}
                      </Text>
                    )}
                  </Space>
                ) : (
                  <Text type="secondary">-</Text>
                ),
            },
            {
              title: "创建时间",
              dataIndex: "created_at",
              width: 200,
              render: (v: string) => (
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {new Date(v).toLocaleString("zh-CN")}
                </Text>
              ),
            },
            {
              title: "操作",
              width: 280,
              render: (_, r) => (
                <Space size="small">
                  <Button
                    size="small"
                    icon={<BlockOutlined />}
                    onClick={() => {
                      setChunksDocId(r.doc_id);
                      setChunksOpen(true);
                    }}
                  >
                    查看分块
                  </Button>
                  <Button
                    size="small"
                    icon={<EyeOutlined />}
                    onClick={() => openDetail(r)}
                  >
                    详情
                  </Button>
                  <Popconfirm
                    title="确认删除？"
                    description={`将删除 ${r.doc_id} 的所有 chunks`}
                    onConfirm={() => onDelete(r)}
                    okText="删除"
                    cancelText="取消"
                    okButtonProps={{ danger: true }}
                  >
                    <Button
                      size="small"
                      danger
                      icon={<DeleteOutlined />}
                    >
                      删除
                    </Button>
                  </Popconfirm>
                </Space>
              ),
            },
          ]}
        />
      )}

      {/* Add Document Drawer */}
      <AddDocumentDrawer
        open={addOpen}
        kbId={kbId}
        onCancel={() => setAddOpen(false)}
        onSuccess={() => {
          setAddOpen(false);
          void load();
        }}
      />

      {/* Chunks Drawer */}
      <ChunksDrawer
        open={chunksOpen}
        kbId={kbId}
        docId={chunksDocId}
        onClose={() => {
          setChunksOpen(false);
          setChunksDocId(null);
        }}
      />

      {/* Detail Drawer */}
      <Drawer
        title={detail ? `文档详情: ${detail.doc_id}` : "文档详情"}
        width={720}
        open={!!detail}
        onClose={() => setDetail(null)}
      >
        {detail && (
          <>
            <Descriptions
              column={1}
              size="small"
              bordered
              items={[
                { key: "kb", label: "KB ID", children: <Text code>{detail.kb_id}</Text> },
                { key: "id", label: "doc_id", children: <Text code>{detail.doc_id}</Text> },
                { key: "t", label: "标题", children: detail.title },
                { key: "s", label: "source", children: detail.source || "-" },
                {
                  key: "ts",
                  label: "创建时间",
                  children: new Date(detail.created_at).toLocaleString("zh-CN"),
                },
              ]}
            />
            <div style={{ marginTop: 16 }}>
              <Text strong>Metadata</Text>
              <pre className="json-view" style={{ marginTop: 8 }}>
                {JSON.stringify(detail.metadata ?? {}, null, 2)}
              </pre>
            </div>
            {detail.content && (
              <div style={{ marginTop: 16 }}>
                <Text strong>正文</Text>
                <Paragraph
                  style={{
                    marginTop: 8,
                    padding: 12,
                    background: "#fafafa",
                    borderRadius: 4,
                    maxHeight: 400,
                    overflow: "auto",
                    whiteSpace: "pre-wrap",
                  }}
                >
                  {detail.content}
                </Paragraph>
              </div>
            )}
          </>
        )}
      </Drawer>
    </div>
  );
}
