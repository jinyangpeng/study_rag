/**
 * SearchTest：可视化检索调试（与 MCP search_kb 共享后端）。
 * - 选 KB / 输入 query / top_k / rerank / filter
 * - 看 hits（score / text / metadata）+ 耗时
 */

import { useEffect, useState } from "react";
import {
  Card,
  Form,
  Input,
  InputNumber,
  Switch,
  Select,
  Button,
  Space,
  Typography,
  Empty,
  Spin,
  Tag,
  Alert,
  Descriptions,
  theme,
  App as AntdApp,
} from "antd";
import {
  SearchOutlined,
  ThunderboltOutlined,
  ClockCircleOutlined,
} from "@ant-design/icons";
import { useSearchParams } from "react-router-dom";
import { useApi } from "../api/client";
import type { KnowledgeBaseSummary, SearchHit, SearchResponse } from "../api/types";

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;

interface SearchFormValues {
  kb_id: string;
  query: string;
  top_k: number;
  use_rerank: boolean;
  filter_expr?: string;
}

export default function SearchTest() {
  const { client } = useApi();
  const { message } = AntdApp.useApp();
  const { token: themeToken } = theme.useToken();
  const [searchParams, setSearchParams] = useSearchParams();
  const [kbs, setKbs] = useState<KnowledgeBaseSummary[]>([]);
  const [kbsLoading, setKbsLoading] = useState(false);
  const [form] = Form.useForm<SearchFormValues>();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // 加载 KB 列表
  useEffect(() => {
    (async () => {
      setKbsLoading(true);
      try {
        const d = await client.listKBs();
        setKbs(d);
        // 默认从 URL ?kb= 取
        const urlKb = searchParams.get("kb");
        const initial = urlKb && d.some((k) => k.kb_id === urlKb) ? urlKb : d[0]?.kb_id;
        if (initial) {
          form.setFieldsValue({
            kb_id: initial,
            top_k: 5,
            use_rerank: true,
          });
        }
      } catch (e) {
        message.error((e as Error).message);
      } finally {
        setKbsLoading(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onSearch = async () => {
    try {
      const v = await form.validateFields();
      setLoading(true);
      setError(null);
      setResult(null);

      // 同步到 URL
      setSearchParams({ kb: v.kb_id });

      let filterExpr: Record<string, unknown> | null = null;
      if (v.filter_expr && v.filter_expr.trim()) {
        try {
          const parsed = JSON.parse(v.filter_expr);
          if (parsed && typeof parsed === "object") {
            filterExpr = parsed as Record<string, unknown>;
          } else {
            throw new Error("filter_expr 必须是 JSON object");
          }
        } catch (e) {
          throw new Error(`filter_expr JSON 解析失败: ${(e as Error).message}`);
        }
      }

      const data = await client.search(v.kb_id, {
        query: v.query,
        top_k: v.top_k,
        use_rerank: v.use_rerank,
        filter_expr: filterExpr,
      });
      setResult(data);
    } catch (e) {
      if ((e as { errorFields?: unknown }).errorFields) return;
      const msg = (e as Error).message;
      setError(msg);
      message.error(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <Card size="small" style={{ marginBottom: 16 }}>
        <Form
          form={form}
          layout="vertical"
          onFinish={onSearch}
          initialValues={{ top_k: 5, use_rerank: true }}
        >
          <Space size="middle" wrap style={{ display: "flex" }}>
            <Form.Item
              label="知识库"
              name="kb_id"
              rules={[{ required: true, message: "请选择 KB" }]}
              style={{ minWidth: 220, marginBottom: 16 }}
            >
              <Select
                placeholder="选择 KB"
                loading={kbsLoading}
                options={kbs.map((k) => ({
                  value: k.kb_id,
                  label: `${k.kb_id} (${k.document_count} docs)`,
                }))}
              />
            </Form.Item>
            <Form.Item
              label="top_k"
              name="top_k"
              style={{ marginBottom: 16 }}
            >
              <InputNumber min={1} max={50} style={{ width: 100 }} />
            </Form.Item>
            <Form.Item
              label="rerank"
              name="use_rerank"
              valuePropName="checked"
              style={{ marginBottom: 16 }}
            >
              <Switch />
            </Form.Item>
            <Form.Item label=" " style={{ marginBottom: 16 }}>
              <Button
                type="primary"
                htmlType="submit"
                icon={<SearchOutlined />}
                loading={loading}
              >
                检索
              </Button>
            </Form.Item>
          </Space>
          <Form.Item
            label="Query"
            name="query"
            rules={[{ required: true, message: "请输入 query" }]}
          >
            <TextArea
              rows={2}
              placeholder="例如：React 性能优化"
              autoFocus
            />
          </Form.Item>
          <Form.Item
            label="filter_expr (JSON object, 可选)"
            name="filter_expr"
            tooltip='metadata 过滤，例如 {"source": "wiki"} 或 {"year": {"$gte": 2024}}'
          >
            <Input placeholder='{"source": "wiki"}' />
          </Form.Item>
        </Form>
      </Card>

      {error && (
        <Alert
          type="error"
          showIcon
          message="检索失败"
          description={error}
          style={{ marginBottom: 16 }}
        />
      )}

      {loading && (
        <div style={{ textAlign: "center", padding: 48 }}>
          <Spin size="large" tip="检索中..." />
        </div>
      )}

      {result && !loading && (
        <>
          <Card
            size="small"
            style={{ marginBottom: 16 }}
            title={
              <Space>
                <ThunderboltOutlined style={{ color: themeToken.colorPrimary }} />
                <Text strong>检索结果</Text>
                <Tag color="blue">{result.hits.length} hits</Tag>
                <Tag icon={<ClockCircleOutlined />} color="default">
                  {result.duration_ms} ms
                </Tag>
              </Space>
            }
          >
            <Descriptions
              column={3}
              size="small"
              items={[
                { key: "kb", label: "KB", children: <Text code>{result.kb_id}</Text> },
                { key: "q", label: "Query", children: result.query },
                { key: "n", label: "结果数", children: result.hits.length },
              ]}
            />
          </Card>

          {result.hits.length === 0 ? (
            <Empty description="无命中" />
          ) : (
            result.hits.map((hit, idx) => (
              <HitCard key={`${hit.doc_id}-${hit.chunk_id}-${idx}`} hit={hit} index={idx} />
            ))
          )}
        </>
      )}

      {!loading && !result && !error && (
        <div className="empty-tip">
          <Title level={4} type="secondary">
            输入 query 后点「检索」开始测试
          </Title>
        </div>
      )}
    </div>
  );
}

function HitCard({ hit, index }: { hit: SearchHit; index: number }) {
  return (
    <Card
      size="small"
      style={{ marginBottom: 12 }}
      title={
        <Space>
          <Tag color="blue">#{index + 1}</Tag>
          <Text code>{hit.doc_id}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            chunk {hit.chunk_id}
          </Text>
        </Space>
      }
      extra={
        <Tag color="green" style={{ fontWeight: 600 }}>
          score: {hit.score.toFixed(4)}
        </Tag>
      }
    >
      <Paragraph
        style={{
          marginBottom: 12,
          whiteSpace: "pre-wrap",
          fontSize: 13,
          lineHeight: 1.6,
        }}
      >
        {hit.text}
      </Paragraph>
      {hit.metadata && Object.keys(hit.metadata).length > 0 && (
        <Space size={4} wrap>
          {Object.entries(hit.metadata).map(([k, v]) => (
            <Tag key={k}>{`${k}: ${String(v)}`}</Tag>
          ))}
        </Space>
      )}
    </Card>
  );
}
