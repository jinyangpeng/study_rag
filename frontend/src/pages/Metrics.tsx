/**
 * Metrics：展示 Prometheus 格式 metrics 文本。
 * 数据来源：GET /metrics (text/plain)
 *
 * 提供：
 *  - 原始文本查看
 *  - 按 metric name 聚合（counter / histogram / gauge）
 *  - 搜索过滤
 */

import { useEffect, useMemo, useState } from "react";
import {
  Card,
  Space,
  Button,
  Input,
  Typography,
  Tag,
  Table,
  Alert,
  App as AntdApp,
  Tabs,
} from "antd";
import { ReloadOutlined, CopyOutlined } from "@ant-design/icons";
import { useApi } from "../api/client";

const { Text, Paragraph } = Typography;
const { Search } = Input;

interface MetricEntry {
  name: string;
  help: string;
  type: "counter" | "gauge" | "histogram" | "summary" | "unknown";
  samples: MetricSample[];
}

interface MetricSample {
  labels: string;
  value: number;
}

function parseProm(text: string): MetricEntry[] {
  const lines = text.split("\n");
  const meta = new Map<string, { help: string; type: MetricEntry["type"] }>();
  const samples = new Map<string, MetricSample[]>();

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;
    if (line.startsWith("# HELP ")) {
      const rest = line.slice(7);
      const spaceIdx = rest.indexOf(" ");
      const name = rest.slice(0, spaceIdx);
      const help = rest.slice(spaceIdx + 1);
      const cur = meta.get(name) || { help: "", type: "unknown" };
      cur.help = help;
      meta.set(name, cur);
    } else if (line.startsWith("# TYPE ")) {
      const rest = line.slice(7);
      const spaceIdx = rest.indexOf(" ");
      const name = rest.slice(0, spaceIdx);
      const t = rest.slice(spaceIdx + 1) as MetricEntry["type"];
      const cur = meta.get(name) || { help: "", type: t };
      cur.type = t;
      meta.set(name, cur);
    } else if (!line.startsWith("#")) {
      // 形如: metric_name{labels} value [timestamp]
      const spaceIdx = line.lastIndexOf(" ");
      if (spaceIdx < 0) continue;
      const head = line.slice(0, spaceIdx);
      const valueStr = line.slice(spaceIdx + 1);
      const value = parseFloat(valueStr);
      if (Number.isNaN(value)) continue;

      const lbIdx = head.indexOf("{");
      let name: string;
      let labels = "";
      if (lbIdx < 0) {
        name = head;
      } else {
        name = head.slice(0, lbIdx);
        labels = head.slice(lbIdx + 1, head.lastIndexOf("}"));
      }
      if (!samples.has(name)) samples.set(name, []);
      samples.get(name)!.push({ labels, value });
    }
  }

  const result: MetricEntry[] = [];
  for (const [name, s] of samples) {
    const m = meta.get(name) || { help: "", type: "unknown" as const };
    result.push({ name, help: m.help, type: m.type, samples: s });
  }
  return result.sort((a, b) => a.name.localeCompare(b.name));
}

export default function Metrics() {
  const { client } = useApi();
  const { message } = AntdApp.useApp();
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const t = await client.getMetricsText();
      setText(t);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const entries = useMemo(() => parseProm(text), [text]);
  const filtered = entries.filter((e) => {
    if (!filter) return true;
    const kw = filter.toLowerCase();
    return (
      e.name.toLowerCase().includes(kw) ||
      e.help.toLowerCase().includes(kw) ||
      e.samples.some((s) => s.labels.toLowerCase().includes(kw))
    );
  });

  const typeColor: Record<MetricEntry["type"], string> = {
    counter: "blue",
    gauge: "green",
    histogram: "purple",
    summary: "magenta",
    unknown: "default",
  };

  return (
    <div>
      <Space style={{ marginBottom: 16 }} wrap>
        <Button
          type="primary"
          icon={<ReloadOutlined />}
          onClick={load}
          loading={loading}
        >
          刷新
        </Button>
        <Search
          placeholder="过滤 metric / label / help"
          allowClear
          onChange={(e) => setFilter(e.target.value)}
          style={{ width: 320 }}
        />
        <Button
          icon={<CopyOutlined />}
          onClick={() => {
            void navigator.clipboard
              .writeText(text)
              .then(() => message.success("已复制到剪贴板"));
          }}
        >
          复制原始
        </Button>
        <Text type="secondary">
          {filtered.length} / {entries.length} metrics
        </Text>
      </Space>

      {error && (
        <Alert
          type="error"
          showIcon
          message="加载失败"
          description={error}
          style={{ marginBottom: 16 }}
        />
      )}

      <Tabs
        items={[
          {
            key: "table",
            label: "结构化视图",
            children: (
              <Table
                rowKey="name"
                size="small"
                pagination={{ pageSize: 20, showSizeChanger: false }}
                dataSource={filtered}
                columns={[
                  {
                    title: "Metric",
                    dataIndex: "name",
                    width: 360,
                    render: (v: string) => <Text code>{v}</Text>,
                  },
                  {
                    title: "Type",
                    dataIndex: "type",
                    width: 110,
                    render: (t: MetricEntry["type"]) => (
                      <Tag color={typeColor[t]}>{t}</Tag>
                    ),
                  },
                  {
                    title: "Samples",
                    width: 100,
                    render: (_, r) => <Tag>{r.samples.length}</Tag>,
                  },
                  {
                    title: "Help",
                    dataIndex: "help",
                    render: (h: string) => (
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {h || "-"}
                      </Text>
                    ),
                  },
                  {
                    title: "最新值",
                    width: 360,
                    render: (_, r) => (
                      <Space direction="vertical" size={2} style={{ width: "100%" }}>
                        {r.samples.slice(0, 3).map((s, i) => (
                          <Space
                            key={i}
                            size={4}
                            style={{ fontSize: 12, fontFamily: "monospace" }}
                          >
                            <Text type="secondary" style={{ fontSize: 12 }}>
                              {s.labels || "{}"}
                            </Text>
                            <Text strong style={{ fontSize: 12 }}>
                              {s.value}
                            </Text>
                          </Space>
                        ))}
                        {r.samples.length > 3 && (
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            +{r.samples.length - 3} more
                          </Text>
                        )}
                      </Space>
                    ),
                  },
                ]}
              />
            ),
          },
          {
            key: "raw",
            label: "原始文本",
            children: (
              <Card size="small">
                <Paragraph
                  style={{ marginBottom: 0, fontSize: 12 }}
                  copyable
                >
                  <pre className="json-view" style={{ background: "#fafafa", color: "#333" }}>
                    {text || "(空)"}
                  </pre>
                </Paragraph>
              </Card>
            ),
          },
        ]}
      />
    </div>
  );
}
