/**
 * Dashboard：展示系统健康状态、限流器、熔断器、KB 概览。
 * 数据来源：GET /health/detailed
 */

import { useEffect, useState } from "react";
import {
  Card,
  Col,
  Row,
  Statistic,
  Tag,
  Typography,
  Space,
  Spin,
  Alert,
  Button,
  Descriptions,
  Progress,
  theme,
} from "antd";
import {
  ReloadOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  CloseCircleOutlined,
  DatabaseOutlined,
  ApiOutlined,
  ThunderboltOutlined,
} from "@ant-design/icons";
import { useApi } from "../api/client";
import type { HealthDetailed } from "../api/types";

const { Title, Text } = Typography;

function StateTag({ state }: { state: string }) {
  if (state === "closed")
    return (
      <Tag icon={<CheckCircleOutlined />} color="success">
        CLOSED
      </Tag>
    );
  if (state === "half_open")
    return (
      <Tag icon={<WarningOutlined />} color="warning">
        HALF_OPEN
      </Tag>
    );
  return (
    <Tag icon={<CloseCircleOutlined />} color="error">
      OPEN
    </Tag>
  );
}

export default function Dashboard() {
  const { client } = useApi();
  const { token: themeToken } = theme.useToken();
  const [data, setData] = useState<HealthDetailed | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const d = await client.getHealthDetailed();
      setData(d);
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

  const cbEntries = data ? Object.entries(data.circuit_breakers) : [];
  const openCBs = cbEntries.filter(([, s]) => s.state !== "closed").length;

  return (
    <div>
      <Space style={{ marginBottom: 16 }}>
        <Button
          icon={<ReloadOutlined />}
          onClick={load}
          loading={loading}
          type="primary"
        >
          刷新
        </Button>
        <Text type="secondary">数据来源：GET /health/detailed</Text>
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

      {loading && !data && (
        <div style={{ textAlign: "center", padding: 64 }}>
          <Spin size="large" tip="加载中..." />
        </div>
      )}

      {data && (
        <>
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={12} md={6}>
              <Card>
                <Statistic
                  title="系统状态"
                  value={data.status}
                  valueStyle={{
                    color:
                      data.status === "ok"
                        ? themeToken.colorSuccess
                        : themeToken.colorError,
                  }}
                  prefix={
                    data.status === "ok" ? (
                      <CheckCircleOutlined />
                    ) : (
                      <CloseCircleOutlined />
                    )
                  }
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Card>
                <Statistic
                  title="知识库"
                  value={data.kbs_enabled}
                  suffix={`/ ${data.kbs_total}`}
                  prefix={<DatabaseOutlined />}
                />
                <Progress
                  percent={
                    data.kbs_total === 0
                      ? 0
                      : Math.round((data.kbs_enabled / data.kbs_total) * 100)
                  }
                  size="small"
                  showInfo={false}
                  style={{ marginTop: 8 }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Card>
                <Statistic
                  title="Embedder / Reranker"
                  value={data.embedders}
                  suffix={`/ ${data.rerankers}`}
                  prefix={<ApiOutlined />}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Card>
                <Statistic
                  title="熔断器 OPEN"
                  value={openCBs}
                  suffix={`/ ${cbEntries.length}`}
                  prefix={<ThunderboltOutlined />}
                  valueStyle={{
                    color:
                      openCBs === 0
                        ? themeToken.colorSuccess
                        : themeToken.colorError,
                  }}
                />
              </Card>
            </Col>
          </Row>

          <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
            <Col xs={24} md={12}>
              <Card title="限流器" size="small">
                <Descriptions
                  column={1}
                  size="small"
                  bordered
                  items={[
                    {
                      key: "admin",
                      label: (
                        <Space>
                          <Text strong>admin</Text>
                          <Tag color="blue">写操作</Tag>
                        </Space>
                      ),
                      children: (
                        <Descriptions
                          column={3}
                          size="small"
                          items={[
                            {
                              key: "cap",
                              label: "容量",
                              children: data.ratelimit.admin.capacity,
                            },
                            {
                              key: "rate",
                              label: "QPS",
                              children: data.ratelimit.admin.refill_rate,
                            },
                            {
                              key: "keys",
                              label: "tracked",
                              children: data.ratelimit.admin.tracked_keys,
                            },
                          ]}
                        />
                      ),
                    },
                    {
                      key: "search",
                      label: (
                        <Space>
                          <Text strong>search</Text>
                          <Tag color="cyan">读操作</Tag>
                        </Space>
                      ),
                      children: (
                        <Descriptions
                          column={3}
                          size="small"
                          items={[
                            {
                              key: "cap",
                              label: "容量",
                              children: data.ratelimit.search.capacity,
                            },
                            {
                              key: "rate",
                              label: "QPS",
                              children: data.ratelimit.search.refill_rate,
                            },
                            {
                              key: "keys",
                              label: "tracked",
                              children: data.ratelimit.search.tracked_keys,
                            },
                          ]}
                        />
                      ),
                    },
                  ]}
                />
              </Card>
            </Col>

            <Col xs={24} md={12}>
              <Card title="熔断器" size="small">
                <Descriptions
                  column={1}
                  size="small"
                  bordered
                  items={cbEntries.map(([name, s]) => ({
                    key: name,
                    label: <Text strong>{name}</Text>,
                    children: (
                      <Space size="middle">
                        <StateTag state={s.state} />
                        <Text type="secondary">
                          失败 {s.failure_count} 次
                          {s.retry_after_s > 0 && (
                            <span> · {s.retry_after_s.toFixed(1)}s 后探测</span>
                          )}
                        </Text>
                      </Space>
                    ),
                  }))}
                />
              </Card>
            </Col>
          </Row>

          <Card
            title="系统信息"
            size="small"
            style={{ marginTop: 16 }}
          >
            <Descriptions
              column={3}
              size="small"
              bordered
              items={[
                {
                  key: "auth",
                  label: "鉴权",
                  children: data.auth_enabled ? (
                    <Tag color="success">已启用</Tag>
                  ) : (
                    <Tag>未启用</Tag>
                  ),
                },
                {
                  key: "reg",
                  label: "Registry",
                  children: data.registry_loaded ? (
                    <Tag color="success">已加载</Tag>
                  ) : (
                    <Tag color="warning">未加载</Tag>
                  ),
                },
                {
                  key: "ts",
                  label: "时间",
                  children: new Date().toLocaleString("zh-CN"),
                },
              ]}
            />
          </Card>
        </>
      )}

      {!loading && !data && !error && (
        <div className="empty-tip">
          <Title level={4} type="secondary">
            暂无数据
          </Title>
        </div>
      )}
    </div>
  );
}
