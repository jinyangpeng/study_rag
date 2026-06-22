/**
 * Settings：配置 Admin Token / API Base URL。
 * - token 存 localStorage（Key: study_rag_admin_token）
 * - baseURL 存 localStorage（Key: study_rag_api_base_url）
 */

import { useState } from "react";
import {
  Card,
  Form,
  Input,
  Button,
  Space,
  Typography,
  Alert,
  App as AntdApp,
  Divider,
  Tag,
} from "antd";
import { useApi } from "../api/client";
import { useNavigate } from "react-router-dom";

const { Title, Text, Paragraph } = Typography;

export default function Settings() {
  const {
    token,
    setToken,
    baseURL,
    setBaseURL,
    authEnabled,
    refreshAuthStatus,
  } = useApi();
  const { message } = AntdApp.useApp();
  const navigate = useNavigate();
  const [form] = Form.useForm<{ token: string; base_url: string }>();
  const [showToken, setShowToken] = useState(false);

  return (
    <div style={{ maxWidth: 720 }}>
      <Title level={4}>客户端设置</Title>
      <Paragraph type="secondary">
        这里的配置只影响当前浏览器，存于 localStorage。清空浏览器数据会丢失。
      </Paragraph>

      <Card size="small" style={{ marginBottom: 16 }}>
        <Form
          form={form}
          layout="vertical"
          initialValues={{ token, base_url: baseURL }}
          onFinish={async (v) => {
            setToken(v.token || "");
            setBaseURL(v.base_url || "");
            message.success("已保存");
            // token 改了之后重探一次服务端鉴权状态
            await refreshAuthStatus();
            navigate("/dashboard");
          }}
        >
          <Form.Item
            label="Admin Token"
            name="token"
            tooltip="与服务端 STUDY_RAG_ADMIN_TOKEN 一致；未配置服务端 token 时保持空"
            extra={
              <Text type="secondary" style={{ fontSize: 12 }}>
                服务端鉴权状态：{" "}
                {authEnabled === null ? (
                  <Tag>探测中...</Tag>
                ) : authEnabled ? (
                  <Tag color="warning">已启用，必须配置</Tag>
                ) : (
                  <Tag color="default">未启用，可不配</Tag>
                )}
                {" · "}
                {token ? (
                  <Text type="success" style={{ fontSize: 12 }}>
                    本地已保存（长度 {token.length}）
                  </Text>
                ) : (
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    本地未保存
                  </Text>
                )}
              </Text>
            }
          >
            <Input.Password
              placeholder="Bearer token"
              visibilityToggle={{
                visible: showToken,
                onVisibleChange: setShowToken,
              }}
            />
          </Form.Item>

          <Form.Item
            label="API Base URL"
            name="base_url"
            tooltip="留空 = 默认（dev 走 vite proxy, prod 用当前 origin）"
            extra={
              <Text type="secondary" style={{ fontSize: 12 }}>
                例如：http://localhost:8765
              </Text>
            }
          >
            <Input placeholder="留空使用默认" />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                保存
              </Button>
              <Button
                onClick={() => {
                  setToken("");
                  setBaseURL("");
                  form.setFieldsValue({ token: "", base_url: "" });
                  message.success("已清空");
                }}
                danger
                ghost
              >
                清空
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Card>

      <Divider />

      <Card size="small" title="路由说明">
        <Paragraph style={{ fontSize: 13 }}>
          1. <Text code>GET /admin/kbs</Text> — 列出所有知识库
        </Paragraph>
        <Paragraph style={{ fontSize: 13 }}>
          2. <Text code>GET /admin/kbs/&lt;kb_id&gt;/documents</Text> — 列出 KB 下文档
        </Paragraph>
        <Paragraph style={{ fontSize: 13 }}>
          3. <Text code>POST /admin/kbs/&lt;kb_id&gt;/documents</Text> — 添加文档
        </Paragraph>
        <Paragraph style={{ fontSize: 13 }}>
          4. <Text code>POST /admin/kbs/&lt;kb_id&gt;/search</Text> — 检索
        </Paragraph>
        <Paragraph style={{ fontSize: 13 }}>
          5. <Text code>GET /health/detailed</Text> — 系统状态
        </Paragraph>
        <Paragraph style={{ fontSize: 13 }}>
          6. <Text code>GET /metrics</Text> — Prometheus metrics
        </Paragraph>
        <Paragraph style={{ fontSize: 13 }}>
          完整 OpenAPI 文档：<Text code>GET /docs</Text>
        </Paragraph>
      </Card>

      <Alert
        type="info"
        showIcon
        style={{ marginTop: 16 }}
        message="鉴权与限流"
        description={
          <div style={{ fontSize: 13 }}>
            - 服务端 <code>STUDY_RAG_ADMIN_TOKEN</code>{" "}
            <b>未设置时免鉴权</b>，前端不需要配 token
            <br />
            - 设置后必须配前端 Token（与服务端一致），否则所有 <code>/admin/*</code>{" "}
            会 401
            <br />
            - 默认限流：120 burst / 2 req·s⁻¹（Token Bucket）
            <br />
            - 触发限流会返回 429 + Retry-After 头
          </div>
        }
      />
    </div>
  );
}
