import { useState, useEffect, type ReactNode } from "react";
import { Layout, Menu, Typography, Space, Tag, Alert, Button, theme } from "antd";
import {
  DashboardOutlined,
  DatabaseOutlined,
  SearchOutlined,
  LineChartOutlined,
  SettingOutlined,
  ApiOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useApi } from "../api/client";

const { Header, Sider, Content } = Layout;

interface MainLayoutProps {
  children: ReactNode;
}

export default function MainLayout({ children }: MainLayoutProps) {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const { hasToken, authEnabled, proxyError, clearProxyError, refreshAuthStatus } =
    useApi();
  const { token: themeToken } = theme.useToken();

  // 选中菜单项
  const selectedKey = (() => {
    if (location.pathname.startsWith("/dashboard")) return "/dashboard";
    if (location.pathname.startsWith("/kbs")) return "/kbs";
    if (location.pathname.startsWith("/search")) return "/search";
    if (location.pathname.startsWith("/metrics")) return "/metrics";
    if (location.pathname.startsWith("/settings")) return "/settings";
    return "/dashboard";
  })();

  useEffect(() => {
    // 让 URL 变化时滚动到顶
    window.scrollTo(0, 0);
  }, [location.pathname]);

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Sider
        collapsible
        collapsed={collapsed}
        onCollapse={setCollapsed}
        width={220}
        style={{ background: themeToken.colorBgContainer }}
      >
        <div
          style={{
            height: 56,
            display: "flex",
            alignItems: "center",
            justifyContent: collapsed ? "center" : "flex-start",
            padding: collapsed ? 0 : "0 16px",
            borderBottom: `1px solid ${themeToken.colorBorderSecondary}`,
          }}
        >
          <DatabaseOutlined
            style={{ fontSize: 22, color: themeToken.colorPrimary }}
          />
          {!collapsed && (
            <Typography.Title
              level={5}
              style={{ margin: "0 0 0 10px", color: themeToken.colorPrimary }}
            >
              study_rag
            </Typography.Title>
          )}
        </div>
        <Menu
          mode="inline"
          selectedKeys={[selectedKey]}
          onClick={({ key }) => navigate(key)}
          items={[
            { key: "/dashboard", icon: <DashboardOutlined />, label: "系统状态" },
            { key: "/kbs", icon: <DatabaseOutlined />, label: "知识库" },
            { key: "/search", icon: <SearchOutlined />, label: "检索测试" },
            { key: "/metrics", icon: <LineChartOutlined />, label: "Metrics" },
            { key: "/settings", icon: <SettingOutlined />, label: "设置" },
          ]}
        />
      </Sider>
      <Layout>
        <Header
          style={{
            background: themeToken.colorBgContainer,
            padding: "0 24px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            borderBottom: `1px solid ${themeToken.colorBorderSecondary}`,
          }}
        >
          <Space>
            <Typography.Text strong style={{ fontSize: 16 }}>
              {(() => {
                if (selectedKey === "/dashboard") return "系统状态";
                if (selectedKey === "/kbs") return "知识库管理";
                if (selectedKey === "/search") return "检索测试";
                if (selectedKey === "/metrics") return "Prometheus Metrics";
                if (selectedKey === "/settings") return "设置";
                return "study_rag Admin";
              })()}
            </Typography.Text>
          </Space>
          <Space size="middle">
            <AuthStatusTag hasToken={hasToken} authEnabled={authEnabled} />
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              <Link
                to="/settings"
                style={{ color: "inherit", textDecoration: "underline" }}
              >
                配置
              </Link>
            </Typography.Text>
          </Space>
        </Header>
        <Content
          style={{
            margin: 16,
            padding: 24,
            background: themeToken.colorBgContainer,
            borderRadius: 8,
            minHeight: 280,
            overflow: "auto",
          }}
        >
          {/* 前端代理不通 / 后端 5xx 时的顶部提示条。
              显式提示用户检查 vite.config.ts 的 proxy target。 */}
          {proxyError && (
            <Alert
              type="error"
              showIcon
              closable
              onClose={clearProxyError}
              message="前端代理不通（Vite Proxy / 后端不可达）"
              description={
                <div style={{ whiteSpace: "pre-wrap" }}>{proxyError}</div>
              }
              action={
                <Button
                  size="small"
                  icon={<ReloadOutlined />}
                  onClick={() => void refreshAuthStatus()}
                >
                  重试探测
                </Button>
              }
              style={{ marginBottom: 16 }}
            />
          )}
          {children}
        </Content>
      </Layout>
    </Layout>
  );
}

interface AuthStatusTagProps {
  hasToken: boolean;
  authEnabled: boolean | null;
}

/**
 * 鉴权状态指示器：
 *   - null（探测中）       → "检测中..."
 *   - authEnabled=false   → "免鉴权" (info/blue)
 *   - authEnabled=true +
 *     hasToken=false       → "需要 Token" (warning/orange)
 *   - authEnabled=true +
 *     hasToken=true        → "已认证" (success/green)
 */
function AuthStatusTag({ hasToken, authEnabled }: AuthStatusTagProps) {
  if (authEnabled === null) {
    return (
      <Tag icon={<ApiOutlined />} color="default">
        检测中...
      </Tag>
    );
  }
  if (!authEnabled) {
    return (
      <Tag icon={<ApiOutlined />} color="blue">
        免鉴权
      </Tag>
    );
  }
  if (!hasToken) {
    return (
      <Tag icon={<ApiOutlined />} color="warning">
        需要 Token
      </Tag>
    );
  }
  return (
    <Tag icon={<ApiOutlined />} color="success">
      已认证
    </Tag>
  );
}
