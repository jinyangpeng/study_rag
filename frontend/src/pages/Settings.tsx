/**
 * Settings — 配置页
 *
 * 布局：
 *  - 服务端连接（baseURL）
 *  - Admin Token（鉴权启用时）
 *  - 系统信息（鉴权状态、registry 等）
 *  - 主题（dark/light 切换）
 */
import { useEffect, useState } from "react";
import {
  Settings as SettingsIcon,
  KeyRound,
  Server,
  Info,
  Sun,
  Moon,
  Eye,
  EyeOff,
  Check,
  X,
  Loader2,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { useApi } from "@/api/client";
import type { HealthDetailed } from "@/api/types";
import { useTheme } from "@/hooks/useTheme";
import { toast } from "sonner";

export default function Settings() {
  const { client, baseURL, setBaseURL, token, setToken, authEnabled, hasToken, refreshAuthStatus } =
    useApi();
  const { theme, setTheme } = useTheme();

  const [baseUrlInput, setBaseUrlInput] = useState(baseURL);
  const [tokenInput, setTokenInput] = useState(token);
  const [showToken, setShowToken] = useState(false);
  const [health, setHealth] = useState<HealthDetailed | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [testing, setTesting] = useState(false);

  useEffect(() => {
    setBaseUrlInput(baseURL);
  }, [baseURL]);

  useEffect(() => {
    setTokenInput(token);
  }, [token]);

  useEffect(() => {
    void (async () => {
      setHealthLoading(true);
      try {
        const h = await client.getHealthDetailed();
        setHealth(h);
      } catch {
        // 静默 — 设置页不强求健康
      } finally {
        setHealthLoading(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function saveBaseURL() {
    setBaseURL(baseUrlInput);
    toast.success("已保存 baseURL，重新探测服务端…");
    await refreshAuthStatus();
  }

  async function saveToken() {
    setToken(tokenInput);
    toast.success("Token 已保存");
    await refreshAuthStatus();
  }

  async function testConnection() {
    setTesting(true);
    try {
      const h = await client.getHealthDetailed();
      setHealth(h);
      toast.success(`连接成功：status=${h.status}`);
    } catch (e) {
      toast.error(`连接失败: ${(e as Error).message}`);
    } finally {
      setTesting(false);
    }
  }

  function clearToken() {
    setToken("");
    setTokenInput("");
    toast.success("Token 已清除");
  }

  return (
    <div className="space-y-5">
      <div>
        <h2 className="flex items-center gap-2 text-base font-semibold">
          <SettingsIcon className="size-3.5 text-accent" />
          设置
        </h2>
        <p className="text-xs text-fg-muted">
          管理服务端连接、鉴权 token、外观
        </p>
      </div>

      {/* Server connection */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <Server className="size-3.5" />
            服务端连接
          </CardTitle>
          <CardDescription>
            后端 API 的 baseURL。dev 模式建议留空（Vite proxy 转发）；生产用当前 origin。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1">
            <Label className="text-xs">API baseURL</Label>
            <div className="flex items-center gap-2">
              <Input
                value={baseUrlInput}
                onChange={(e) => setBaseUrlInput(e.target.value)}
                placeholder="https://api.example.com  或留空"
                className="font-mono"
              />
              <Button size="sm" onClick={saveBaseURL}>
                <Check className="size-3.5" />
                保存
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={testConnection}
                disabled={testing}
              >
                {testing ? (
                  <Loader2 className="size-3.5 animate-spin" />
                ) : (
                  <Server className="size-3.5" />
                )}
                测试
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Auth */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <KeyRound className="size-3.5" />
            鉴权 Token
          </CardTitle>
          <CardDescription>
            服务端 <code className="rounded bg-bg-tertiary px-1 text-[10px]">STUDY_RAG_ADMIN_TOKEN</code> 启用时
            必须填写。Token 仅保存在浏览器 localStorage。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center gap-2 text-xs">
            <span className="text-fg-muted">服务端鉴权状态：</span>
            {authEnabled === null ? (
              <Badge variant="muted">探测中…</Badge>
            ) : authEnabled ? (
              <Badge variant="warning">已启用</Badge>
            ) : (
              <Badge variant="success">未启用（免鉴权）</Badge>
            )}
            {hasToken && <Badge variant="default">已配置 token</Badge>}
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Bearer token</Label>
            <div className="flex items-center gap-2">
              <div className="relative flex-1">
                <Input
                  type={showToken ? "text" : "password"}
                  value={tokenInput}
                  onChange={(e) => setTokenInput(e.target.value)}
                  placeholder="sk-..."
                  className="pr-9 font-mono"
                />
                <button
                  type="button"
                  onClick={() => setShowToken((v) => !v)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-fg-muted hover:text-fg"
                >
                  {showToken ? <EyeOff className="size-3.5" /> : <Eye className="size-3.5" />}
                </button>
              </div>
              <Button size="sm" onClick={saveToken} disabled={!tokenInput}>
                <Check className="size-3.5" />
                保存
              </Button>
              {hasToken && (
                <Button variant="ghost" size="sm" onClick={clearToken}>
                  <X className="size-3.5" />
                  清除
                </Button>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Appearance */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            {theme === "dark" ? <Moon className="size-3.5" /> : <Sun className="size-3.5" />}
            外观
          </CardTitle>
          <CardDescription>Linear 风格，深色 / 浅色二选一（选择会保存到本地）</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between rounded border border-border-subtle bg-bg-tertiary px-3 py-2">
            <div>
              <Label className="text-xs">深色模式</Label>
              <div className="text-[10px] text-fg-muted">
                当前：{theme === "dark" ? "Dark" : "Light"}
              </div>
            </div>
            <Switch
              checked={theme === "dark"}
              onCheckedChange={(v) => setTheme(v ? "dark" : "light")}
            />
          </div>
        </CardContent>
      </Card>

      {/* System info */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <Info className="size-3.5" />
            系统信息
          </CardTitle>
        </CardHeader>
        <CardContent>
          {healthLoading ? (
            <div className="space-y-2">
              <Skeleton className="h-4" />
              <Skeleton className="h-4" />
              <Skeleton className="h-4" />
            </div>
          ) : !health ? (
            <div className="text-xs text-fg-muted">
              探测不到服务端，点击「测试」按钮重试
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              <InfoRow label="Status" value={health.status} />
              <InfoRow label="Auth Enabled" value={String(health.auth_enabled)} />
              <InfoRow label="Registry Loaded" value={String(health.registry_loaded)} />
              <InfoRow label="Total KBs" value={String(health.kbs_total)} />
              <InfoRow label="Enabled KBs" value={String(health.kbs_enabled)} />
              <InfoRow label="Embedders" value={String(health.embedders)} />
              <InfoRow label="Rerankers" value={String(health.rerankers)} />
              <InfoRow
                label="Admin ratelimit"
                value={`${health.ratelimit.admin.tracked_keys} keys`}
              />
              <InfoRow
                label="Search ratelimit"
                value={`${health.ratelimit.search.tracked_keys} keys`}
              />
            </div>
          )}
        </CardContent>
      </Card>

      <Separator />

      <div className="text-center text-[10px] text-fg-muted">
        study_rag admin UI · v0.1.0
      </div>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between rounded border border-border-subtle bg-bg-tertiary px-3 py-1.5 text-xs">
      <span className="text-fg-muted">{label}</span>
      <span className="font-mono text-fg">{value}</span>
    </div>
  );
}
