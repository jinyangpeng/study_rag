/**
 * Metrics — Prometheus 指标页
 *
 * 展示 /metrics 端点的原始文本（与 Prometheus / Grafana 直接对接）。
 * 同时在顶部做一下简单的 metric 解析（按行），方便快速浏览。
 */
import { useEffect, useMemo, useState } from "react";
import {
  Gauge,
  Search as SearchIcon,
  Copy,
  RefreshCw,
  ExternalLink,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { ErrorState } from "@/components/shared/ErrorState";
import { useApi } from "@/api/client";
import { toast } from "sonner";

export default function Metrics() {
  const { client } = useApi();
  const [raw, setRaw] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const text = await client.getMetricsText();
      setRaw(text);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 解析 metric 名称
  const metricNames = useMemo(() => {
    const set = new Set<string>();
    for (const line of raw.split("\n")) {
      if (!line || line.startsWith("#")) continue;
      const m = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{|$)/);
      if (m) set.add(m[1]);
    }
    return Array.from(set).sort();
  }, [raw]);

  const filteredNames = useMemo(
    () =>
      search
        ? metricNames.filter((m) =>
            m.toLowerCase().includes(search.toLowerCase())
          )
        : metricNames,
    [metricNames, search]
  );

  function copyRaw() {
    navigator.clipboard
      .writeText(raw)
      .then(() => toast.success("已复制到剪贴板"))
      .catch(() => toast.error("复制失败"));
  }

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-end justify-between">
        <div>
          <h2 className="flex items-center gap-2 text-base font-semibold">
            <Gauge className="size-3.5 text-accent" />
            Prometheus Metrics
          </h2>
          <p className="text-xs text-fg-muted">
            {raw
              ? `${raw.split("\n").length} 行 · ${metricNames.length} 个指标`
              : "从 /metrics 端点拉取原始指标"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={copyRaw} disabled={!raw}>
            <Copy className="size-3.5" />
            复制原始文本
          </Button>
          <Button variant="outline" size="sm" onClick={() => void load()}>
            <RefreshCw className="size-3.5" />
            刷新
          </Button>
          <Button asChild variant="ghost" size="sm">
            <a href="/metrics" target="_blank" rel="noreferrer">
              <ExternalLink className="size-3.5" />
              新窗口打开
            </a>
          </Button>
        </div>
      </div>

      {error && !loading ? (
        <ErrorState message={error} onRetry={() => void load()} />
      ) : (
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
          {/* Metric names list */}
          <Card className="lg:col-span-1">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle>指标列表</CardTitle>
              <Badge variant="muted" className="font-mono">
                {filteredNames.length}
              </Badge>
            </CardHeader>
            <CardContent className="space-y-2 pt-0">
              <div className="relative">
                <SearchIcon className="absolute left-2 top-1/2 size-3 -translate-y-1/2 text-fg-muted" />
                <Input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="过滤指标名..."
                  className="h-7 pl-7 text-xs"
                />
              </div>
              <ScrollArea className="h-[60vh]">
                {loading ? (
                  <div className="space-y-1">
                    {Array.from({ length: 10 }).map((_, i) => (
                      <Skeleton key={i} className="h-6" />
                    ))}
                  </div>
                ) : filteredNames.length === 0 ? (
                  <EmptyState
                    title="无匹配指标"
                    description="试试其他关键词"
                    icon={SearchIcon}
                    className="py-8"
                  />
                ) : (
                  <div className="space-y-0.5 pr-2">
                    {filteredNames.map((name) => (
                      <a
                        key={name}
                        href={`#${name}`}
                        className="block truncate rounded px-2 py-1 font-mono text-[11px] text-fg-secondary transition-colors hover:bg-bg-tertiary hover:text-fg"
                      >
                        {name}
                      </a>
                    ))}
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>

          {/* Raw text */}
          <Card className="lg:col-span-2">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle>原始输出</CardTitle>
              <span className="text-[10px] text-fg-muted">
                text/plain (Prometheus 格式)
              </span>
            </CardHeader>
            <CardContent className="pt-0">
              {loading ? (
                <div className="space-y-1">
                  {Array.from({ length: 12 }).map((_, i) => (
                    <Skeleton key={i} className="h-4" />
                  ))}
                </div>
              ) : (
                <ScrollArea className="h-[60vh] rounded border border-border-subtle bg-bg-tertiary">
                  <pre className="whitespace-pre p-3 font-mono text-[11px] leading-relaxed text-fg-secondary">
                    {raw}
                  </pre>
                </ScrollArea>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
