/**
 * SearchTest — 检索测试页
 *
 * 布局：
 *  - 顶部：KB 选择 + top_k + rerank switch + query input + Search 按钮
 *  - 结果列表：每条 hit 显示 score（高亮）+ doc_id/chunk_id + 文本
 *  - 底部：耗时统计
 */
import { useEffect, useState } from "react";
import {
  Search as SearchIcon,
  Sparkles,
  Database,
  Loader2,
  Clock,
  FileText,
  Hash,
  ListOrdered,
} from "lucide-react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { EmptyState } from "@/components/shared/EmptyState";
import { ErrorState } from "@/components/shared/ErrorState";
import { useApi } from "@/api/client";
import type {
  KnowledgeBaseSummary,
  RerankerInfo,
  SearchResponse,
} from "@/api/types";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

export default function SearchTest() {
  const { client } = useApi();
  const [kbs, setKbs] = useState<KnowledgeBaseSummary[]>([]);
  const [kbId, setKbId] = useState<string>("");
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [useRerank, setUseRerank] = useState(true);
  const [rerankers, setRerankers] = useState<RerankerInfo[]>([]);
  // reranker 选择："" = 跟随 KB 默认；其它 = 显式指定配置名
  const [rerankerName, setRerankerName] = useState<string>("");
  const [searching, setSearching] = useState(false);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [k, r] = await Promise.all([
          client.listKBs(),
          client.listRerankers(),
        ]);
        setKbs(k);
        setRerankers(r);
        if (k.length > 0) setKbId(k[0].kb_id);
      } catch (e) {
        toast.error((e as Error).message);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 切换 KB 时，reranker 选择重置为「跟随 KB 默认」
  useEffect(() => {
    setRerankerName("");
  }, [kbId]);

  async function onSearch() {
    if (!kbId) {
      toast.warning("请先选择知识库");
      return;
    }
    if (!query.trim()) {
      toast.warning("请输入查询文本");
      return;
    }
    setSearching(true);
    setError(null);
    try {
      const r = await client.search(kbId, {
        query: query.trim(),
        top_k: topK,
        use_rerank: useRerank,
        // 显式选了 reranker 时才透传；空串表示用 KB 默认
        reranker_name: useRerank && rerankerName ? rerankerName : null,
      });
      setResult(r);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSearching(false);
    }
  }

  // 当前 KB 绑定的 reranker 名（用于下拉默认提示）
  const selectedKb = kbs.find((k) => k.kb_id === kbId);
  const kbDefaultReranker = selectedKb?.reranker ?? null;

  return (
    <div className="space-y-5">
      {/* Header */}
      <div>
        <h2 className="text-base font-semibold">检索测试</h2>
        <p className="text-xs text-fg-muted">
          对知识库发起一次检索，查看 top-K 结果与分数
        </p>
      </div>

      {/* Controls */}
      <Card>
        <CardContent className="space-y-3 p-4">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            <div className="space-y-1">
              <Label className="text-xs">知识库</Label>
              <Select value={kbId} onValueChange={setKbId}>
                <SelectTrigger>
                  <SelectValue placeholder="选择 KB" />
                </SelectTrigger>
                <SelectContent>
                  {kbs.map((k) => (
                    <SelectItem key={k.kb_id} value={k.kb_id}>
                      <div className="flex items-center gap-2">
                        <Database className="size-3 text-fg-muted" />
                        <span className="font-mono">{k.kb_id}</span>
                        <span className="text-fg-muted">· {k.name}</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1">
              <Label className="text-xs">Top K</Label>
              <Input
                type="number"
                min={1}
                max={50}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value) || 5)}
                className="font-mono"
              />
            </div>
            <div className="flex flex-col space-y-1">
              <Label className="text-xs">Rerank</Label>
              <div className="flex h-8 items-center justify-between rounded border border-border bg-bg-tertiary px-3">
                <span className="flex items-center gap-1.5 text-xs text-fg-secondary">
                  <Sparkles className="size-3" />
                  {useRerank ? "启用" : "关闭"}
                </span>
                <Switch checked={useRerank} onCheckedChange={setUseRerank} />
              </div>
            </div>
          </div>
          {useRerank && (
            <div className="space-y-1">
              <Label className="text-xs">
                Reranker 模型
                <span className="ml-1 font-normal text-fg-muted">
                  （空 = 跟随 KB 默认）
                </span>
              </Label>
              <Select
                value={rerankerName}
                onValueChange={(v) => setRerankerName(v === "__kb_default__" ? "" : v)}
                disabled={!useRerank}
              >
                <SelectTrigger className="h-8">
                  <SelectValue placeholder="跟随 KB 默认" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__kb_default__">
                    <span className="text-fg-muted">
                      跟随 KB 默认
                      {kbDefaultReranker ? `（${kbDefaultReranker}）` : "（无）"}
                    </span>
                  </SelectItem>
                  <Separator className="my-1" />
                  {rerankers.map((r) => (
                      <SelectItem key={r.name} value={r.name}>
                        <div className="flex items-center gap-2">
                          <Badge
                            variant={r.loaded ? "success" : "secondary"}
                            className="font-mono text-[10px]"
                          >
                            {r.provider}
                          </Badge>
                          <span className="font-mono text-xs">{r.name}</span>
                          {!r.loaded && (
                            <span className="text-[10px] text-fg-muted">
                              按需加载
                            </span>
                          )}
                        </div>
                      </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
          <Separator />
          <div className="flex items-center gap-2">
            <div className="relative flex-1">
              <SearchIcon className="absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-fg-muted" />
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && void onSearch()}
                placeholder="输入查询文本，按 Enter 检索..."
                className="h-9 pl-8 text-sm"
              />
            </div>
            <Button
              onClick={() => void onSearch()}
              disabled={searching || !kbId}
              className="h-9"
            >
              {searching ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : (
                <SearchIcon className="size-3.5" />
              )}
              检索
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Result */}
      {error ? (
        <ErrorState message={error} onRetry={() => void onSearch()} />
      ) : searching ? (
        <div className="space-y-2">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
      ) : result ? (
        <div className="space-y-3">
          {/* Stats bar */}
          <div className="flex items-center justify-between text-xs text-fg-muted">
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1">
                <ListOrdered className="size-3" />
                {result.hits.length} hits
              </span>
              <span className="flex items-center gap-1">
                <Clock className="size-3" />
                {result.duration_ms}ms
              </span>
              <span className="font-mono text-[10px] text-fg-muted">
                KB: {result.kb_id}
              </span>
            </div>
            <span className="text-[10px] text-fg-muted">
              query: <code className="rounded bg-bg-tertiary px-1">{result.query}</code>
            </span>
          </div>

          {result.hits.length === 0 ? (
            <EmptyState
              title="未命中"
              description="没有找到匹配的 chunk，试试调整查询或 KB"
              icon={SearchIcon}
            />
          ) : (
            <div className="space-y-2">
              {result.hits.map((hit, i) => (
                <HitCard key={`${hit.doc_id}-${hit.chunk_id}-${i}`} hit={hit} index={i} />
              ))}
            </div>
          )}
        </div>
      ) : (
        <EmptyState
          title="输入查询以开始"
          description="选择 KB + 输入文本 → 检索"
          icon={SearchIcon}
        />
      )}
    </div>
  );
}

function HitCard({
  hit,
  index,
}: {
  hit: {
    doc_id: string;
    chunk_id: string;
    score: number;
    text: string;
    metadata: Record<string, unknown>;
  };
  index: number;
}) {
  const pct = Math.min(100, Math.max(0, hit.score * 100));
  const tone = pct >= 70 ? "success" : pct >= 40 ? "warning" : "danger";

  return (
    <Card>
      <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-2">
        <div className="flex items-center gap-2 text-xs">
          <Badge variant="outline" className="font-mono">
            #{index + 1}
          </Badge>
          <span className="flex items-center gap-1 text-fg">
            <FileText className="size-3" />
            <span className="font-mono text-[11px]">{hit.doc_id}</span>
          </span>
          <span className="flex items-center gap-1 text-fg-muted">
            <Hash className="size-2.5" />
            <span className="font-mono text-[10px]">{hit.chunk_id}</span>
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={cn(
              "font-mono text-[11px] font-medium",
              tone === "success" && "text-success",
              tone === "warning" && "text-warning",
              tone === "danger" && "text-danger"
            )}
          >
            {hit.score.toFixed(4)}
          </span>
          <div className="h-1.5 w-16 overflow-hidden rounded-full bg-bg-tertiary">
            <div
              className={cn(
                "h-full",
                tone === "success" && "bg-success",
                tone === "warning" && "bg-warning",
                tone === "danger" && "bg-danger"
              )}
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="rounded border border-border-subtle bg-bg-tertiary p-3 text-[11px] leading-relaxed text-fg-secondary">
          {hit.text}
        </div>
        {hit.metadata && Object.keys(hit.metadata).length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {Object.entries(hit.metadata).map(([k, v]) => (
              <Badge
                key={k}
                variant="muted"
                className="font-mono text-[10px] font-normal"
              >
                {k}={String(v)}
              </Badge>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
