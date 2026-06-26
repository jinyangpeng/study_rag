/**
 * SearchTest — 检索测试页
 *
 * 布局：
 *  - 顶部：KB 选择 + 策略选择 + 召回数 + rerank switch + query input + Search 按钮
 *  - 策略参数面板：根据选择的策略显示可调参数（折叠式，每个参数有说明）
 *  - 结果列表：每条 hit 显示 score + doc_id/chunk_id + 文本 + metadata
 *  - 底部：耗时统计 + 检索元信息
 *
 * Top K 语义：
 *  - Top K = embedding/BM25 召回数（向量检索或关键词检索返回的候选数量）
 *  - 启用 rerank 时，最终返回数由 reranker 配置的 top_k 决定（如 local_bge_reranker_base=3）
 *  - 未启用 rerank 时，最终返回数 = Top K
 */
import { useEffect, useState } from "react";
import {
  Search as SearchIcon,
  Sparkles,
  Loader2,
  Clock,
  FileText,
  Hash,
  ListOrdered,
  SlidersHorizontal,
  ChevronDown,
  ChevronUp,
  Info,
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
  SelectItemText,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { EmptyState } from "@/components/shared/EmptyState";
import { ErrorState } from "@/components/shared/ErrorState";
import { useApi } from "@/api/client";
import type {
  KnowledgeBaseSummary,
  RerankerInfo,
  SearchResponse,
  RetrievalStrategyInfo,
} from "@/api/types";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

export default function SearchTest() {
  const { client } = useApi();
  const [kbs, setKbs] = useState<KnowledgeBaseSummary[]>([]);
  const [kbId, setKbId] = useState<string>("");
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState<number | null>(null);
  const [useRerank, setUseRerank] = useState(true);
  const [rerankers, setRerankers] = useState<RerankerInfo[]>([]);
  const [rerankerName, setRerankerName] = useState<string>("");
  const [strategies, setStrategies] = useState<RetrievalStrategyInfo[]>([]);
  const [strategy, setStrategy] = useState<string>("");
  const [strategyParams, setStrategyParams] = useState<Record<string, unknown>>({});
  const [showParams, setShowParams] = useState(false);
  const [searching, setSearching] = useState(false);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [k, r, s] = await Promise.all([
          client.listKBs(),
          client.listRerankers(),
          client.listRetrievalStrategies(),
        ]);
        setKbs(k);
        setRerankers(r);
        setStrategies(s);
        if (k.length > 0) setKbId(k[0].kb_id);
      } catch (e) {
        toast.error((e as Error).message);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    setRerankerName("");
  }, [kbId]);

  useEffect(() => {
    if (strategy) {
      const s = strategies.find((s) => s.name === strategy);
      if (s) setStrategyParams(s.params);
    } else {
      setStrategyParams({});
    }
  }, [strategy, strategies]);

  const selectedKb = kbs.find((k) => k.kb_id === kbId);
  const kbDefaultStrategy = selectedKb?.retrieval_strategy ?? null;
  const effectiveStrategy = strategy || kbDefaultStrategy || "dense";

  // 选中的 reranker 信息（用于显示其配置的 top_k）
  const effectiveRerankerName = useRerank && rerankerName ? rerankerName : selectedKb?.reranker ?? null;
  const effectiveReranker = rerankers.find((r) => r.name === effectiveRerankerName);

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
        reranker_name: useRerank && rerankerName ? rerankerName : null,
        strategy: strategy || null,
        strategy_params: Object.keys(strategyParams).length > 0 ? strategyParams : null,
      });
      setResult(r);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSearching(false);
    }
  }

  const kbDefaultReranker = selectedKb?.reranker ?? null;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div>
        <h2 className="text-base font-semibold">检索测试</h2>
        <p className="text-xs text-fg-muted">
          在知识库中检索，支持 Dense / Sparse / Hybrid / Milvus BM25 多种策略
        </p>
      </div>

      {/* Controls */}
      <Card>
        <CardContent className="space-y-3 p-4">
          {/* Row 1: KB + Strategy + TopK */}
          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            {/* 知识库 */}
            <div className="space-y-1">
              <div className="flex items-center gap-1">
                <Label className="text-xs">知识库 (KB)</Label>
                {kbId && (() => {
                  const k = kbs.find((x) => x.kb_id === kbId);
                  if (!k) return null;
                  return (
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="size-3 cursor-help text-fg-muted" />
                        </TooltipTrigger>
                        <TooltipContent className="max-w-xs">
                          <p className="text-xs">name: {k.name}</p>
                          <p className="text-xs">department: {k.department ?? "-"}</p>
                          <p className="text-xs">chunks: {k.chunk_count}</p>
                          <p className="text-xs">strategy: {k.retrieval_strategy ?? "（跟随全局）"}</p>
                          <p className="text-xs">reranker: {k.reranker ?? "（无）"}</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  );
                })()}
              </div>
              <Select value={kbId} onValueChange={setKbId}>
                <SelectTrigger>
                  <SelectValue placeholder="选择知识库" />
                </SelectTrigger>
                <SelectContent>
                  {kbs.map((k) => (
                    <SelectItem key={k.kb_id} value={k.kb_id}>
                      <div className="flex flex-col gap-0.5 py-0.5">
                        {/* 主值：纯文本（trigger 显示） */}
                        <SelectItemText>{k.kb_id}</SelectItemText>
                        {/* 描述（仅展开时显示） */}
                        <span className="text-[10px] text-fg-muted">
                          {k.name} · {k.chunk_count} chunks
                          {k.retrieval_strategy && ` · ${k.retrieval_strategy}`}
                        </span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* 检索策略 */}
            <div className="space-y-1">
              <div className="flex items-center gap-1">
                <Label className="text-xs">检索策略</Label>
                {effectiveStrategy && (() => {
                  const s = strategies.find((x) => x.name === effectiveStrategy);
                  if (!s) return null;
                  return (
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="size-3 cursor-help text-fg-muted" />
                        </TooltipTrigger>
                        <TooltipContent className="max-w-xs">
                          <p className="text-xs">{s.description}</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  );
                })()}
              </div>
              <Select
                value={strategy}
                onValueChange={(v) => setStrategy(v === "__kb_default__" ? "" : v)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="跟随 KB 默认" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__kb_default__">
                    <div className="flex flex-col gap-0.5 py-0.5">
                      <SelectItemText>跟随 KB 默认</SelectItemText>
                      {kbDefaultStrategy && (
                        <span className="text-[10px] text-fg-muted">
                          当前: {kbDefaultStrategy}
                        </span>
                      )}
                    </div>
                  </SelectItem>
                  <Separator className="my-1" />
                  {strategies.map((s) => (
                    <SelectItem key={s.name} value={s.name}>
                      <div className="flex flex-col gap-0.5 py-0.5">
                        {/* 主值：策略名纯文本（trigger 显示） */}
                        <div className="flex items-center gap-2">
                          <SelectItemText>{s.name}</SelectItemText>
                          {s.is_default && (
                            <Badge variant="success" className="font-mono text-[9px]">
                              全局默认
                            </Badge>
                          )}
                        </div>
                        {/* 描述（仅展开时显示） */}
                        <span className="text-[10px] text-fg-muted">
                          {s.description}
                        </span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Top K（召回数） */}
            <div className="space-y-1">
              <div className="flex items-center gap-1">
                <Label className="text-xs">Top K（召回数）</Label>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Info className="size-3 cursor-help text-fg-muted" />
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs">
                      <p className="text-xs">
                        <strong>召回数</strong>：embedding/BM25 检索返回的候选数量。
                      </p>
                      <p className="mt-1 text-xs">
                        启用 rerank 时，最终返回数由 reranker 配置的 top_k 决定
                        {effectiveReranker && (
                          <>（当前 {effectiveReranker.name} = {effectiveReranker.top_k}）</>
                        )}
                        ，而非此值。
                      </p>
                      <p className="mt-1 text-xs">
                        未启用 rerank 时，最终返回数 = Top K。
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <Input
                type="number"
                min={1}
                max={50}
                value={topK ?? 5}
                onChange={(e) => {
                  const v = e.target.value;
                  setTopK(v === "" ? null : Number(v) || null);
                }}
                placeholder="5"
                className="font-mono"
              />
              {useRerank && effectiveReranker && (
                <p className="text-[10px] text-fg-muted">
                  rerank 后返回 {effectiveReranker.top_k} 条（由 reranker 配置决定）
                </p>
              )}
            </div>
          </div>

          {/* Row 2: Rerank + Strategy Params Toggle */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Label className="text-xs">Rerank 重排</Label>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className="flex h-8 items-center justify-between rounded border border-border bg-bg-tertiary px-3 cursor-help">
                      <span className="flex items-center gap-1.5 text-xs text-fg-secondary">
                        <Sparkles className="size-3" />
                        {useRerank ? "启用" : "关闭"}
                      </span>
                      <Switch checked={useRerank} onCheckedChange={setUseRerank} />
                    </div>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs">
                    <p className="text-xs">
                      <strong>Rerank</strong>：用 cross-encoder 对召回结果精排，提升相关性。
                    </p>
                    <p className="mt-1 text-xs">
                      启用后：召回 Top K 条 → reranker 精排 → 返回 reranker 配置的 top_k 条。
                    </p>
                    <p className="mt-1 text-xs">
                      关闭后：直接返回 Top K 条（按检索分数排序）。
                    </p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="h-8 text-xs"
              onClick={() => setShowParams(!showParams)}
            >
              <SlidersHorizontal className="mr-1 size-3" />
              策略参数
              {showParams ? (
                <ChevronUp className="ml-1 size-3" />
              ) : (
                <ChevronDown className="ml-1 size-3" />
              )}
            </Button>
          </div>

          {/* Strategy Params Panel (Collapsible) */}
          {showParams && (
            <StrategyParamsPanel
              strategy={effectiveStrategy}
              params={strategyParams}
              onChange={setStrategyParams}
            />
          )}

          {/* Reranker Selection */}
          {useRerank && (
            <div className="space-y-1">
              <div className="flex items-center gap-1">
                <Label className="text-xs">Reranker 模型</Label>
                {rerankerName && (() => {
                  const r = rerankers.find((x) => x.name === rerankerName);
                  if (!r) return null;
                  return (
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="size-3 cursor-help text-fg-muted" />
                        </TooltipTrigger>
                        <TooltipContent className="max-w-xs">
                          <p className="text-xs">provider: {r.provider}</p>
                          <p className="text-xs">top_k: {r.top_k}（最终返回数）</p>
                          <p className="text-xs">loaded: {r.loaded ? "是" : "否"}</p>
                          <p className="text-xs">model: {r.model_name}</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  );
                })()}
              </div>
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
                    <div className="flex flex-col gap-0.5 py-0.5">
                      <SelectItemText>跟随 KB 默认</SelectItemText>
                      {kbDefaultReranker && (
                        <span className="text-[10px] text-fg-muted">
                          当前: {kbDefaultReranker}
                        </span>
                      )}
                    </div>
                  </SelectItem>
                  <Separator className="my-1" />
                  {rerankers.map((r) => (
                    <SelectItem key={r.name} value={r.name}>
                      <div className="flex flex-col gap-0.5 py-0.5">
                        {/* 主值：模型名纯文本（trigger 显示） */}
                        <SelectItemText>{r.name}</SelectItemText>
                        {/* 描述（仅展开时显示） */}
                        <div className="flex flex-wrap items-center gap-1.5 text-[10px] text-fg-muted">
                          <Badge
                            variant={r.loaded ? "success" : "secondary"}
                            className="font-mono text-[9px]"
                          >
                            {r.provider}
                          </Badge>
                          <span>top_k={r.top_k}</span>
                          {r.description && <span>· {r.description}</span>}
                          {!r.description && <span>· {r.model_name}</span>}
                        </div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          <Separator />

          {/* Query Input */}
          <div className="space-y-1">
            <Label className="text-xs">查询文本</Label>
            <div className="flex items-center gap-2">
              <div className="relative flex-1">
                <SearchIcon className="absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-fg-muted" />
                <Input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && void onSearch()}
                  placeholder="输入要检索的问题或关键词，按 Enter 检索..."
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
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-fg-muted">
            <div className="flex flex-wrap items-center gap-3">
              <span className="flex items-center gap-1">
                <ListOrdered className="size-3" />
                {result.hits.length} 条结果
              </span>
              <span className="flex items-center gap-1">
                <Clock className="size-3" />
                {result.duration_ms}ms
              </span>
              <Badge variant="outline" className="font-mono text-[10px]">
                {result.strategy || effectiveStrategy}
              </Badge>
              <span className="font-mono text-[10px] text-fg-muted">
                KB: {result.kb_id}
              </span>
              {/* 检索 meta：根据策略展示不同的调试信息 */}
              {result.meta && Object.keys(result.meta).length > 0 && (
                <RetrievalMetaBadges meta={result.meta} />
              )}
            </div>
            <span className="text-[10px] text-fg-muted">
              query: <code className="rounded bg-bg-tertiary px-1">{result.query}</code>
            </span>
          </div>

          {result.hits.length === 0 ? (
            <EmptyState
              title="未命中"
              description="没有找到匹配的内容，试试调整查询文本或更换知识库"
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
          description="选择知识库 + 输入查询文本 → 点击检索"
          icon={SearchIcon}
        />
      )}
    </div>
  );
}

/** 策略参数调节面板 — 每个参数有标签 + 输入框 + 说明文字 */
function StrategyParamsPanel({
  strategy,
  params,
  onChange,
}: {
  strategy: string;
  params: Record<string, unknown>;
  onChange: (p: Record<string, unknown>) => void;
}) {
  type ParamSpec = {
    key: string;
    label: string;
    hint: string;
    min?: number;
    max?: number;
    step?: number;
    type?: "switch";
  };
  const strategyInfo: Record<string, { label: string; desc: string; params: ParamSpec[] }> = {
    dense: {
      label: "Dense",
      desc: "向量语义检索：基于 embedding 向量相似度，擅长理解查询语义",
      params: [
        {
          key: "over_fetch_factor",
          label: "多召回倍率",
          hint: "启用 rerank 时，候选数 = Top K × 此倍率。越大 rerank 效果越好但延迟越高（默认 4）",
          min: 1,
          max: 20,
        },
      ],
    },
    sparse: {
      label: "Sparse",
      desc: "BM25 关键词检索（纯 Python 内存索引）：精确匹配关键词，适合专有名词/代码/API 名",
      params: [
        {
          key: "k1",
          label: "BM25 k1",
          hint: "词频饱和参数：越大词频影响越大（推荐 1.2~2.0，默认 1.5）",
          min: 0,
          max: 3,
          step: 0.1,
        },
        {
          key: "b",
          label: "BM25 b",
          hint: "文档长度归一化：0=不归一化，1=完全归一化（推荐 0.75）",
          min: 0,
          max: 1,
          step: 0.05,
        },
        {
          key: "use_jieba",
          label: "jieba 中文分词",
          hint: "启用 jieba 分词（需安装 jieba），关闭则用正则逐字拆分",
          type: "switch",
        },
      ],
    },
    hybrid: {
      label: "Hybrid",
      desc: "Dense + Sparse 融合（客户端 RRF）：兼顾语义理解和关键词精确匹配",
      params: [
        {
          key: "dense_weight",
          label: "Dense 权重",
          hint: "语义检索权重（0~1）。语义为主→0.7~0.8，关键词为主→0.3~0.4，均衡→0.5~0.6",
          min: 0,
          max: 1,
          step: 0.1,
        },
        {
          key: "rrf_k",
          label: "RRF 常数 k",
          hint: "Reciprocal Rank Fusion 常数：越大分数越平滑（默认 60）",
          min: 1,
          max: 200,
        },
        {
          key: "over_fetch_factor",
          label: "多召回倍率",
          hint: "Dense 路额外召回倍率，为 rerank 准备候选（默认 4）",
          min: 1,
          max: 20,
        },
      ],
    },
    sparse_milvus: {
      label: "Sparse-Milvus",
      desc: "Milvus 2.5+ 原生 BM25 全文检索：服务端分词+评分，无内存索引/无冷启动",
      params: [
        {
          key: "over_fetch_factor",
          label: "多召回倍率",
          hint: "启用 rerank 时，候选数 = Top K × 此倍率（默认 4）",
          min: 1,
          max: 20,
        },
      ],
    },
    hybrid_milvus: {
      label: "Hybrid-Milvus",
      desc: "Milvus 2.5+ 原生混合检索：Dense + BM25 服务端 RRF 融合，单次调用完成",
      params: [
        {
          key: "dense_weight",
          label: "Dense 权重",
          hint: "语义检索权重（0~1）。语义为主→0.7~0.8，关键词为主→0.3~0.4",
          min: 0,
          max: 1,
          step: 0.1,
        },
        {
          key: "rrf_k",
          label: "RRF 常数 k",
          hint: "Reciprocal Rank Fusion 常数：越大分数越平滑（默认 60）",
          min: 1,
          max: 200,
        },
        {
          key: "over_fetch_factor",
          label: "多召回倍率",
          hint: "启用 rerank 时，候选数 = Top K × 此倍率（默认 4）",
          min: 1,
          max: 20,
        },
      ],
    },
  };

  const info = strategyInfo[strategy as keyof typeof strategyInfo];
  if (!info) return null;

  return (
    <Card className="border-dashed">
      <CardContent className="p-3 space-y-2">
        <div className="flex items-start gap-2 text-xs">
          <Badge variant="outline" className="font-mono text-[10px] shrink-0">
            {info.label}
          </Badge>
          <span className="text-fg-muted">{info.desc}</span>
        </div>
        <Separator />
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          {info.params.map((p) => (
            <div key={p.key} className="space-y-1">
              <Label className="text-[10px] font-medium">{p.label}</Label>
              {p.type === "switch" ? (
                <div className="flex h-7 items-center gap-2">
                  <Switch
                    checked={params[p.key] !== false}
                    onCheckedChange={(v) => onChange({ ...params, [p.key]: v })}
                  />
                  <span className="text-[10px] text-fg-muted">
                    {params[p.key] !== false ? "开启" : "关闭"}
                  </span>
                </div>
              ) : (
                <Input
                  type="number"
                  min={p.min}
                  max={p.max}
                  step={p.step}
                  value={String(params[p.key] ?? "")}
                  onChange={(e) => onChange({ ...params, [p.key]: Number(e.target.value) })}
                  className="h-7 font-mono text-xs"
                />
              )}
              <p className="text-[9px] leading-relaxed text-fg-muted">{p.hint}</p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

/** 检索结果卡片 */
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

  // 关键元数据：node_ref_id（即 chunk_id = 向量库主键 = llama-index node_id）、
  // source（文档来源）
  const nodeRefId = hit.chunk_id;
  const source = (hit.metadata?.source as string | undefined) ?? "";

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        {/* 第一行：序号 + node_ref_id + source（同一行，UI 更紧凑） */}
        <div className="flex min-w-0 flex-1 items-center gap-2 text-xs">
          <Badge variant="outline" className="shrink-0 font-mono">
            #{index + 1}
          </Badge>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="flex shrink-0 items-center gap-1 text-fg-muted cursor-help">
                  <Hash className="size-3" />
                  <span className="font-mono text-[10px]">{nodeRefId}</span>
                </span>
              </TooltipTrigger>
              <TooltipContent>
                <p className="text-xs">node_ref_id：向量库主键（= llama-index node_id）</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
          {source && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="flex shrink-0 items-center gap-1 text-fg-muted cursor-help">
                    <FileText className="size-3" />
                    <span className="font-mono text-[10px]">{source}</span>
                  </span>
                </TooltipTrigger>
                <TooltipContent>
                  <p className="text-xs">source：文档来源（wiki / git / 手动录入等）</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-2">
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

/** 检索元信息徽章：根据策略展示 candidates_fetched / index_size / fused_count 等。 */
function RetrievalMetaBadges({
  meta,
}: {
  meta: Record<string, unknown>;
}) {
  const labels: Record<string, { label: string; title: string }> = {
    candidates_fetched: { label: "候选", title: "检索召回的候选总数（rerank 前）" },
    index_size: { label: "索引", title: "BM25 索引中的文档总数" },
    fused_count: { label: "融合", title: "RRF 融合后的结果数" },
    dense_count: { label: "Dense", title: "Dense 路检索返回数" },
    sparse_count: { label: "Sparse", title: "Sparse 路检索返回数" },
    over_fetch_factor: { label: "倍率", title: "多召回倍率（over_fetch_factor）" },
    dense_weight: { label: "权重", title: "Dense 权重（dense_weight）" },
    rrf_k: { label: "RRF k", title: "RRF 常数 k" },
    reranked: { label: "rerank", title: "是否经过 rerank 精排" },
    backend: { label: "后端", title: "检索后端实现" },
    duration_ms: { label: "耗时", title: "引擎内部耗时（毫秒）" },
  };

  return (
    <TooltipProvider>
      {Object.entries(meta).map(([k, v]) => {
        const info = labels[k];
        if (!info) return null;
        const val = typeof v === "boolean" ? (v ? "✓" : "✗") : String(v);
        return (
          <Tooltip key={k}>
            <TooltipTrigger asChild>
              <span className="flex items-center gap-0.5 font-mono text-[10px] text-fg-muted cursor-help">
                <span className="text-fg-muted/70">{info.label}:</span>
                <span className="text-fg-secondary">{val}</span>
              </span>
            </TooltipTrigger>
            <TooltipContent>
              <p className="text-xs">{info.title}</p>
            </TooltipContent>
          </Tooltip>
        );
      })}
    </TooltipProvider>
  );
}
