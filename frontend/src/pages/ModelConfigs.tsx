/**
 * ModelConfigs — 模型配置管理页
 *
 * 布局：
 *  - Tab：Embeddings / Rerankers
 *  - 每个 Tab：表格（name / provider / model / 关键参数 / 状态 / 操作）+ 新建按钮
 *  - 操作：编辑（弹窗）/ 删除（确认）
 *
 * 配置改完需重启服务生效（运行时实例不热更新，避免影响在服务的 KB）。
 */
import { useEffect, useState } from "react";
import {
  Boxes,
  Sparkles,
  Scissors,
  Plus,
  Pencil,
  Trash2,
  Loader2,
  CheckCircle2,
  Clock,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { EmptyState } from "@/components/shared/EmptyState";
import { ErrorState } from "@/components/shared/ErrorState";
import { useApi } from "@/api/client";
import type {
  EmbedderConfigItem,
  ParserConfigItem,
  RerankerConfigItem,
} from "@/api/types";
import ModelConfigDialog from "@/components/ModelConfigDialog";
import { toast } from "sonner";

type Tab = "embedders" | "rerankers" | "parsers";

export default function ModelConfigs() {
  const { client } = useApi();
  const [tab, setTab] = useState<Tab>("embedders");

  const [embedders, setEmbedders] = useState<EmbedderConfigItem[]>([]);
  const [rerankers, setRerankers] = useState<RerankerConfigItem[]>([]);
  const [parsers, setParsers] = useState<ParserConfigItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [dialogOpen, setDialogOpen] = useState(false);
  const [editTarget, setEditTarget] = useState<
    EmbedderConfigItem | RerankerConfigItem | ParserConfigItem | null
  >(null);
  const [deleteTarget, setDeleteTarget] = useState<
    { name: string; kind: Tab } | null
  >(null);
  const [deleting, setDeleting] = useState(false);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const [e, r, p] = await Promise.all([
        client.listEmbedderConfigs(),
        client.listRerankerConfigs(),
        client.listParserConfigs(),
      ]);
      setEmbedders(e);
      setRerankers(r);
      setParsers(p);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  function openCreate() {
    setEditTarget(null);
    setDialogOpen(true);
  }

  function openEdit(item: EmbedderConfigItem | RerankerConfigItem | ParserConfigItem) {
    setEditTarget(item);
    setDialogOpen(true);
  }

  async function confirmDelete() {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      if (deleteTarget.kind === "embedders") {
        await client.deleteEmbedderConfig(deleteTarget.name);
      } else if (deleteTarget.kind === "rerankers") {
        await client.deleteRerankerConfig(deleteTarget.name);
      } else {
        await client.deleteParserConfig(deleteTarget.name);
      }
      toast.success(`已删除 ${deleteTarget.name}`);
      setDeleteTarget(null);
      await load();
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setDeleting(false);
    }
  }

  return (
    <div className="space-y-5">
      <div>
        <h2 className="flex items-center gap-2 text-base font-semibold">
          <Boxes className="size-3.5 text-accent" />
          模型配置
        </h2>
        <p className="text-xs text-fg-muted">
          管理 Embedding、Reranker 模型与 Parser 分块配置（写 YAML）。修改后需重启服务生效。
        </p>
      </div>

      {/* 配置作用说明 */}
      <Card>
        <CardContent className="space-y-2 p-4 text-xs text-fg-secondary">
          <div>
            <span className="font-medium text-fg">
              <Boxes className="mr-1 inline size-3" />
              Embedding 模型
            </span>
            <span className="ml-2 text-fg-muted">— 把文本转成向量，用于向量检索召回。</span>
          </div>
          <div className="ml-4 text-fg-muted">
            作用：写入文档时切块向量化入库；检索时把 query 向量化后在向量库做相似度召回。
            <code className="mx-1 rounded bg-bg-tertiary px-1 font-mono text-[10px]">dimension</code>
            必须与向量库 collection 维度一致；
            <code className="mx-1 rounded bg-bg-tertiary px-1 font-mono text-[10px]">batch_size</code>
            控制批量推理大小。
          </div>
          <Separator className="my-2" />
          <div>
            <span className="font-medium text-fg">
              <Sparkles className="mr-1 inline size-3" />
              Reranker 模型
            </span>
            <span className="ml-2 text-fg-muted">— 对向量召回的结果做二次重排，提升精度。</span>
          </div>
          <div className="ml-4 text-fg-muted">
            作用：检索时在 embedding 召回的候选集上用 cross-encoder 重新打分排序，
            过滤到
            <code className="mx-1 rounded bg-bg-tertiary px-1 font-mono text-[10px]">top_k</code>
            条返回。不影响入库，只影响检索结果质量。
            <code className="mx-1 rounded bg-bg-tertiary px-1 font-mono text-[10px]">protocol</code>
            仅 provider=http 时生效（tei/jina/...）。
          </div>
          <Separator className="my-2" />
          <div>
            <span className="font-medium text-fg">
              <Scissors className="mr-1 inline size-3" />
              Parser 分块配置
            </span>
            <span className="ml-2 text-fg-muted">— 把文档切成多个 chunk 入库，影响检索粒度。</span>
          </div>
          <div className="ml-4 text-fg-muted">
            作用：添加文档时按
            <code className="mx-1 rounded bg-bg-tertiary px-1 font-mono text-[10px]">strategy</code>
            （whole/sentence/semantic/token）切分，每块独立向量化。
            <code className="mx-1 rounded bg-bg-tertiary px-1 font-mono text-[10px]">chunk_size</code>
            越大粒度越粗召回越多上下文，越小粒度越精但可能丢上下文。
            semantic 策略需 embed_model。
          </div>
          <div className="mt-2 rounded border border-border-subtle bg-bg-tertiary p-2 text-[11px]">
            <span className="font-medium">入库链路：</span>
            文档 → parser 切块 → embedding 向量化 → 向量库
            <span className="mx-2 text-fg-muted">|</span>
            <span className="font-medium">检索链路：</span>
            query → embedding → 向量库召回 Top K → reranker 重排过滤到 top_k → 返回
          </div>
        </CardContent>
      </Card>

      <Tabs value={tab} onValueChange={(v) => setTab(v as Tab)}>
        <TabsList>
          <TabsTrigger value="embedders" className="text-xs">
            <Boxes className="mr-1.5 size-3.5" />
            Embeddings
            <Badge variant="muted" className="ml-1.5 text-[10px]">
              {embedders.length}
            </Badge>
          </TabsTrigger>
          <TabsTrigger value="rerankers" className="text-xs">
            <Sparkles className="mr-1.5 size-3.5" />
            Rerankers
            <Badge variant="muted" className="ml-1.5 text-[10px]">
              {rerankers.length}
            </Badge>
          </TabsTrigger>
          <TabsTrigger value="parsers" className="text-xs">
            <Scissors className="mr-1.5 size-3.5" />
            Parsers
            <Badge variant="muted" className="ml-1.5 text-[10px]">
              {parsers.length}
            </Badge>
          </TabsTrigger>
        </TabsList>

        {/* Embedders */}
        <TabsContent value="embedders" className="mt-4">
          <Card>
            <CardContent className="p-4">
              <div className="mb-3 flex items-center justify-between">
                <span className="text-xs text-fg-muted">
                  embeddings.yaml 中的配置
                </span>
                <Button size="sm" onClick={openCreate}>
                  <Plus className="size-3.5" />
                  新建
                </Button>
              </div>
              <EmbedderTable
                list={embedders}
                loading={loading}
                error={error}
                onEdit={openEdit}
                onDelete={(name) => setDeleteTarget({ name, kind: "embedders" })}
                onRetry={load}
              />
            </CardContent>
          </Card>
        </TabsContent>

        {/* Rerankers */}
        <TabsContent value="rerankers" className="mt-4">
          <Card>
            <CardContent className="p-4">
              <div className="mb-3 flex items-center justify-between">
                <span className="text-xs text-fg-muted">
                  reranker.yaml 中的配置
                </span>
                <Button
                  size="sm"
                  onClick={() => {
                    setTab("rerankers");
                    openCreate();
                  }}
                >
                  <Plus className="size-3.5" />
                  新建
                </Button>
              </div>
              <RerankerTable
                list={rerankers}
                loading={loading}
                error={error}
                onEdit={openEdit}
                onDelete={(name) => setDeleteTarget({ name, kind: "rerankers" })}
                onRetry={load}
              />
            </CardContent>
          </Card>
        </TabsContent>

        {/* Parsers */}
        <TabsContent value="parsers" className="mt-4">
          <Card>
            <CardContent className="p-4">
              <div className="mb-3 flex items-center justify-between">
                <span className="text-xs text-fg-muted">
                  llamaindex.yaml 中的 parser 配置
                </span>
                <Button
                  size="sm"
                  onClick={() => {
                    setTab("parsers");
                    openCreate();
                  }}
                >
                  <Plus className="size-3.5" />
                  新建
                </Button>
              </div>
              <ParserTable
                list={parsers}
                loading={loading}
                error={error}
                onEdit={openEdit}
                onDelete={(name) => setDeleteTarget({ name, kind: "parsers" })}
                onRetry={load}
              />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* 新增/编辑弹窗 */}
      <ModelConfigDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        kind={
          tab === "embedders"
            ? "embedder"
            : tab === "rerankers"
              ? "reranker"
              : "parser"
        }
        initial={editTarget}
        onSaved={load}
      />

      {/* 删除确认 */}
      <Dialog
        open={deleteTarget !== null}
        onOpenChange={(v) => !v && setDeleteTarget(null)}
      >
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle>删除配置</DialogTitle>
            <DialogDescription>
              确定删除{" "}
              <code className="rounded bg-bg-tertiary px-1 font-mono text-xs">
                {deleteTarget?.name}
              </code>
              ？此操作不可恢复。
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteTarget(null)}>
              取消
            </Button>
            <Button
              variant="danger"
              onClick={confirmDelete}
              disabled={deleting}
            >
              {deleting && <Loader2 className="size-3.5 animate-spin" />}
              <Trash2 className="size-3.5" />
              删除
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

// ===== Embedder 表格 =====

function EmbedderTable({
  list,
  loading,
  error,
  onEdit,
  onDelete,
  onRetry,
}: {
  list: EmbedderConfigItem[];
  loading: boolean;
  error: string | null;
  onEdit: (item: EmbedderConfigItem) => void;
  onDelete: (name: string) => void;
  onRetry: () => void;
}) {
  if (loading) {
    return <Skeleton className="h-40" />;
  }
  if (error) {
    return <ErrorState message={error} onRetry={onRetry} />;
  }
  if (list.length === 0) {
    return (
      <EmptyState
        title="暂无 Embedder 配置"
        description="点击「新建」添加第一个 embedding 模型"
        icon={Boxes}
      />
    );
  }
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="w-[160px]">配置名</TableHead>
          <TableHead className="w-[100px]">Provider</TableHead>
          <TableHead>模型</TableHead>
          <TableHead className="w-[80px]">维度</TableHead>
          <TableHead className="w-[70px]">状态</TableHead>
          <TableHead className="w-[100px] text-right">操作</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {list.map((e) => (
          <TableRow key={e.name}>
            <TableCell className="font-mono text-xs">{e.name}</TableCell>
            <TableCell>
              <Badge variant="muted" className="font-mono text-[10px]">
                {e.provider}
              </Badge>
            </TableCell>
            <TableCell className="font-mono text-xs text-fg-secondary">
              {e.model_name || "-"}
            </TableCell>
            <TableCell className="font-mono text-xs">{e.dimension}</TableCell>
            <TableCell>
              {e.loaded ? (
                <span className="flex items-center gap-1 text-[11px] text-success">
                  <CheckCircle2 className="size-3" />
                  已加载
                </span>
              ) : (
                <span className="flex items-center gap-1 text-[11px] text-fg-muted">
                  <Clock className="size-3" />
                  按需
                </span>
              )}
            </TableCell>
            <TableCell className="text-right">
              <div className="flex justify-end gap-1">
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-7"
                  onClick={() => onEdit(e)}
                >
                  <Pencil className="size-3.5" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-7 text-danger hover:text-danger"
                  onClick={() => onDelete(e.name)}
                >
                  <Trash2 className="size-3.5" />
                </Button>
              </div>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

// ===== Reranker 表格 =====

function RerankerTable({
  list,
  loading,
  error,
  onEdit,
  onDelete,
  onRetry,
}: {
  list: RerankerConfigItem[];
  loading: boolean;
  error: string | null;
  onEdit: (item: RerankerConfigItem) => void;
  onDelete: (name: string) => void;
  onRetry: () => void;
}) {
  if (loading) {
    return <Skeleton className="h-40" />;
  }
  if (error) {
    return <ErrorState message={error} onRetry={onRetry} />;
  }
  if (list.length === 0) {
    return (
      <EmptyState
        title="暂无 Reranker 配置"
        description="点击「新建」添加第一个 reranker 模型"
        icon={Sparkles}
      />
    );
  }
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="w-[180px]">配置名</TableHead>
          <TableHead className="w-[90px]">Provider</TableHead>
          <TableHead className="w-[90px]">Protocol</TableHead>
          <TableHead>模型</TableHead>
          <TableHead className="w-[60px]">TopK</TableHead>
          <TableHead className="w-[70px]">状态</TableHead>
          <TableHead className="w-[100px] text-right">操作</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {list.map((r) => (
          <TableRow key={r.name}>
            <TableCell className="font-mono text-xs">{r.name}</TableCell>
            <TableCell>
              <Badge variant="muted" className="font-mono text-[10px]">
                {r.provider}
              </Badge>
            </TableCell>
            <TableCell className="font-mono text-[11px] text-fg-secondary">
              {r.protocol || "-"}
            </TableCell>
            <TableCell className="font-mono text-xs text-fg-secondary">
              {r.model_name || "-"}
            </TableCell>
            <TableCell className="font-mono text-xs">{r.top_k}</TableCell>
            <TableCell>
              {r.loaded ? (
                <span className="flex items-center gap-1 text-[11px] text-success">
                  <CheckCircle2 className="size-3" />
                  已加载
                </span>
              ) : (
                <span className="flex items-center gap-1 text-[11px] text-fg-muted">
                  <Clock className="size-3" />
                  按需
                </span>
              )}
            </TableCell>
            <TableCell className="text-right">
              <div className="flex justify-end gap-1">
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-7"
                  onClick={() => onEdit(r)}
                >
                  <Pencil className="size-3.5" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-7 text-danger hover:text-danger"
                  onClick={() => onDelete(r.name)}
                >
                  <Trash2 className="size-3.5" />
                </Button>
              </div>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

// ===== Parser 表格 =====

function ParserTable({
  list,
  loading,
  error,
  onEdit,
  onDelete,
  onRetry,
}: {
  list: ParserConfigItem[];
  loading: boolean;
  error: string | null;
  onEdit: (item: ParserConfigItem) => void;
  onDelete: (name: string) => void;
  onRetry: () => void;
}) {
  if (loading) {
    return <Skeleton className="h-40" />;
  }
  if (error) {
    return <ErrorState message={error} onRetry={onRetry} />;
  }
  if (list.length === 0) {
    return (
      <EmptyState
        title="暂无 Parser 配置"
        description="点击「新建」添加分块策略"
        icon={Scissors}
      />
    );
  }
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="w-[160px]">配置名</TableHead>
          <TableHead className="w-[100px]">策略</TableHead>
          <TableHead className="w-[80px]">块大小</TableHead>
          <TableHead className="w-[70px]">重叠</TableHead>
          <TableHead>段落分隔符</TableHead>
          <TableHead className="w-[80px]">语义参数</TableHead>
          <TableHead className="w-[100px] text-right">操作</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {list.map((p) => (
          <TableRow key={p.name}>
            <TableCell className="font-mono text-xs">{p.name}</TableCell>
            <TableCell>
              <Badge variant="muted" className="font-mono text-[10px]">
                {p.strategy}
              </Badge>
            </TableCell>
            <TableCell className="font-mono text-xs">{p.chunk_size}</TableCell>
            <TableCell className="font-mono text-xs">{p.chunk_overlap}</TableCell>
            <TableCell className="font-mono text-[11px] text-fg-muted">
              {p.paragraph_separator === "\n\n"
                ? "\\n\\n"
                : p.paragraph_separator === "\n"
                  ? "\\n"
                  : p.paragraph_separator}
            </TableCell>
            <TableCell className="font-mono text-[11px] text-fg-muted">
              {p.strategy === "semantic"
                ? `buf=${p.buffer_size ?? "-"}, ${p.breakpoint_percentile_threshold ?? "-"}%`
                : "-"}
            </TableCell>
            <TableCell className="text-right">
              <div className="flex justify-end gap-1">
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-7"
                  onClick={() => onEdit(p)}
                >
                  <Pencil className="size-3.5" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-7 text-danger hover:text-danger"
                  onClick={() => onDelete(p.name)}
                >
                  <Trash2 className="size-3.5" />
                </Button>
              </div>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
