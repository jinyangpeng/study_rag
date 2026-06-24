/**
 * KnowledgeBases — 知识库管理页
 *
 * 布局：
 *  - 顶部 toolbar（搜索 + 新建按钮）
 *  - KB 卡片网格（3-4 列）
 *  - 每张卡片：name + description + 部门 + doc/chunk 统计 + 状态 + 操作
 */
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Database,
  FileText,
  Layers,
  Plus,
  Search as SearchIcon,
  MoreHorizontal,
  Trash2,
  Settings as SettingsIcon,
  Building2,
  Cpu,
  ArrowUpRight,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
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
import type { KnowledgeBaseSummary } from "@/api/types";
import AddKBDialog from "@/components/AddKBDialog";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

export default function KnowledgeBases() {
  const { client } = useApi();
  const navigate = useNavigate();
  const [list, setList] = useState<KnowledgeBaseSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [createOpen, setCreateOpen] = useState(false);
  const [editTarget, setEditTarget] = useState<KnowledgeBaseSummary | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<KnowledgeBaseSummary | null>(null);
  const [deleting, setDeleting] = useState(false);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const k = await client.listKBs();
      setList(k);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const filtered = list.filter((kb) => {
    if (!search) return true;
    const q = search.toLowerCase();
    return (
      kb.kb_id.toLowerCase().includes(q) ||
      kb.name.toLowerCase().includes(q) ||
      (kb.description ?? "").toLowerCase().includes(q) ||
      (kb.department ?? "").toLowerCase().includes(q)
    );
  });

  async function confirmDelete() {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      await client.deleteKB(deleteTarget.kb_id);
      toast.success(`KB '${deleteTarget.kb_id}' 已删除`);
      setDeleteTarget(null);
      void load();
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setDeleting(false);
    }
  }

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-base font-semibold">知识库管理</h2>
          <p className="text-xs text-fg-muted">
            {list.length > 0
              ? `共 ${list.length} 个知识库 · ${list.filter((k) => k.enabled).length} 个启用`
              : "配置 KB 的 embedding / reranker / collection"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="relative">
            <SearchIcon className="absolute left-2 top-1/2 size-3.5 -translate-y-1/2 text-fg-muted" />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="搜索 KB..."
              className="h-8 w-56 pl-7 text-xs"
            />
          </div>
          <Button size="sm" onClick={() => setCreateOpen(true)}>
            <Plus className="size-3.5" />
            新建知识库
          </Button>
        </div>
      </div>

      {error && !loading ? (
        <ErrorState message={error} onRetry={() => void load()} />
      ) : loading ? (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-44" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <EmptyState
          title={search ? "无匹配的知识库" : "还没有知识库"}
          description={
            search ? "试试其他关键词" : "点击右上角「新建知识库」开始"
          }
          icon={Database}
          action={
            !search && (
              <Button size="sm" onClick={() => setCreateOpen(true)}>
                <Plus className="size-3.5" />
                新建知识库
              </Button>
            )
          }
        />
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {filtered.map((kb) => (
            <KBCard
              key={kb.kb_id}
              kb={kb}
              onOpen={() => navigate(`/kbs/${kb.kb_id}/documents`)}
              onEdit={() => setEditTarget(kb)}
              onDelete={() => setDeleteTarget(kb)}
            />
          ))}
        </div>
      )}

      {/* Create */}
      <AddKBDialog
        open={createOpen}
        mode="create"
        onCancel={() => setCreateOpen(false)}
        onSuccess={() => {
          setCreateOpen(false);
          void load();
        }}
      />

      {/* Edit (load full config first) */}
      <EditKBLoader
        target={editTarget}
        onClose={() => setEditTarget(null)}
        onSaved={() => {
          setEditTarget(null);
          void load();
        }}
      />

      {/* Delete confirm */}
      <Dialog
        open={!!deleteTarget}
        onOpenChange={(o) => !o && setDeleteTarget(null)}
      >
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>删除知识库</DialogTitle>
            <DialogDescription>
              将永久删除 <code className="rounded bg-bg-tertiary px-1 font-mono text-xs">{deleteTarget?.kb_id}</code> 及其全部文档 / 向量。
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setDeleteTarget(null)}>
              取消
            </Button>
            <Button
              variant="danger"
              onClick={confirmDelete}
              disabled={deleting}
            >
              {deleting ? "删除中..." : "确认删除"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function KBCard({
  kb,
  onOpen,
  onEdit,
  onDelete,
}: {
  kb: KnowledgeBaseSummary;
  onOpen: () => void;
  onEdit: () => void;
  onDelete: () => void;
}) {
  return (
    <Card
      className={cn(
        "group relative cursor-pointer transition-colors hover:border-fg-muted",
        !kb.enabled && "opacity-60"
      )}
      onClick={onOpen}
    >
      <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-2">
        <div className="flex items-center gap-2">
          <div className="flex size-7 items-center justify-center rounded bg-accent/15 text-accent">
            <Database className="size-3.5" />
          </div>
          <div>
            <div className="text-sm font-medium text-fg">{kb.name}</div>
            <div className="font-mono text-[10px] text-fg-muted">{kb.kb_id}</div>
          </div>
        </div>
        <div onClick={(e) => e.stopPropagation()}>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="h-7 w-7">
                <MoreHorizontal className="size-3.5" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={onOpen}>
                <FileText className="size-3.5" /> 管理文档
              </DropdownMenuItem>
              <DropdownMenuItem onClick={onEdit}>
                <SettingsIcon className="size-3.5" /> 编辑
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onClick={onDelete}
                className="text-danger focus:text-danger"
              >
                <Trash2 className="size-3.5" /> 删除
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 pt-0">
        <p className="line-clamp-2 text-xs text-fg-secondary">
          {kb.description || "（无描述）"}
        </p>
        <div className="grid grid-cols-2 gap-2">
          <Stat icon={FileText} label="文档" value={kb.document_count ?? 0} />
          <Stat icon={Layers} label="Chunks" value={kb.chunk_count ?? 0} />
        </div>
        <div className="flex flex-wrap items-center gap-1.5 text-[10px] text-fg-muted">
          {kb.department && (
            <Badge variant="muted" className="font-normal">
              <Building2 className="mr-1 size-2.5" />
              {kb.department}
            </Badge>
          )}
          {kb.embedder && (
            <Badge variant="outline" className="font-normal">
              <Cpu className="mr-1 size-2.5" />
              {kb.embedder}
            </Badge>
          )}
        </div>
        <div className="flex items-center justify-between border-t border-border-subtle pt-2">
          <Badge variant={kb.enabled ? "success" : "muted"}>
            {kb.enabled ? "启用" : "停用"}
          </Badge>
          <span className="flex items-center gap-1 text-[10px] text-fg-muted opacity-0 transition-opacity group-hover:opacity-100">
            进入 <ArrowUpRight className="size-2.5" />
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

function Stat({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof FileText;
  label: string;
  value: number;
}) {
  return (
    <div className="flex items-center gap-2 rounded border border-border-subtle bg-bg-tertiary px-2 py-1.5">
      <Icon className="size-3 text-fg-muted" />
      <div>
        <div className="text-[10px] text-fg-muted">{label}</div>
        <div className="text-xs font-medium text-fg">{value}</div>
      </div>
    </div>
  );
}

/**
 * EditKBLoader — 编辑模式需要拉取完整 config（含 collection / embedding），
 * 拉取成功后再渲染 AddKBDialog。
 */
function EditKBLoader({
  target,
  onClose,
  onSaved,
}: {
  target: KnowledgeBaseSummary | null;
  onClose: () => void;
  onSaved: () => void;
}) {
  const { client } = useApi();
  const [config, setConfig] = useState<KnowledgeBaseSummary | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!target) {
      setConfig(null);
      return;
    }
    setLoading(true);
    (async () => {
      try {
        const c = await client.getKB(target.kb_id);
        setConfig(c);
      } catch (e) {
        toast.error((e as Error).message);
        onClose();
      } finally {
        setLoading(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target]);

  if (!target) return null;
  return (
    <AddKBDialog
      open={!!target}
      mode="edit"
      initial={config}
      loading={loading}
      onCancel={onClose}
      onSuccess={onSaved}
    />
  );
}
