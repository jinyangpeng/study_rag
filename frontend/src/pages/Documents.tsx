/**
 * Documents — 文档管理页
 *
 * 布局：
 *  - 顶部 toolbar（KB 选择 + 搜索 + 添加按钮 + 刷新）
 *  - Table（高密度 Linear 风格）：doc_id / title / source / chunks / size / created_at / 操作
 *  - AddDocumentDialog（点击「添加文档」打开）
 *  - 删除确认 Dialog
 *  - 进入 KB 列表前要选 KB（默认从 URL param 读）
 */
import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  Plus,
  RefreshCw,
  Search as SearchIcon,
  FileText,
  Trash2,
  Blocks,
  Database,
  AlertCircle,
  Upload,
  Eye,
  Loader2,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { EmptyState } from "@/components/shared/EmptyState";
import { ErrorState } from "@/components/shared/ErrorState";
import AddDocumentDialog from "../components/AddDocumentDialog";
import DocumentDetailDialog from "../components/DocumentDetailDialog";
import BatchAddDocumentDialog from "../components/BatchAddDocumentDialog";
import { useApi } from "@/api/client";
import type { DocumentMeta, KnowledgeBaseSummary } from "@/api/types";
import { formatRelativeTime, formatChars } from "@/lib/utils";
import { toast } from "sonner";
import { useJobPolling, getActiveJobs } from "@/hooks/useJobPolling";

export default function Documents() {
  const { kbId: urlKbId } = useParams<{ kbId?: string }>();
  const { client } = useApi();
  const navigate = useNavigate();

  const [kbs, setKbs] = useState<KnowledgeBaseSummary[]>([]);
  const [activeKbId, setActiveKbId] = useState<string>(urlKbId ?? "");
  const [docs, setDocs] = useState<DocumentMeta[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [addOpen, setAddOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<DocumentMeta | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [detailTarget, setDetailTarget] = useState<DocumentMeta | null>(null);
  const [batchOpen, setBatchOpen] = useState(false);

  // 加载 KB 列表
  useEffect(() => {
    (async () => {
      try {
        const k = await client.listKBs();
        setKbs(k);
        if (!activeKbId && k.length > 0) {
          setActiveKbId(k[0].kb_id);
        }
      } catch (e) {
        setError((e as Error).message);
        toast.error((e as Error).message);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 加载文档
  useEffect(() => {
    if (!activeKbId) return;
    void loadDocs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeKbId]);

  // 顶栏后台任务指示器（简单的"处理中"提示）
  const [activeJobs, setActiveJobs] = useState(getActiveJobs());

  // 后台任务（如文件上传）轮询：完成时刷新列表 + 弹 toast；进行中时更新顶栏计数
  // 仅当 job 属于当前 active KB 时才触发，避免误刷新
  useJobPolling({
    onTerminal: ({ status, job }) => {
      if (job.kb_id && job.kb_id !== activeKbId) return;
      const name = job.filename || job.doc_id || "文档";
      if (status === "done") {
        toast.success(`${name} 处理完成`);
        void loadDocs();
      } else if (status === "error") {
        toast.error(`${name} 处理失败: ${job.error ?? "未知错误"}`);
        void loadDocs();
      } else if (status === "cancelled") {
        toast.warning(`${name} 任务已取消`);
      }
    },
    onProgress: (jobs) => setActiveJobs(jobs),
  });
  const pendingCount = activeJobs.filter(
    (j) => !j.kb_id || j.kb_id === activeKbId
  ).length;

  async function loadDocs() {
    if (!activeKbId) return;
    setLoading(true);
    setError(null);
    try {
      const d = await client.listDocuments(activeKbId);
      setDocs(d);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  const filtered = docs.filter((d) => {
    if (!search) return true;
    const q = search.toLowerCase();
    return (
      d.doc_id.toLowerCase().includes(q) ||
      d.title.toLowerCase().includes(q) ||
      (d.source ?? "").toLowerCase().includes(q)
    );
  });

  const totalChunks = docs.reduce((s, d) => s + (d.chunk_count ?? 0), 0);
  const activeKb = kbs.find((k) => k.kb_id === activeKbId);

  async function confirmDelete() {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      await client.deleteDocument(deleteTarget.kb_id, deleteTarget.doc_id);
      toast.success(`文档 ${deleteTarget.doc_id} 已删除`);
      setDeleteTarget(null);
      void loadDocs();
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
          <h2 className="text-base font-semibold">文档管理</h2>
          <p className="text-xs text-fg-muted">
            {activeKb
              ? `${activeKb.name} · ${docs.length} 个文档 · ${totalChunks} 个 chunks`
              : "选择一个知识库查看其文档"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {pendingCount > 0 && (
            <div
              className="flex items-center gap-1.5 rounded border border-accent/30 bg-accent/5 px-2 py-1 text-[10px] text-accent"
              title={`当前有 ${pendingCount} 个后台任务在处理`}
            >
              <Loader2 className="size-3 animate-spin" />
              <span>处理中 {pendingCount}</span>
            </div>
          )}
          <div className="relative">
            <SearchIcon className="absolute left-2 top-1/2 size-3.5 -translate-y-1/2 text-fg-muted" />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="搜索文档..."
              className="h-8 w-48 pl-7 text-xs"
              disabled={!activeKbId}
            />
          </div>
          <Button
            variant="outline"
            size="icon"
            onClick={() => void loadDocs()}
            disabled={!activeKbId}
            title="刷新"
          >
            <RefreshCw className="size-3.5" />
          </Button>
          <Button
            size="sm"
            onClick={() => setAddOpen(true)}
            disabled={!activeKbId}
          >
            <Plus className="size-3.5" />
            添加文档
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setBatchOpen(true)}
            disabled={!activeKbId}
          >
            <Upload className="size-3.5" />
            批量添加
          </Button>
        </div>
      </div>

      {/* KB selector */}
      <Card>
        <CardContent className="flex items-center gap-3 p-3">
          <Database className="size-3.5 text-fg-muted" />
          <span className="text-xs text-fg-muted">知识库</span>
          <Select value={activeKbId} onValueChange={setActiveKbId}>
            <SelectTrigger className="h-7 w-64 text-xs">
              <SelectValue placeholder="选择 KB" />
            </SelectTrigger>
            <SelectContent>
              {kbs.map((k) => (
                <SelectItem key={k.kb_id} value={k.kb_id}>
                  <div className="flex items-center gap-2">
                    <span className="font-mono">{k.kb_id}</span>
                    <span className="text-fg-muted">· {k.name}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {kbs.length === 0 && (
            <div className="flex items-center gap-1 text-[10px] text-warning">
              <AlertCircle className="size-3" />
              还没有知识库，先去
              <button
                onClick={() => navigate("/kbs")}
                className="text-accent underline"
              >
                知识库管理
              </button>
              创建一个
            </div>
          )}
        </CardContent>
      </Card>

      {/* Table */}
      <Card>
        <CardContent className="p-0">
          {!activeKbId ? (
            <EmptyState
              title="请先选择知识库"
              description="从上方下拉框选择一个 KB"
              icon={Database}
            />
          ) : loading ? (
            <div className="space-y-2 p-4">
              {Array.from({ length: 6 }).map((_, i) => (
                <Skeleton key={i} className="h-9" />
              ))}
            </div>
          ) : error ? (
            <div className="p-4">
              <ErrorState message={error} onRetry={() => void loadDocs()} />
            </div>
          ) : filtered.length === 0 ? (
            <EmptyState
              title={search ? "无匹配的文档" : "该 KB 还没有文档"}
              description={
                search ? "试试其他关键词" : "点击「添加文档」开始添加"
              }
              icon={FileText}
              action={
                !search && (
                  <Button size="sm" onClick={() => setAddOpen(true)}>
                    <Plus className="size-3.5" />
                    添加文档
                  </Button>
                )
              }
            />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>doc_id</TableHead>
                  <TableHead>标题</TableHead>
                  <TableHead>source</TableHead>
                  <TableHead>分块方式</TableHead>
                  <TableHead className="text-right">chunks</TableHead>
                  <TableHead className="text-right">字符数</TableHead>
                  <TableHead>创建时间</TableHead>
                  <TableHead className="w-24 text-right">操作</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filtered.map((d) => (
                  <TableRow key={d.doc_id}>
                    <TableCell className="font-mono text-[11px] text-fg">
                      {d.doc_id}
                    </TableCell>
                    <TableCell className="text-fg">{d.title}</TableCell>
                    <TableCell>
                      {d.source ? (
                        <Badge variant="muted" className="font-normal">
                          {d.source}
                        </Badge>
                      ) : (
                        <span className="text-fg-muted">—</span>
                      )}
                    </TableCell>
                    <TableCell>
                      {d.parser ? (
                        <Badge variant="outline" className="font-mono text-[10px]">
                          {d.parser}
                        </Badge>
                      ) : (
                        <span className="text-fg-muted">—</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      <Badge variant="outline" className="font-mono">
                        <Blocks className="mr-1 size-2.5" />
                        {d.chunk_count ?? 0}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right font-mono text-[11px] text-fg-muted">
                      {d.char_count != null ? formatChars(d.char_count) : "—"}
                    </TableCell>
                    <TableCell className="text-[10px] text-fg-muted">
                      {d.created_at ? formatRelativeTime(d.created_at) : "—"}
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex justify-end gap-1">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7 text-accent hover:text-accent"
                          onClick={() => setDetailTarget(d)}
                          title="查看详情"
                        >
                          <Eye className="size-3" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7 text-danger hover:text-danger"
                          onClick={() => setDeleteTarget(d)}
                          title="删除"
                        >
                          <Trash2 className="size-3" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {activeKbId && (
        <AddDocumentDialog
          open={addOpen}
          kbId={activeKbId}
          onCancel={() => setAddOpen(false)}
          onSuccess={() => {
            setAddOpen(false);
            void loadDocs();
          }}
        />
      )}

      {activeKbId && detailTarget && (
        <DocumentDetailDialog
          open={!!detailTarget}
          kbId={activeKbId}
          docId={detailTarget.doc_id}
          onCancel={() => setDetailTarget(null)}
        />
      )}

      {activeKbId && (
        <BatchAddDocumentDialog
          open={batchOpen}
          kbId={activeKbId}
          kbs={kbs}
          onCancel={() => setBatchOpen(false)}
          onSuccess={() => {
            setBatchOpen(false);
            void loadDocs();
          }}
        />
      )}

      <Dialog
        open={!!deleteTarget}
        onOpenChange={(o) => !o && setDeleteTarget(null)}
      >
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>删除文档</DialogTitle>
            <DialogDescription>
              将永久删除 <code className="rounded bg-bg-tertiary px-1 font-mono text-xs">{deleteTarget?.doc_id}</code> 及其全部向量。
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
