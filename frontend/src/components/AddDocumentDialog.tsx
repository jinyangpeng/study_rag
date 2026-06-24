/**
 * AddDocumentDialog — 添加文档弹窗
 *
 * 替代原 AddDocumentDrawer：
 *   - shadcn Dialog（不是 Drawer）— 高密度信息展示
 *   - react-hook-form + zod
 *   - Tabs（文本 / 上传）+ parser 选择
 *   - 实时预览分块
 *   - 异步上传 → 轮询 job
 */
import { useEffect, useRef, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import {
  FileText,
  UploadCloud,
  Eye,
  Loader2,
  CheckCircle2,
  XCircle,
  StopCircle,
  Clock,
  Inbox,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import { useApi } from "@/api/client";
import type {
  JobInfo,
  JobStatus,
  JobStage,
  ParserSpec,
  ChunkPreviewItem,
  UploadDocumentResponse,
} from "@/api/types";
import { formatBytes, formatRelativeTime } from "@/lib/utils";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

const schema = z
  .object({
    doc_id: z
      .string()
      .min(1, "请输入 doc_id")
      .max(128, "长度 1-128"),
    title: z.string().min(1, "请输入标题"),
    content: z.string().optional(),
    source: z.string().optional(),
    parser: z.string().min(1, "请选择切块策略"),
    file: z.any().optional(),
  })
  .refine(() => true, { message: "" });

type FormValues = z.infer<typeof schema>;

interface Props {
  open: boolean;
  kbId: string;
  onCancel: () => void;
  onSuccess: () => void;
}

const STAGE_LABEL: Record<JobStage, string> = {
  queued: "排队中",
  parsing: "解析文件",
  chunking: "切分文本",
  embedding: "生成向量",
  saving: "写入数据库",
  done: "完成",
};

const POLL_INTERVAL_MS = 1000;

export default function AddDocumentDialog({
  open,
  kbId,
  onCancel,
  onSuccess,
}: Props) {
  const { client } = useApi();
  const [tab, setTab] = useState<"text" | "file">("text");
  const [parsers, setParsers] = useState<ParserSpec[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<ChunkPreviewItem[] | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [job, setJob] = useState<JobInfo | null>(null);
  const pollTimerRef = useRef<number | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const form = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      doc_id: "",
      title: "",
      content: "",
      source: "",
      parser: "",
    },
    mode: "onTouched",
  });

  const { register, handleSubmit, watch, setValue, reset, getValues, formState } = form;
  const { errors } = formState;
  const watchedParser = watch("parser");

  // 拉 parser 列表
  useEffect(() => {
    if (!open) return;
    (async () => {
      try {
        const ps = await client.listParsers();
        setParsers(ps);
        const def = ps.find((p) => p.name === "sentence_512") ?? ps[0];
        if (def && !getValues("parser")) {
          setValue("parser", def.name);
        }
      } catch (e) {
        toast.error((e as Error).message);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // 关闭时清状态
  useEffect(() => {
    if (!open) {
      reset();
      setFile(null);
      setPreview(null);
      setJob(null);
      stopPolling();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => () => stopPolling(), []);

  function stopPolling() {
    if (pollTimerRef.current !== null) {
      window.clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }

  function startPolling(jobId: string) {
    stopPolling();
    pollTimerRef.current = window.setInterval(async () => {
      try {
        const info = await client.getJob(jobId);
        setJob(info);
        const status: JobStatus = info.status;
        if (status === "done" || status === "error" || status === "cancelled") {
          stopPolling();
          if (status === "done") {
            toast.success(`${info.doc_id ?? "文档"} 上传完成`);
            onSuccess();
            onCancel();
          } else if (status === "error") {
            toast.error(`上传失败: ${info.error ?? "未知错误"}`);
          } else if (status === "cancelled") {
            toast.warning("任务已取消");
          }
        }
      } catch (e) {
        console.warn("job poll failed:", e);
      }
    }, POLL_INTERVAL_MS);
  }

  async function handleCancelJob() {
    if (!job) return;
    try {
      await client.cancelJob(job.job_id);
      toast.info("已请求取消");
    } catch (e) {
      toast.error((e as Error).message);
    }
  }

  async function onPreview() {
    const v = getValues();
    if (!v.content) {
      toast.warning("请先输入正文");
      return;
    }
    if (!v.parser) {
      toast.warning("请选择切块策略");
      return;
    }
    setPreviewLoading(true);
    try {
      const r = await client.previewChunk(kbId, v.content, v.parser, v.title || "preview");
      setPreview(r.chunks);
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setPreviewLoading(false);
    }
  }

  const onSubmit = handleSubmit(async (v) => {
    setSubmitting(true);
    try {
      if (tab === "text") {
        if (!v.content) {
          toast.warning("请输入正文");
          return;
        }
        const parserSpec = parsers.find((p) => p.name === v.parser);
        await client.addDocumentChunked({
          kb_id: kbId,
          doc_id: v.doc_id,
          title: v.title,
          content: v.content,
          source: v.source || null,
          metadata: {},
          chunk_size: parserSpec?.chunk_size ?? 512,
          chunk_overlap: parserSpec?.chunk_overlap ?? 50,
        });
        toast.success(`文档 ${v.doc_id} 添加成功`);
        onSuccess();
        onCancel();
      } else {
        if (!file) {
          toast.error("请先选择文件");
          return;
        }
        const fd = new FormData();
        fd.append("file", file);
        fd.append("doc_id", v.doc_id);
        fd.append("title", v.title);
        fd.append("parser", v.parser);
        if (v.source) fd.append("source", v.source);
        const r: UploadDocumentResponse = await client.uploadDocument(kbId, fd);
        setJob({
          job_id: r.job_id,
          type: "upload_doc",
          status: r.status as JobStatus,
          stage: "queued",
          current: 0,
          total: 0,
          progress: 0,
          message: "已提交，等待处理",
          error: null,
          kb_id: r.kb_id,
          doc_id: r.doc_id,
          filename: file.name,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        });
        startPolling(r.job_id);
      }
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  });

  function handleFile(f: File | null) {
    setFile(f);
    if (f && !getValues("title")) {
      const nameOnly = f.name.replace(/\.[^.]+$/, "");
      setValue("title", nameOnly);
    }
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  }

  const jobInProgress = !!job && (job.status === "running" || job.status === "pending");

  return (
    <Dialog open={open} onOpenChange={(o) => !o && !jobInProgress && onCancel()}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileText className="size-3.5 text-accent" />
            添加文档
          </DialogTitle>
          <DialogDescription>
            支持文本输入（手动粘贴）或文件上传（txt/md/html/pdf/docx）
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={onSubmit} className="space-y-3">
          <Tabs value={tab} onValueChange={(v) => setTab(v as "text" | "file")}>
            <TabsList>
              <TabsTrigger value="text">
                <FileText className="size-3" />
                文本输入
              </TabsTrigger>
              <TabsTrigger value="file">
                <UploadCloud className="size-3" />
                文件上传
              </TabsTrigger>
            </TabsList>
            <TabsContent value="text" className="space-y-1.5">
              <Label className="text-xs">正文</Label>
              <Textarea
                {...register("content")}
                rows={6}
                placeholder="粘贴文本内容（100-10000 字）"
              />
              {errors.content && (
                <div className="text-[10px] text-danger">{errors.content.message}</div>
              )}
            </TabsContent>
            <TabsContent value="file" className="space-y-2">
              <Label className="text-xs">文件</Label>
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={onDrop}
                className={cn(
                  "flex flex-col items-center justify-center gap-2 rounded border border-dashed border-border bg-bg-tertiary p-6 text-center text-xs text-fg-muted transition-colors",
                  dragOver && "border-accent bg-accent/5"
                )}
              >
                <UploadCloud className="size-5 text-fg-muted" />
                <div>点击或拖拽文件到此处</div>
                <div className="text-[10px] text-fg-muted">
                  支持 txt / md / html / pdf / docx，单文件最大 50MB
                </div>
                <input
                  id="file-input"
                  type="file"
                  className="hidden"
                  accept=".txt,.md,.markdown,.html,.htm,.pdf,.docx"
                  onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
                />
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => document.getElementById("file-input")?.click()}
                >
                  选择文件
                </Button>
              </div>
              {file && (
                <div className="flex items-center justify-between rounded border border-border-subtle bg-bg-tertiary px-3 py-2 text-xs">
                  <div className="flex items-center gap-2">
                    <FileText className="size-3 text-fg-muted" />
                    <span className="font-mono">{file.name}</span>
                    <Badge variant="muted">{formatBytes(file.size)}</Badge>
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => handleFile(null)}
                  >
                    移除
                  </Button>
                </div>
              )}
            </TabsContent>
          </Tabs>

          <Separator />

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label className="text-xs">doc_id</Label>
              <Input
                {...register("doc_id")}
                placeholder="react_perf_001"
                className="font-mono"
              />
              {errors.doc_id && (
                <div className="text-[10px] text-danger">{errors.doc_id.message}</div>
              )}
            </div>
            <div className="space-y-1">
              <Label className="text-xs">标题</Label>
              <Input
                {...register("title")}
                placeholder="React 性能优化指南"
              />
              {errors.title && (
                <div className="text-[10px] text-danger">{errors.title.message}</div>
              )}
            </div>
          </div>

          <div className="space-y-1">
            <Label className="text-xs">source（可选）</Label>
            <Input
              {...register("source")}
              placeholder="来源标识"
            />
          </div>

          <div className="space-y-1">
            <Label className="text-xs">切块策略</Label>
            <Select
              value={watchedParser}
              onValueChange={(v) => setValue("parser", v, { shouldValidate: true })}
            >
              <SelectTrigger>
                <SelectValue placeholder="选择 parser" />
              </SelectTrigger>
              <SelectContent>
                {parsers.map((p) => (
                  <SelectItem key={p.name} value={p.name}>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="font-normal">
                        {p.strategy}
                      </Badge>
                      <span className="font-mono">{p.name}</span>
                      <span className="text-[10px] text-fg-muted">
                        size={p.chunk_size}, overlap={p.chunk_overlap}
                      </span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {errors.parser && (
              <div className="text-[10px] text-danger">{errors.parser.message}</div>
            )}
          </div>

          {previewLoading && (
            <div className="space-y-2 rounded border border-border-subtle bg-bg-tertiary p-3">
              <Skeleton className="h-4 w-32" />
              <Skeleton className="h-16" />
              <Skeleton className="h-16" />
            </div>
          )}

          {preview && !previewLoading && (
            <div className="space-y-2 rounded border border-border-subtle bg-bg-tertiary p-3">
              <div className="flex items-center gap-2 text-xs font-medium text-fg">
                <Inbox className="size-3" />
                预览：{preview.length} 个块
                <Badge variant="muted" className="font-normal">
                  总字符 {preview.reduce((s, c) => s + c.char_count, 0)}
                </Badge>
              </div>
              <ScrollArea className="h-40">
                <div className="space-y-2 pr-2">
                  {preview.map((c) => (
                    <div
                      key={c.chunk_index}
                      className="rounded border border-border-subtle bg-bg-secondary p-2"
                    >
                      <div className="flex items-center gap-2 text-[10px] text-fg-muted">
                        <Badge variant="outline" className="font-normal">
                          #{c.chunk_index}
                        </Badge>
                        <span>{c.char_count} chars</span>
                      </div>
                      <div className="mt-1 whitespace-pre-wrap text-[11px] text-fg-secondary">
                        {c.text}
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </div>
          )}

          {job && <JobProgressPanel job={job} onCancel={handleCancelJob} />}

          <DialogFooter className="gap-2 pt-2">
            <Button
              type="button"
              variant="ghost"
              onClick={onCancel}
              disabled={jobInProgress}
            >
              取消
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={onPreview}
              disabled={jobInProgress || tab !== "text"}
            >
              <Eye className="size-3" />
              预览分块
            </Button>
            <Button type="submit" disabled={submitting || jobInProgress}>
              {submitting ? "提交中..." : tab === "text" ? "添加" : "上传"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function JobProgressPanel({
  job,
  onCancel,
}: {
  job: JobInfo;
  onCancel: () => void;
}) {
  const pct = Math.round((job.progress ?? 0) * 100);
  const status = job.status;
  const stageLabel = STAGE_LABEL[job.stage] ?? job.stage;
  const cancellable = status === "pending" || status === "running";

  let icon: React.ReactNode = <Loader2 className="size-3.5 animate-spin text-accent" />;
  let color: string = "text-accent";
  if (status === "done") {
    icon = <CheckCircle2 className="size-3.5 text-success" />;
    color = "text-success";
  } else if (status === "error") {
    icon = <XCircle className="size-3.5 text-danger" />;
    color = "text-danger";
  } else if (status === "cancelled") {
    icon = <StopCircle className="size-3.5 text-warning" />;
    color = "text-warning";
  } else if (status === "pending") {
    icon = <Clock className="size-3.5 text-fg-muted" />;
    color = "text-fg-muted";
  }

  return (
    <div className="space-y-2 rounded border border-border-subtle bg-bg-tertiary p-3">
      <div className="flex items-center gap-2 text-xs">
        {icon}
        <span className="font-medium text-fg">异步上传进度</span>
        <Badge variant="outline" className="font-normal">
          {stageLabel}
        </Badge>
        <span className="ml-auto text-[10px] text-fg-muted">
          {formatRelativeTime(job.updated_at)}
        </span>
      </div>
      <Progress value={pct} />
      <div className="flex items-center justify-between text-[10px] text-fg-muted">
        <span>{job.message || stageLabel}</span>
        <span>
          {job.current > 0 && job.total > 0
            ? `${job.current} / ${job.total}`
            : `${pct}%`}
        </span>
      </div>
      {status === "error" && job.error && (
        <div className="rounded border border-danger/30 bg-danger/5 px-2 py-1 text-[10px] text-danger">
          {job.error}
        </div>
      )}
      {cancellable && (
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={onCancel}
          className={color}
        >
          取消任务
        </Button>
      )}
    </div>
  );
}
