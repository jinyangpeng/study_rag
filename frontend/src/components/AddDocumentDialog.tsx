/**
 * AddDocumentDialog — 添加文档弹窗
 *
 * 设计：
 *   - shadcn Dialog + react-hook-form + zod
 *   - Tabs：文本输入 / 文件上传（默认选中"文件上传"）
 *   - 文本输入：同步提交，弹窗内立即看到结果
 *   - 文件上传：异步任务。提交成功后**立即关弹窗**（用户希望），
 *     后台分块/写库由 useJobPolling 在调用方页面继续监控，job 结束
 *     通过 toast 通知，并刷新文档列表
 *   - 文本 tab 可选"预览分块"功能（依赖 parser）
 */
import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import {
  FileText,
  UploadCloud,
  Eye,
  Inbox,
  CheckCircle2,
  X as CloseIcon,
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
  SelectItemText,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import { useApi } from "@/api/client";
import type {
  ParserSpec,
  ChunkPreviewItem,
  UploadDocumentResponse,
  JobStatus,
} from "@/api/types";
import { formatBytes } from "@/lib/utils";
import { toast } from "sonner";
import { registerJob } from "@/hooks/useJobPolling";
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
  })
  .refine(() => true, { message: "" });

type FormValues = z.infer<typeof schema>;

interface Props {
  open: boolean;
  kbId: string;
  onCancel: () => void;
  onSuccess: () => void;
}

export default function AddDocumentDialog({
  open,
  kbId,
  onCancel,
  onSuccess,
}: Props) {
  const { client } = useApi();
  // 默认 tab：文件上传（按用户要求）
  const [tab, setTab] = useState<"text" | "file">("file");
  const [parsers, setParsers] = useState<ParserSpec[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<ChunkPreviewItem[] | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  // 文件上传提交后：只显示一个"已提交"短提示 + 立即关闭按钮，不阻塞关闭
  const [submittedJobId, setSubmittedJobId] = useState<string | null>(null);

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
      setSubmittedJobId(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

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
          // 关键：传命名 parser（如 'sentence_512'），后端会写到 DocumentMeta.parser
          parser_name: parserSpec?.name ?? v.parser,
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
        // 把 job 注册到全局轮询集合（由调用方页面 useJobPolling 持续监控）
        registerJob({
          job_id: r.job_id,
          type: "upload_doc",
          status: (r.status ?? "pending") as JobStatus,
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
        // 在弹窗内显示一个"已提交"短状态，让用户可以确认后台在跑、随时关弹窗
        setSubmittedJobId(r.job_id);
        toast.info(`${file.name} 已提交后台处理，可关闭弹窗`);
        // 通知父页面：用于在 job 完成时由 hook 再次触发 loadDocs
        onSuccess();
      }
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  });

  function handleCloseAfterSubmit() {
    setSubmittedJobId(null);
    onCancel();
  }

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

  // ============ 文件上传已提交后的"完成态" UI ============
  if (submittedJobId) {
    return (
      <Dialog open={open} onOpenChange={(o) => !o && handleCloseAfterSubmit()}>
        <DialogContent className="max-w-md p-0 gap-0 overflow-hidden">
          <DialogHeader className="px-5 pt-5 pb-3 shrink-0">
            <DialogTitle className="flex items-center gap-2">
              <CheckCircle2 className="size-4 text-success" />
              任务已提交
            </DialogTitle>
            <DialogDescription>
              分块、embedding、写库等步骤在后台异步进行。
              关闭弹窗不影响处理，任务完成时会自动通知。
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3 px-5 py-2">
            <div className="rounded border border-border-subtle bg-bg-tertiary p-3 text-xs space-y-1">
              <div className="flex items-center gap-2">
                <FileText className="size-3 text-fg-muted" />
                <span className="font-mono">{file?.name}</span>
                <Badge variant="muted" className="font-normal">
                  {file ? formatBytes(file.size) : ""}
                </Badge>
              </div>
              <div className="text-fg-muted font-mono text-[10px]">
                job_id: {submittedJobId}
              </div>
            </div>
          </div>
          <DialogFooter className="gap-2 pt-3 shrink-0 border-t border-border-subtle px-5 pb-5">
            <Button onClick={handleCloseAfterSubmit}>
              <CloseIcon className="size-3" />
              关闭弹窗
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  }

  // ============ 主表单 UI ============
  return (
    <Dialog open={open} onOpenChange={(o) => !o && onCancel()}>
      <DialogContent className="max-w-2xl p-0 gap-0 overflow-hidden">
        <DialogHeader className="px-5 pt-5 pb-3 shrink-0">
          <DialogTitle className="flex items-center gap-2">
            <FileText className="size-3.5 text-accent" />
            添加文档
          </DialogTitle>
          <DialogDescription>
            支持文本输入（手动粘贴）或文件上传（txt / md / html / pdf / docx）
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={onSubmit} className="space-y-3 overflow-y-auto px-5 py-2 min-h-0 flex-1">
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
                      {/* 主值：trigger 显示 */}
                      <SelectItemText>
                        <span className="font-mono">{p.name}</span>
                      </SelectItemText>
                      {/* 描述：仅展开时显示 */}
                      <Badge variant="outline" className="font-normal">
                        {p.strategy}
                      </Badge>
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

          <DialogFooter className="gap-2 pt-3 shrink-0 border-t border-border-subtle px-5 pb-5">
            <Button
              type="button"
              variant="ghost"
              onClick={onCancel}
            >
              取消
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={onPreview}
              disabled={tab !== "text"}
            >
              <Eye className="size-3" />
              预览分块
            </Button>
            <Button type="submit" disabled={submitting}>
              {submitting ? "提交中..." : tab === "text" ? "添加" : "上传"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
