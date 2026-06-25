/**
 * AddKBDialog — 新建/编辑知识库弹窗
 *
 * 用 react-hook-form + zod 替代 antd Form；
 * shadcn Dialog 替代 antd Modal。
 *
 * 字段：
 *   - kb_id（创建时必填，编辑时禁用）
 *   - name
 *   - description
 *   - department
 *   - collection（可选；创建时默认 kb_<kb_id>）
 *   - embedding（下拉选）
 *   - reranker（下拉选；可空）
 *   - enabled
 */
import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Database, Info, Cpu, CheckCircle2, AlertTriangle } from "lucide-react";
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
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import { useApi } from "@/api/client";
import type {
  EmbedderInfo,
  KnowledgeBaseCreate,
  KnowledgeBaseSummary,
  KnowledgeBaseUpdate,
  RerankerInfo,
} from "@/api/types";
import { toast } from "sonner";

const createSchema = z.object({
  kb_id: z
    .string()
    .min(2, "长度 2-64")
    .max(64, "长度 2-64")
    .regex(/^[a-z][a-z0-9_]*$/, "必须以小写字母开头，只能包含小写字母/数字/下划线"),
  name: z.string().min(1, "请输入名称"),
  description: z.string().min(1, "请输入描述"),
  department: z.string().min(1, "请输入部门"),
  collection: z.string().optional().or(z.literal("")),
  embedding: z.string().min(1, "请选 embedding"),
  reranker: z.string().optional().nullable(),
  enabled: z.boolean(),
});

type FormValues = z.infer<typeof createSchema>;

interface Props {
  open: boolean;
  mode: "create" | "edit";
  initial?: KnowledgeBaseSummary | null;
  loading?: boolean;
  onCancel: () => void;
  onSuccess: () => void;
}

export default function AddKBDialog({
  open,
  mode,
  initial,
  loading: externalLoading,
  onCancel,
  onSuccess,
}: Props) {
  const { client } = useApi();
  const [embedders, setEmbedders] = useState<EmbedderInfo[]>([]);
  const [rerankers, setRerankers] = useState<RerankerInfo[]>([]);
  const [loadingOptions, setLoadingOptions] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const form = useForm<FormValues>({
    resolver: zodResolver(createSchema),
    defaultValues: {
      kb_id: "",
      name: "",
      description: "",
      department: "",
      collection: "",
      embedding: "",
      reranker: null,
      enabled: true,
    },
    mode: "onTouched",
  });

  const { register, handleSubmit, watch, setValue, reset, formState } = form;
  const { errors } = formState;
  const watchedEmbedding = watch("embedding");
  const watchedKbId = watch("kb_id");

  // 拉 embedders / rerankers
  useEffect(() => {
    if (!open) return;
    setLoadingOptions(true);
    (async () => {
      try {
        const [e, r] = await Promise.all([
          client.listEmbedders(),
          client.listRerankers(),
        ]);
        setEmbedders(e);
        setRerankers(r);
      } catch (err) {
        toast.error((err as Error).message);
      } finally {
        setLoadingOptions(false);
      }
    })();
  }, [open, client]);

  // 弹窗打开时填表单
  useEffect(() => {
    if (!open) return;
    if (mode === "edit" && initial) {
      // 后端 getKB 返回的是 Summary（embedder 字段），不是 Config（embedding 字段）。
      // 兼容两种命名：用 embedder 优先，否则用 embedding。
      const embedding =
        (initial as unknown as { embedding?: string }).embedding ??
        (initial as unknown as { embedder?: string }).embedder ??
        "";
      reset({
        kb_id: initial.kb_id,
        name: initial.name,
        description: initial.description ?? "",
        department: initial.department ?? "",
        collection: initial.collection ?? "",
        embedding,
        reranker: initial.reranker ?? null,
        enabled: initial.enabled,
      });
    } else if (mode === "create") {
      // 等 embedders 拉回来后选一个默认
      const t = setTimeout(() => {
        const def = embedders.find((e) => e.loaded) ?? embedders[0];
        reset({
          kb_id: "",
          name: "",
          description: "",
          department: "",
          collection: "",
          embedding: def?.name ?? "",
          reranker: null,
          enabled: true,
        });
      }, 50);
      return () => clearTimeout(t);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, mode, initial, embedders.length]);

  // 创建时 kb_id 变化 → 自动填 collection
  useEffect(() => {
    if (mode !== "create") return;
    if (watchedKbId) {
      setValue("collection", `kb_${watchedKbId}`);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [watchedKbId, mode]);

  const selectedEmbedder = embedders.find((e) => e.name === watchedEmbedding);

  const onSubmit = handleSubmit(async (v) => {
    setSubmitting(true);
    try {
      if (mode === "create") {
        const payload: KnowledgeBaseCreate = {
          kb_id: v.kb_id,
          name: v.name,
          description: v.description,
          department: v.department,
          collection: v.collection || undefined,
          embedding: v.embedding,
          reranker: v.reranker ?? null,
          enabled: v.enabled,
        };
        const cfg = await client.createKB(payload);
        toast.success(
          `KB '${cfg.kb_id}' 创建成功${
            selectedEmbedder && !selectedEmbedder.loaded
              ? "（注意：embedder 未加载，KB 会被 skip）"
              : ""
          }`
        );
      } else {
        const patch: KnowledgeBaseUpdate = {
          name: v.name,
          description: v.description,
          department: v.department,
          reranker: v.reranker ?? null,
          enabled: v.enabled,
        };
        const cfg = await client.updateKB(initial!.kb_id, patch);
        toast.success(`KB '${cfg.kb_id}' 已更新`);
      }
      onSuccess();
    } catch (err) {
      toast.error((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  });

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onCancel()}>
      <DialogContent className="max-w-2xl p-0 gap-0 overflow-hidden">
        <DialogHeader className="px-5 pt-5 pb-3 shrink-0">
          <DialogTitle className="flex items-center gap-2">
            <Database className="size-3.5 text-accent" />
            {mode === "create" ? "新建知识库" : `编辑知识库 · ${initial?.kb_id}`}
          </DialogTitle>
          <DialogDescription>
            {mode === "create"
              ? "配置 KB 的 embedding、reranker、collection"
              : "修改后保存会立即更新 KB 运行时配置"}
          </DialogDescription>
        </DialogHeader>

        {mode === "edit" && (
          <div className="flex items-start gap-2 rounded border border-warning/30 bg-warning/5 px-3 py-2 text-xs text-warning mx-5 mb-3">
            <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
            <div>
              kb_id / collection / embedding 改完需要重建 collection（破坏数据），
              因此编辑模式下这三个字段已锁定。
            </div>
          </div>
        )}

        {externalLoading ? (
          <div className="space-y-3 overflow-y-auto px-5 py-2 min-h-0 flex-1">
            <Skeleton className="h-9" />
            <Skeleton className="h-9" />
            <Skeleton className="h-20" />
            <Skeleton className="h-9" />
          </div>
        ) : (
          <form onSubmit={onSubmit} className="flex flex-col min-h-0 flex-1">
            <div className="space-y-3 overflow-y-auto px-5 py-2 min-h-0 flex-1">
            <div className="grid grid-cols-2 gap-3">
              <Field
                label="KB ID"
                error={errors.kb_id?.message}
                hint="以小写字母开头，只能包含小写字母/数字/下划线"
              >
                <Input
                  {...register("kb_id")}
                  placeholder="rd_frontend"
                  disabled={mode === "edit"}
                  className="font-mono"
                />
              </Field>
              <Field label="部门" error={errors.department?.message}>
                <Input {...register("department")} placeholder="RD" />
              </Field>
            </div>
            <Field label="名称" error={errors.name?.message}>
              <Input {...register("name")} placeholder="前端研发知识库" />
            </Field>
            <Field
              label="描述"
              error={errors.description?.message}
              hint="Agent 选 KB 的依据，写清楚有什么内容、解决什么问题"
            >
              <Textarea
                {...register("description")}
                rows={2}
                placeholder="React/Vue/TypeScript/性能优化等前端开发相关的内部技术文档"
              />
            </Field>
            <Field
              label="Collection 名称（向量库）"
              hint="不填则用 'kb_<kb_id>'"
            >
              <Input
                {...register("collection")}
                placeholder={`kb_${watchedKbId || "<kb_id>"}`}
                disabled={mode === "edit"}
                className="font-mono"
              />
            </Field>
            <Field label="Embedding 模型" error={errors.embedding?.message}>
              <Select
                value={watchedEmbedding}
                onValueChange={(v) => setValue("embedding", v, { shouldValidate: true })}
                disabled={mode === "edit"}
              >
                <SelectTrigger>
                  <SelectValue placeholder="选择 embedding 配置" />
                </SelectTrigger>
                <SelectContent>
                  {embedders.map((e) => (
                    <SelectItem key={e.name} value={e.name}>
                      <div className="flex items-center gap-2">
                        <Cpu className="size-3" />
                        <span className="font-mono">{e.name}</span>
                        <Badge variant="outline" className="font-normal">
                          {e.provider}
                        </Badge>
                        <span className="text-[10px] text-fg-muted">
                          dim={e.dimension}
                        </span>
                        {e.loaded ? (
                          <CheckCircle2 className="size-3 text-success" />
                        ) : (
                          <Badge variant="danger" className="font-normal">
                            未加载
                          </Badge>
                        )}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
            {selectedEmbedder && (
              <div className="rounded border border-border-subtle bg-bg-tertiary p-2 text-[11px]">
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-fg-secondary">
                  <span>模型：{selectedEmbedder.model_name}</span>
                  <span>·</span>
                  <span>维度：{selectedEmbedder.dimension}</span>
                  <span>·</span>
                  <span>batch：{selectedEmbedder.batch_size}</span>
                  {selectedEmbedder.loaded ? (
                    <Badge variant="success" className="font-normal">
                      已加载
                    </Badge>
                  ) : (
                    <Badge variant="danger" className="font-normal">
                      未加载（KB 会被 skip）
                    </Badge>
                  )}
                </div>
                {selectedEmbedder.description && (
                  <div className="mt-1 text-fg-muted">
                    {selectedEmbedder.description}
                  </div>
                )}
              </div>
            )}
            <Field
              label={
                <span className="flex items-center gap-1">
                  Reranker 模型
                  <Info className="size-3 text-fg-muted" />
                  <span className="text-[10px] text-fg-muted">
                    不选 = 不用重排
                  </span>
                </span>
              }
            >
              <Select
                value={watch("reranker") ?? "__none__"}
                onValueChange={(v) =>
                  setValue("reranker", v === "__none__" ? null : v)
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="不选 = 不用重排" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__none__">
                    <span className="text-fg-muted">（不使用）</span>
                  </SelectItem>
                  {rerankers.map((r) => (
                    <SelectItem key={r.name} value={r.name}>
                      <div className="flex items-center gap-2">
                        <Cpu className="size-3" />
                        <span className="font-mono">{r.name}</span>
                        <Badge variant="outline" className="font-normal">
                          {r.provider}
                        </Badge>
                        <span className="text-[10px] text-fg-muted">
                          top_k={r.top_k}
                        </span>
                        {!r.loaded && (
                          <Badge variant="danger" className="font-normal">
                            未加载
                          </Badge>
                        )}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
            <Separator />
            <div className="flex items-center justify-between rounded border border-border-subtle bg-bg-tertiary px-3 py-2">
              <div>
                <Label className="text-xs">启用</Label>
                <div className="text-[10px] text-fg-muted">
                  关闭后 KB 在 MCP 端不可见、检索时跳过
                </div>
              </div>
              <Switch
                checked={watch("enabled")}
                onCheckedChange={(v) => setValue("enabled", v)}
              />
            </div>
            </div>
            <DialogFooter className="gap-2 pt-3 shrink-0 border-t border-border-subtle px-5 pb-5">
              <Button type="button" variant="ghost" onClick={onCancel}>
                取消
              </Button>
              <Button type="submit" disabled={submitting || loadingOptions}>
                {submitting
                  ? "提交中..."
                  : mode === "create"
                  ? "创建"
                  : "保存"}
              </Button>
            </DialogFooter>
          </form>
        )}
      </DialogContent>
    </Dialog>
  );
}

function Field({
  label,
  error,
  hint,
  children,
}: {
  label: React.ReactNode;
  error?: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1">
      <Label className="text-xs">{label}</Label>
      {children}
      {hint && !error && (
        <div className="text-[10px] text-fg-muted">{hint}</div>
      )}
      {error && <div className="text-[10px] text-danger">{error}</div>}
    </div>
  );
}
