/**
 * BatchAddDocumentDialog — 批量添加文档弹窗
 */
import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import {
  Upload,
  FileText,
  Loader2,
  CheckCircle2,
  XCircle,
  AlertCircle,
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
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { useApi } from "@/api/client";
import type { KnowledgeBaseSummary } from "@/api/types";
import { toast } from "sonner";

const schema = z.object({
  documents: z
    .array(
      z.object({
        doc_id: z.string().min(1, "请输入 doc_id"),
        title: z.string().min(1, "请输入标题"),
        content: z.string().min(1, "请输入内容"),
        source: z.string().optional(),
      })
    )
    .min(1, "至少添加一个文档")
    .max(500, "最多添加500个文档"),
});

type FormValues = z.infer<typeof schema>;

interface Props {
  open: boolean;
  kbId: string;
  kbs: KnowledgeBaseSummary[];
  onCancel: () => void;
  onSuccess: () => void;
}

export default function BatchAddDocumentDialog({
  open,
  kbId,
  kbs,
  onCancel,
  onSuccess,
}: Props) {
  const { client } = useApi();
  const [submitting, setSubmitting] = useState(false);
  const [results, setResults] = useState<{
    succeeded: string[];
    failed: Array<{ doc_id: string; error: string }>;
  } | null>(null);

  const form = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      documents: [
        {
          doc_id: "",
          title: "",
          content: "",
          source: "",
        },
      ],
    },
    mode: "onTouched",
  });

  const { register, handleSubmit, watch, formState } = form;
  const { errors } = formState;
  const watchedDocuments = watch("documents");

  // 添加一个空文档行
  const addDocumentRow = () => {
    form.setValue("documents", [
      ...watchedDocuments,
      {
        doc_id: "",
        title: "",
        content: "",
        source: "",
      },
    ]);
  };

  // 移除一个文档行
  const removeDocumentRow = (index: number) => {
    const newDocuments = [...watchedDocuments];
    newDocuments.splice(index, 1);
    form.setValue("documents", newDocuments);
  };

  // 提交表单
  const onSubmit = handleSubmit(async (v) => {
    setSubmitting(true);
    setResults(null);

    try {
      const response = await client.addDocumentsBatch(kbId, {
        documents: v.documents.map(d => ({
          ...d,
          metadata: {},
        })),
        overwrite: false,
      });

      setResults(response);

      if (response.failed.length === 0) {
        toast.success(`成功添加 ${response.succeeded.length} 个文档`);
        onSuccess();
        onCancel();
      } else {
        toast.warning(
          `成功 ${response.succeeded.length} 个，失败 ${response.failed.length} 个`
        );
      }
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  });

  // 关闭时重置表单
  const handleClose = () => {
    if (!submitting) {
      form.reset({
        documents: [
          {
            doc_id: "",
            title: "",
            content: "",
            source: "",
          },
        ],
      });
      setResults(null);
      onCancel();
    }
  };

  if (!open) return null;

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="max-w-3xl p-0 gap-0 overflow-hidden">
        <DialogHeader className="px-5 pt-5 pb-3 shrink-0">
          <DialogTitle className="flex items-center gap-2">
            <Upload className="size-3.5 text-accent" />
            批量添加文档
          </DialogTitle>
          <DialogDescription>
            一次性添加多个文档到知识库。最多添加 500 个文档。
          </DialogDescription>
        </DialogHeader>

        {!results ? (
          <form onSubmit={onSubmit} className="flex flex-col min-h-0 flex-1 px-5">
            <div className="space-y-2">
              <Label className="text-xs">知识库</Label>
              <div className="rounded border border-border-subtle bg-bg-tertiary px-3 py-2 text-xs">
                {kbs.find(k => k.kb_id === kbId)?.name || kbId}
              </div>
            </div>

            <Separator />

            <div className="space-y-3 flex-1 min-h-0 flex flex-col">
              <div className="flex items-center justify-between">
                <Label className="text-xs">文档列表</Label>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={addDocumentRow}
                  disabled={submitting || watchedDocuments.length >= 500}
                >
                  <FileText className="size-3 mr-1" />
                  添加文档
                </Button>
              </div>

              <ScrollArea className="flex-1 min-h-0">
                <div className="space-y-3 p-1">
                  {watchedDocuments.map((_doc, index) => (
                    <div
                      key={index}
                      className="rounded border border-border-subtle bg-bg-tertiary p-3"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-medium">文档 #{index + 1}</span>
                        {watchedDocuments.length > 1 && (
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            onClick={() => removeDocumentRow(index)}
                            disabled={submitting}
                            title="删除"
                          >
                            <XCircle className="size-3" />
                          </Button>
                        )}
                      </div>

                      <div className="space-y-2">
                        <div>
                          <Label className="text-[10px]">doc_id</Label>
                          <input
                            {...register(`documents.${index}.doc_id`)}
                            type="text"
                            className="w-full rounded border border-border bg-bg px-2 py-1 text-xs"
                            placeholder="文档ID"
                          />
                          {errors.documents?.[index]?.doc_id && (
                            <div className="text-[10px] text-danger mt-1">
                              {errors.documents[index]?.doc_id?.message}
                            </div>
                          )}
                        </div>

                        <div>
                          <Label className="text-[10px]">标题</Label>
                          <input
                            {...register(`documents.${index}.title`)}
                            type="text"
                            className="w-full rounded border border-border bg-bg px-2 py-1 text-xs"
                            placeholder="文档标题"
                          />
                          {errors.documents?.[index]?.title && (
                            <div className="text-[10px] text-danger mt-1">
                              {errors.documents[index]?.title?.message}
                            </div>
                          )}
                        </div>

                        <div>
                          <Label className="text-[10px]">内容</Label>
                          <Textarea
                            {...register(`documents.${index}.content`)}
                            rows={3}
                            className="text-xs"
                            placeholder="文档内容"
                          />
                          {errors.documents?.[index]?.content && (
                            <div className="text-[10px] text-danger mt-1">
                              {errors.documents[index]?.content?.message}
                            </div>
                          )}
                        </div>

                        <div>
                          <Label className="text-[10px]">来源（可选）</Label>
                          <input
                            {...register(`documents.${index}.source`)}
                            type="text"
                            className="w-full rounded border border-border bg-bg px-2 py-1 text-xs"
                            placeholder="来源标识"
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>

              {errors.documents && (
                <div className="text-[10px] text-danger flex items-center gap-1">
                  <AlertCircle className="size-3" />
                  {errors.documents.message}
                </div>
              )}
            </div>

            <DialogFooter className="gap-2 pt-2 shrink-0 border-t border-border-subtle px-5 pb-5">
              <Button
                type="button"
                variant="ghost"
                onClick={handleClose}
                disabled={submitting}
              >
                取消
              </Button>
              <Button type="submit" disabled={submitting}>
                {submitting ? (
                  <>
                    <Loader2 className="size-3.5 animate-spin mr-2" />
                    提交中...
                  </>
                ) : (
                  <>
                    <Upload className="size-3.5 mr-2" />
                    批量添加
                  </>
                )}
              </Button>
            </DialogFooter>
          </form>
        ) : (
          <div className="space-y-4 flex-1 min-h-0 flex flex-col overflow-y-auto px-5">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="size-4 text-success" />
                <span className="text-sm font-medium">批量添加完成</span>
              </div>
              <div className="flex gap-4">
                <Badge variant="outline" className="text-green-700 border-green-700 bg-green-50">
                  成功: {results.succeeded.length}
                </Badge>
                <Badge variant="outline" className="text-red-700 border-red-700 bg-red-50">
                  失败: {results.failed.length}
                </Badge>
              </div>
            </div>

            <Separator />

            {results.failed.length > 0 && (
              <div className="space-y-2">
                <div className="text-xs font-medium text-danger">失败列表</div>
                <ScrollArea className="h-64">
                  <div className="space-y-2 p-1">
                    {results.failed.map((f, index) => (
                      <div
                        key={index}
                        className="rounded border border-danger/30 bg-danger/5 p-2 text-xs"
                      >
                        <div className="font-mono">{f.doc_id}</div>
                        <div className="text-danger">{f.error}</div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </div>
            )}

            <DialogFooter className="gap-2 pt-2 shrink-0 border-t border-border-subtle px-5 pb-5 -mx-5">
              <Button variant="outline" onClick={handleClose}>
                关闭
              </Button>
              <Button onClick={onSuccess}>
                完成
              </Button>
            </DialogFooter>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
