/**
 * DocumentDetailDialog — 文档详情弹窗
 *
 * 显示文档元数据 + 分块列表
 */
import { useEffect, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { useApi } from "@/api/client";
import type { DocumentMeta, ChunkInfo } from "@/api/types";
import { formatChars, formatRelativeTime } from "@/lib/utils";
import { toast } from "sonner";
import { FileText, Blocks, Database, Hash, Link2, Calendar } from "lucide-react";

interface Props {
  open: boolean;
  kbId: string;
  docId: string;
  onCancel: () => void;
}

export default function DocumentDetailDialog({ open, kbId, docId, onCancel }: Props) {
  const { client } = useApi();
  const [doc, setDoc] = useState<DocumentMeta | null>(null);
  const [chunks, setChunks] = useState<ChunkInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [chunkLoading, setChunkLoading] = useState(false);
  const [page, setPage] = useState(0);
  const [hasMore, setHasMore] = useState(true);

  // 加载文档详情
  useEffect(() => {
    if (!open || !kbId || !docId) return;
    void loadDoc();
  }, [open, kbId, docId]);

  async function loadDoc() {
    setLoading(true);
    try {
      const d = await client.getDocument(kbId, docId);
      setDoc(d);
      // 重置分页
      setPage(0);
      setHasMore(true);
      void loadChunks();
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  // 加载分块
  async function loadChunks(reset = false) {
    if (!docId || !kbId) return;

    if (reset) {
      setChunks([]);
      setPage(0);
      setHasMore(true);
    }

    if (!hasMore || chunkLoading) return;

    setChunkLoading(true);
    try {
      const limit = 50;
      const offset = reset ? 0 : page * limit;
      const response = await client.listDocumentChunks(kbId, docId, limit, offset);

      if (reset) {
        setChunks(response.chunks);
      } else {
        // 后端在重复数据场景下可能返回重复的 chunk_id+chunk_index，
        // 这里按 (chunk_id, chunk_index) 联合去重，避免 UI 上出现重复块。
        setChunks((prev) => {
          const seen = new Set(prev.map((c) => `${c.chunk_id}#${c.chunk_index}`));
          const merged = [...prev];
          for (const c of response.chunks) {
            const k = `${c.chunk_id}#${c.chunk_index}`;
            if (!seen.has(k)) {
              seen.add(k);
              merged.push(c);
            }
          }
          return merged;
        });
      }


      // 如果返回的数量小于请求的数量，说明没有更多了
      setHasMore(response.chunks.length === limit);
      if (!reset) {
        setPage(prev => prev + 1);
      }
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setChunkLoading(false);
    }
  }

  // 刷新数据
  const handleRefresh = () => {
    void loadDoc();
  };

  if (!open) return null;

  return (
    <Dialog open={open} onOpenChange={onCancel}>
      <DialogContent className="max-w-4xl p-0 gap-0 overflow-hidden">
        <DialogHeader className="px-5 pt-5 pb-3 shrink-0">
          <DialogTitle className="flex items-center gap-2">
            <FileText className="size-3.5 text-accent" />
            文档详情
          </DialogTitle>
          <DialogDescription>
            {doc?.title || docId}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 overflow-y-auto px-5 py-3 min-h-0 flex-1">
          {/* 基本信息 */}
          {loading ? (
            <div className="space-y-2">
              <div className="h-4 w-3/4 bg-bg-secondary rounded animate-pulse" />
              <div className="h-3 w-1/2 bg-bg-secondary rounded animate-pulse" />
              <div className="h-3 w-1/3 bg-bg-secondary rounded animate-pulse" />
            </div>
          ) : doc ? (
            <div className="space-y-3">
              {/* 基本信息（紧凑一行） */}
              <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
                <div className="flex items-center gap-1.5">
                  <FileText className="size-3.5 text-accent" />
                  <span className="font-mono font-semibold">{doc.doc_id}</span>
                </div>
                <span className="font-medium text-fg">{doc.title}</span>
                <Badge variant="outline" className="font-mono">
                  <Hash className="size-3 mr-1" />
                  {doc.chunk_count ?? 0} 分块
                </Badge>
                <span className="font-mono text-fg-muted">
                  {formatChars(doc.char_count ?? 0)} 字符
                </span>
                {doc.parser && (
                  <Badge variant="muted" className="font-mono text-[10px]">
                    {doc.parser}
                  </Badge>
                )}
                {doc.source && (
                  <Badge variant="secondary" className="font-normal">
                    <Link2 className="size-3 mr-1" />
                    {doc.source}
                  </Badge>
                )}
                <span className="ml-auto flex items-center gap-1 text-[10px] text-fg-muted">
                  <Calendar className="size-3" />
                  {formatRelativeTime(doc.created_at)}
                </span>
              </div>

              <Separator />

              <div className="space-y-2">
                <div className="text-xs text-fg-muted">元数据</div>
                <div className="rounded border border-border-subtle bg-bg-tertiary p-3 text-xs">
                  <pre className="whitespace-pre-wrap break-words">
                    {JSON.stringify(doc.metadata, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-4 text-sm text-fg-muted">
              文档加载失败
            </div>
          )}

          {/* 分块列表 */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-xs">
                <Blocks className="size-3" />
                <span>分块列表</span>
                <Badge variant="outline" className="font-normal">
                  {chunks.length} 个
                </Badge>
                {doc?.parser && (
                  <Badge variant="muted" className="font-mono text-[10px]">
                    {doc.parser}
                  </Badge>
                )}
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => void loadChunks(true)}
                disabled={chunkLoading}
              >
                刷新
              </Button>
            </div>

            <ScrollArea className="h-96">
              {chunkLoading && chunks.length === 0 ? (
                <div className="space-y-2 p-4">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <div key={i} className="h-16 bg-bg-secondary rounded animate-pulse" />
                  ))}
                </div>
              ) : chunks.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-8 text-sm text-fg-muted">
                  <Database className="size-6 mb-2" />
                  没有分块数据
                </div>
              ) : (
                <div className="space-y-2 p-2">
                  {chunks.map((chunk, idx) => (
                    <div
                      key={`${chunk.chunk_id}-${chunk.chunk_index}-${idx}`}
                      className="rounded border border-border-subtle bg-bg-tertiary p-3"
                    >
                      <div className="flex items-center gap-2 mb-2 text-xs">
                        <Badge variant="outline" className="font-normal">
                          #{chunk.chunk_index}
                        </Badge>
                        <span className="text-fg-muted">
                          {chunk.char_count} 字符
                        </span>
                      </div>
                      <div className="text-xs text-fg-secondary whitespace-pre-wrap">
                        {chunk.text}
                      </div>
                    </div>
                  ))}

                  {hasMore && (
                    <div className="flex justify-center py-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => void loadChunks()}
                        disabled={chunkLoading}
                      >
                        {chunkLoading ? "加载中..." : "加载更多"}
                      </Button>
                    </div>
                  )}
                </div>
              )}
            </ScrollArea>
          </div>
        </div>

        <div className="flex justify-end gap-2 px-5 py-3 border-t border-border-subtle shrink-0 bg-bg-secondary">
          <Button variant="ghost" onClick={onCancel}>
            关闭
          </Button>
          <Button variant="outline" onClick={handleRefresh}>
            刷新
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
