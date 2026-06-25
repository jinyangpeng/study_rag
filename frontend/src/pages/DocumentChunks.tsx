/**
 * DocumentChunks — 文档分块查看页
 *
 * 显示文档的所有分块，支持分页加载
 */
import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { useApi } from "@/api/client";
import type { ChunkInfo } from "@/api/types";
import { toast } from "sonner";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { ArrowLeft, Blocks, Database } from "lucide-react";

export default function DocumentChunks() {
  const params = useParams<{ kbId: string; docId: string }>();
  const kbId = params.kbId;
  const docId = params.docId;
  const { client } = useApi();

  const [chunks, setChunks] = useState<ChunkInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [page, setPage] = useState(0);
  const [total, setTotal] = useState(0);

  // 加载分块
  useEffect(() => {
    if (kbId && docId) {
      void loadChunks(true);
    }
  }, [kbId, docId]);

  async function loadChunks(reset = false) {
    if (!kbId || !docId) return;

    if (reset) {
      setChunks([]);
      setPage(0);
      setHasMore(true);
      setLoading(true);
    } else {
      setLoadingMore(true);
    }

    try {
      const limit = 20;
      const offset = reset ? 0 : page * limit;
      const response = await client.listDocumentChunks(kbId, docId, limit, offset);

      if (reset) {
        setChunks(response.chunks);
        setTotal(response.total);
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
      setLoading(false);
      setLoadingMore(false);
    }
  }

  if (loading) {
    return (
      <div className="space-y-4">
        <Button variant="ghost" asChild>
          <Link to={`/kbs/${kbId}/documents`}>
            <ArrowLeft className="size-4 mr-2" />
            返回文档列表
          </Link>
        </Button>
        <Card>
          <CardContent className="p-6">
            <div className="space-y-3">
              <Skeleton className="h-4 w-32" />
              <Skeleton className="h-4 w-48" />
              <div className="space-y-2">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Skeleton key={i} className="h-16" />
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* 返回按钮 */}
      <Button variant="ghost" asChild>
        <Link to={`/kbs/${kbId}/documents`}>
          <ArrowLeft className="size-4 mr-2" />
          返回文档列表
        </Link>
      </Button>

      {/* 文档头部信息（紧凑一行） */}
      <Card>
        <CardContent className="p-3">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
            <div className="flex items-center gap-1.5">
              <Blocks className="size-3.5 text-accent" />
              <span className="font-mono font-semibold">{docId}</span>
            </div>
            <span className="text-fg-muted">@</span>
            <span className="font-mono text-fg-muted">{kbId}</span>
            <Badge variant="outline" className="font-mono">
              {total} 个分块
            </Badge>
          </div>
        </CardContent>
      </Card>

      {/* 分块列表 */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">分块详情</CardTitle>
        </CardHeader>
        <CardContent>
          {chunks.length === 0 ? (
            <EmptyState
              title="没有分块数据"
              description="该文档还没有被分块或分块数据为空"
              icon={Database}
            />
          ) : (
            <div className="space-y-3">
              {chunks.map((chunk, idx) => (
                <div
                  key={`${chunk.chunk_id}-${chunk.chunk_index}-${idx}`}
                  className="rounded border border-border-subtle bg-bg-tertiary p-4"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <Badge variant="outline" className="font-normal">
                      #{chunk.chunk_index}
                    </Badge>
                    <span className="text-sm text-fg-muted">
                      {chunk.char_count} 字符
                    </span>
                  </div>
                  <div className="text-sm text-fg-secondary whitespace-pre-wrap">
                    {chunk.text}
                  </div>
                </div>
              ))}

              {hasMore && (
                <div className="flex justify-center py-2">
                  <Button
                    variant="outline"
                    onClick={() => void loadChunks()}
                    disabled={loadingMore}
                  >
                    {loadingMore ? "加载中..." : "加载更多"}
                  </Button>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
