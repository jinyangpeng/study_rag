/**
 * LoadingState — 加载中状态
 * 用 Skeleton + 文案
 */
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

interface Props {
  rows?: number;
  className?: string;
  label?: string;
}

export function LoadingState({ rows = 4, className, label = "加载中..." }: Props) {
  return (
    <div className={cn("space-y-3", className)}>
      <div className="text-xs text-fg-muted">{label}</div>
      <div className="space-y-2">
        {Array.from({ length: rows }).map((_, i) => (
          <Skeleton
            key={i}
            className="h-9 w-full"
            style={{ width: `${100 - (i % 3) * 6}%` }}
          />
        ))}
      </div>
    </div>
  );
}
