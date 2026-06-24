/**
 * ErrorState — 错误状态
 * 用 lucide AlertCircle + 错误文本 + 重试按钮
 */
import { AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface Props {
  message: string;
  onRetry?: () => void;
  className?: string;
}

export function ErrorState({ message, onRetry, className }: Props) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-3 rounded border border-danger/30 bg-danger/5 p-6 text-center",
        className
      )}
    >
      <AlertCircle className="size-5 text-danger" />
      <div className="text-sm text-fg">{message}</div>
      {onRetry && (
        <Button variant="outline" size="sm" onClick={onRetry}>
          重试
        </Button>
      )}
    </div>
  );
}
