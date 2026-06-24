/**
 * EmptyState — 占位空状态
 * 用 lucide Inbox + 文案 + 可选 CTA
 */
import { Inbox, type LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface Props {
  title?: string;
  description?: string;
  icon?: LucideIcon;
  action?: ReactNode;
  className?: string;
}

export function EmptyState({
  title = "暂无数据",
  description,
  icon: Icon = Inbox,
  action,
  className,
}: Props) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-3 py-12 text-center",
        className
      )}
    >
      <div className="flex size-10 items-center justify-center rounded-full border border-border bg-bg-tertiary">
        <Icon className="size-5 text-fg-muted" />
      </div>
      <div className="space-y-1">
        <div className="text-sm font-medium text-fg">{title}</div>
        {description && (
          <div className="text-xs text-fg-muted">{description}</div>
        )}
      </div>
      {action}
    </div>
  );
}
