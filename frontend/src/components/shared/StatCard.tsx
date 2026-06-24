/**
 * StatCard — KPI 卡片（Linear 风格：紧凑 + icon + label + value + 副文本）
 */
import type { LucideIcon } from "lucide-react";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface Props {
  label: string;
  value: number | string;
  hint?: string;
  icon: LucideIcon;
  accent?: "default" | "success" | "warning" | "danger";
  className?: string;
}

const accentClass = {
  default: "text-fg-muted",
  success: "text-success",
  warning: "text-warning",
  danger: "text-danger",
} as const;

export function StatCard({
  label,
  value,
  hint,
  icon: Icon,
  accent = "default",
  className,
}: Props) {
  return (
    <Card className={cn("p-4", className)}>
      <div className="flex items-center justify-between">
        <span className="text-xs text-fg-muted">{label}</span>
        <Icon className={cn("size-3.5", accentClass[accent])} />
      </div>
      <div className="mt-2 text-2xl font-semibold tracking-tight text-fg">
        {value}
      </div>
      {hint && <div className="mt-1 text-[11px] text-fg-muted">{hint}</div>}
    </Card>
  );
}
