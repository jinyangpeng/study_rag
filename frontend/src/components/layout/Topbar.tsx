import { useLocation } from "react-router-dom";
import { Bell, Search, Sun, Moon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTheme } from "@/hooks/useTheme";

const TITLES: Record<string, string> = {
  "/dashboard": "系统状态",
  "/kbs": "知识库管理",
  "/search": "检索测试",
  "/jobs": "异步任务",
  "/metrics": "Prometheus Metrics",
  "/settings": "设置",
};

export function Topbar({ onOpenCommand }: { onOpenCommand: () => void }) {
  const { pathname } = useLocation();
  const { theme, toggleTheme } = useTheme();

  const topKey = "/" + (pathname.split("/").filter(Boolean)[0] ?? "dashboard");
  const title = TITLES[topKey] ?? "study_rag";

  return (
    <header className="flex h-12 shrink-0 items-center justify-between border-b border-border bg-bg-primary px-4">
      <h1 className="text-sm font-medium">{title}</h1>
      <div className="flex items-center gap-2">
        <button
          onClick={onOpenCommand}
          className="flex h-7 items-center gap-2 rounded border border-border bg-bg-secondary px-2.5 text-xs text-fg-muted transition-colors hover:bg-bg-tertiary hover:text-fg"
        >
          <Search className="size-3.5" />
          <span>搜索或跳转...</span>
          <kbd className="ml-4 rounded border border-border bg-bg-tertiary px-1 text-[10px] font-mono">
            ⌘K
          </kbd>
        </button>
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleTheme}
          title="切换主题"
        >
          {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
        </Button>
        <Button variant="ghost" size="icon" title="通知">
          <Bell className="size-4" />
        </Button>
      </div>
    </header>
  );
}
