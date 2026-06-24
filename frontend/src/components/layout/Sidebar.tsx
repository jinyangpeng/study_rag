import { Link, useLocation } from "react-router-dom";
import {
  LayoutDashboard,
  Database,
  Search,
  LineChart,
  Settings,
  ListTodo,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/dashboard", icon: LayoutDashboard, label: "系统状态" },
  { to: "/kbs", icon: Database, label: "知识库" },
  { to: "/search", icon: Search, label: "检索测试" },
  { to: "/jobs", icon: ListTodo, label: "异步任务" },
  { to: "/metrics", icon: LineChart, label: "Metrics" },
  { to: "/settings", icon: Settings, label: "设置" },
];

export function Sidebar() {
  const { pathname } = useLocation();
  return (
    <aside className="flex h-screen w-56 shrink-0 flex-col border-r border-border bg-bg-secondary">
      <Link
        to="/dashboard"
        className="flex h-12 items-center gap-2 border-b border-border px-4"
      >
        <div className="flex size-6 items-center justify-center rounded bg-accent text-xs font-bold text-accent-fg">
          R
        </div>
        <span className="text-sm font-semibold">study_rag</span>
      </Link>
      <nav className="flex-1 space-y-0.5 p-2">
        {NAV.map(({ to, icon: Icon, label }) => {
          const active = pathname.startsWith(to);
          return (
            <Link
              key={to}
              to={to}
              className={cn(
                "flex items-center gap-2.5 rounded px-2.5 py-1.5 text-sm transition-colors",
                active
                  ? "bg-bg-tertiary text-fg"
                  : "text-fg-secondary hover:bg-bg-tertiary hover:text-fg"
              )}
            >
              <Icon className="size-4" />
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="border-t border-border p-3 text-[10px] text-fg-muted">
        v0.1.0 · Admin UI
      </div>
    </aside>
  );
}
