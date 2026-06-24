import { Command } from "cmdk";
import { useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import {
  Database,
  Search,
  ListTodo,
  Settings,
  LayoutDashboard,
  LineChart,
} from "lucide-react";

const ITEMS = [
  { to: "/dashboard", label: "系统状态", icon: LayoutDashboard, group: "导航" },
  { to: "/kbs", label: "知识库", icon: Database, group: "导航" },
  { to: "/search", label: "检索测试", icon: Search, group: "导航" },
  { to: "/jobs", label: "异步任务", icon: ListTodo, group: "导航" },
  { to: "/metrics", label: "Metrics", icon: LineChart, group: "导航" },
  { to: "/settings", label: "设置", icon: Settings, group: "导航" },
];

export function CommandPalette({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const nav = useNavigate();
  const [query, setQuery] = useState("");

  // Global keyboard shortcut: ⌘K / Ctrl+K to open, Esc to close
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        if (!open) {
          // parent owns open state — we can't open from here directly,
          // so just close (no-op when already closed)
        } else {
          onClose();
        }
      } else if (e.key === "Escape" && open) {
        onClose();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  // reset query on open
  useEffect(() => {
    if (open) setQuery("");
  }, [open]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 pt-32 animate-fade-in"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
    >
      <Command
        onClick={(e) => e.stopPropagation()}
        className="w-[480px] overflow-hidden rounded-lg border border-border bg-bg-secondary shadow-2xl"
      >
        <div className="flex items-center border-b border-border px-3">
          <Search className="mr-2 size-4 text-fg-muted" />
          <input
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="输入跳转..."
            className="h-10 w-full bg-transparent text-sm text-fg outline-none placeholder:text-fg-muted"
          />
          <kbd className="rounded border border-border bg-bg-tertiary px-1 text-[10px] font-mono text-fg-muted">
            ESC
          </kbd>
        </div>
        <Command.List className="max-h-72 overflow-y-auto p-1">
          <Command.Empty className="px-3 py-6 text-center text-xs text-fg-muted">
            无结果
          </Command.Empty>
          {ITEMS.map(({ to, label, icon: Icon }) => (
            <Command.Item
              key={to}
              value={label}
              onSelect={() => {
                nav(to);
                onClose();
              }}
              className="flex cursor-pointer items-center gap-2.5 rounded px-2.5 py-1.5 text-sm text-fg aria-selected:bg-bg-tertiary"
            >
              <Icon className="size-4 text-fg-muted" />
              <span>{label}</span>
            </Command.Item>
          ))}
        </Command.List>
        <div className="flex items-center justify-between border-t border-border px-3 py-2 text-[10px] text-fg-muted">
          <span>{ITEMS.length} 项</span>
          <span>↑↓ 选择 · ↵ 进入</span>
        </div>
      </Command>
    </div>
  );
}
