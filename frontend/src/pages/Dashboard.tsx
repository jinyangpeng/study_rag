/**
 * Dashboard — Phase 1 占位符
 *
 * Phase 2 会在这个位置重写整个 dashboard 页面
 * （StatCards + activity feed + circuit breaker 状态）。
 *
 * 现在保留 default export 以兼容路由。
 */
export default function Dashboard() {
  return (
    <div className="flex h-64 items-center justify-center rounded-lg border border-dashed border-border bg-bg-secondary text-sm text-fg-muted">
      <div className="text-center">
        <div className="text-base font-medium text-fg">Dashboard</div>
        <div className="mt-1 text-xs text-fg-muted">
          Phase 2 will rewrite this page (StatCards + activity feed)
        </div>
      </div>
    </div>
  );
}
