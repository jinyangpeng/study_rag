import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import http from "node:http";
import dns from "node:dns";

/**
 * 创建一个 http.Agent，对 localhost 强制走 IPv4 (127.0.0.1)。
 *
 * 背景：
 *   - Windows 上 Node 把 `localhost` 默认解析成 ::1 (IPv6) 优先；
 *   - 但开发常见后端启动方式是 `uvicorn --host 127.0.0.1 --port 8765`（只 IPv4）；
 *   - Vite proxy 不带 agent 时会按 Node 默认走 IPv6，连接被拒 → 500 (空 body)。
 *
 * 行为：
 *   - target 是 `localhost` / `::1`   → 强制 family=4 → 实际连 127.0.0.1
 *   - target 是 `127.0.0.1`            → 直接用，不需要 DNS 解析
 *   - target 是其他域名（如 LAN IP）   → 不做特殊处理，按 OS 默认解析
 *
 * 这样 VITE_API_PROXY 配 `http://localhost:8765` 或 `http://127.0.0.1:8765` 都能通。
 */
function ipv4FirstAgent(): http.Agent {
  return new http.Agent({
    keepAlive: true,
    lookup: (
      hostname: string,
      options: dns.LookupOptions,
      callback: (
        err: NodeJS.ErrnoException | null,
        address: string | dns.LookupAddress[],
        family: number
      ) => void
    ) => {
      const isLocalhost =
        hostname === "localhost" || hostname === "::1" || hostname === "[::1]";
      dns.lookup(
        hostname,
        { ...options, family: isLocalhost ? 4 : options.family ?? 0 },
        callback
      );
    },
  });
}

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  // dev 模式默认 proxy target；同时支持 http://localhost:8765 和 http://127.0.0.1:8765
  //   （前者靠 ipv4FirstAgent 把 ::1 解析改成 127.0.0.1，后者直接用）
  const proxyTarget = env.VITE_API_PROXY ?? "http://localhost:8765";
  const agent = ipv4FirstAgent();

  return {
    plugins: [react()],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    server: {
      port: 5173,
      proxy: {
        "/admin": {
          target: proxyTarget,
          changeOrigin: true,
          agent,
        },
        "/health": {
          target: proxyTarget,
          changeOrigin: true,
          agent,
        },
        "/metrics": {
          target: proxyTarget,
          changeOrigin: true,
          agent,
        },
        "/mcp": {
          target: proxyTarget,
          changeOrigin: true,
          agent,
        },
      },
    },
    // SPA 部署在 /admin/ui/ 下，所有静态资源需要用相对路径 /admin/ui/assets/...
    // dev 模式用 "/" 即可（vite 会从 dev server 根路径提供）
    base: mode === "production" ? "/admin/ui/" : "/",
    build: {
      // 产物输出到 Python 包内（让 wheel 自带前端）
      outDir: path.resolve(__dirname, "../src/study_rag/web/dist"),
      emptyOutDir: true,
      sourcemap: mode !== "production",
      rollupOptions: {
        output: {
          manualChunks: {
            "react-vendor": ["react", "react-dom", "react-router-dom"],
            "antd-vendor": ["antd", "@ant-design/icons"],
          },
        },
      },
    },
  };
});
