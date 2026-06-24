import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { ConfigProvider, App as AntdApp } from "antd";
import zhCN from "antd/locale/zh_CN";
import { Toaster } from "sonner";
import "antd/dist/reset.css";
import "dayjs/locale/zh-cn";
import App from "./App";
import { ApiProvider } from "./api/client";
import "./styles/globals.css";

// import.meta.env.BASE_URL = vite.config.ts 的 base
//   dev:        "/"      (vite dev server)
//   production: "/admin/ui/"  (FastAPI 挂载点)
const basename = import.meta.env.BASE_URL.replace(/\/$/, "") || "/";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ConfigProvider
      locale={zhCN}
      theme={{
        token: {
          colorPrimary: "#1677ff",
          borderRadius: 6,
        },
      }}
    >
      <AntdApp>
        <ApiProvider>
          <BrowserRouter basename={basename}>
            <App />
          </BrowserRouter>
          <Toaster position="bottom-right" theme="dark" richColors />
        </ApiProvider>
      </AntdApp>
    </ConfigProvider>
  </React.StrictMode>
);
