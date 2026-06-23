import { lazy, Suspense } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { Spin } from "antd";
import MainLayout from "./components/MainLayout";

const Dashboard = lazy(() => import("./pages/Dashboard"));
const KnowledgeBases = lazy(() => import("./pages/KnowledgeBases"));
const Documents = lazy(() => import("./pages/Documents"));
const SearchTest = lazy(() => import("./pages/SearchTest"));
const Metrics = lazy(() => import("./pages/Metrics"));
const Settings = lazy(() => import("./pages/Settings"));
const JobsPage = lazy(() => import("./pages/JobsPage"));

function PageFallback() {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        minHeight: 400,
      }}
    >
      <Spin size="large" tip="加载中..." />
    </div>
  );
}

export default function App() {
  return (
    <MainLayout>
      <Suspense fallback={<PageFallback />}>
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/kbs" element={<KnowledgeBases />} />
          <Route path="/kbs/:kbId/documents" element={<Documents />} />
          <Route path="/search" element={<SearchTest />} />
          <Route path="/metrics" element={<Metrics />} />
          <Route path="/jobs" element={<JobsPage />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </Suspense>
    </MainLayout>
  );
}
