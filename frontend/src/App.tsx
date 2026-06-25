import { Suspense } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import Dashboard from "./pages/Dashboard";
import KnowledgeBases from "./pages/KnowledgeBases";
import Documents from "./pages/Documents";
import DocumentChunks from "./pages/DocumentChunks";
import SearchTest from "./pages/SearchTest";
import JobsPage from "./pages/JobsPage";
import Metrics from "./pages/Metrics";
import ModelConfigs from "./pages/ModelConfigs";
import Settings from "./pages/Settings";

export default function App() {
  return (
    <Suspense fallback={null}>
      <Routes>
        <Route
          path="/"
          element={
            <AppShell>
              <Dashboard />
            </AppShell>
          }
        />
        <Route
          path="/dashboard"
          element={
            <AppShell>
              <Dashboard />
            </AppShell>
          }
        />
        <Route
          path="/kbs"
          element={
            <AppShell>
              <KnowledgeBases />
            </AppShell>
          }
        />
        <Route
          path="/kbs/:kbId/documents"
          element={
            <AppShell>
              <Documents />
            </AppShell>
          }
        />
        <Route
          path="/kbs/:kbId/documents/:docId/chunks"
          element={
            <AppShell>
              <DocumentChunks />
            </AppShell>
          }
        />
        <Route
          path="/search"
          element={
            <AppShell>
              <SearchTest />
            </AppShell>
          }
        />
        <Route
          path="/metrics"
          element={
            <AppShell>
              <Metrics />
            </AppShell>
          }
        />
        <Route
          path="/models"
          element={
            <AppShell>
              <ModelConfigs />
            </AppShell>
          }
        />
        <Route
          path="/jobs"
          element={
            <AppShell>
              <JobsPage />
            </AppShell>
          }
        />
        <Route
          path="/settings"
          element={
            <AppShell>
              <Settings />
            </AppShell>
          }
        />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </Suspense>
  );
}
