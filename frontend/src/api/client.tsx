/**
 * API 客户端：基于 axios，统一处理 baseURL / 鉴权 / 错误。
 *
 * 鉴权策略（懒加载）：
 *  1. mount 时 fetch /health/detailed（这个端点不要鉴权），
 *     得到 auth_enabled = 服务端是否启用了 STUDY_RAG_ADMIN_TOKEN
 *  2. auth_enabled = false：免鉴权模式，前端不应弹任何 token 框
 *  3. auth_enabled = true：所有 /admin/* 调用会被服务端 401，
 *     全局 axios 拦截器弹模态框引导用户去「设置」配 token
 *  4. 401 弹框里的 token 保存到 localStorage；下次请求自动带 Bearer
 *
 * 设计动机：避免"明明没启用鉴权，前端却骚扰用户配 token"的反模式。
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import axios, {
  AxiosError,
  type AxiosInstance,
  type AxiosRequestConfig,
} from "axios";
import { useNavigate } from "react-router-dom";
import { AuthPromptDialog } from "@/components/AuthPromptDialog";
import { toast } from "sonner";
import type {
  ChunkPreviewResponse,
  ChunksListResponse,
  DocumentCreate,
  DocumentChunkedCreate,
  DocumentMeta,
  EmbedderConfigCreate,
  EmbedderConfigItem,
  EmbedderConfigUpdate,
  EmbedderInfo,
  HealthDetailed,
  JobInfo,
  CancelJobResponse,
  KnowledgeBaseConfig,
  KnowledgeBaseCreate,
  KnowledgeBaseSummary,
  KnowledgeBaseUpdate,
  ParserConfigCreate,
  ParserConfigItem,
  ParserConfigUpdate,
  ParserSpec,
  RerankerConfigCreate,
  RerankerConfigItem,
  RerankerConfigUpdate,
  RerankerInfo,
  SearchResponse,
  UploadDocumentResponse,
} from "./types";

const TOKEN_STORAGE_KEY = "study_rag_admin_token";
const BASE_URL_STORAGE_KEY = "study_rag_api_base_url";

function getStoredToken(): string {
  return localStorage.getItem(TOKEN_STORAGE_KEY) ?? "";
}

function getStoredBaseURL(): string {
  // 优先用用户显式配置；否则用当前页面 origin（生产 build 嵌入 FastAPI）
  const stored = localStorage.getItem(BASE_URL_STORAGE_KEY);
  if (stored) return stored;
  // dev 模式（vite dev 5173）：用相对路径，让 vite proxy 转发
  if (import.meta.env.DEV) return "";
  // 生产：当前 origin
  return window.location.origin;
}

class ApiClient {
  private http: AxiosInstance;
  private token: string;
  private baseURL: string;

  constructor() {
    this.token = getStoredToken();
    this.baseURL = getStoredBaseURL();
    this.http = axios.create({
      baseURL: this.baseURL,
      timeout: 30_000,
      headers: { "Content-Type": "application/json" },
    });
    this.http.interceptors.request.use((cfg) => {
      if (this.token) {
        cfg.headers.Authorization = `Bearer ${this.token}`;
      }
      return cfg;
    });
  }

  setToken(token: string) {
    this.token = token;
    if (token) {
      localStorage.setItem(TOKEN_STORAGE_KEY, token);
    } else {
      localStorage.removeItem(TOKEN_STORAGE_KEY);
    }
  }

  getToken(): string {
    return this.token;
  }

  hasToken(): boolean {
    return this.token.length > 0;
  }

  setBaseURL(url: string) {
    this.baseURL = url;
    if (url) {
      localStorage.setItem(BASE_URL_STORAGE_KEY, url);
    } else {
      localStorage.removeItem(BASE_URL_STORAGE_KEY);
    }
    this.http.defaults.baseURL = this.baseURL;
  }

  getBaseURL(): string {
    return this.baseURL;
  }

  private unwrapError(e: unknown): string {
    if (axios.isAxiosError(e)) {
      const data = e.response?.data as { detail?: string | unknown } | undefined;
      if (data && typeof data.detail === "string") return data.detail;
      if (e.response?.status === 401) return "鉴权失败 (401)";
      if (e.response?.status === 429) return "触发限流 (429)，请稍后再试";
      if (e.response?.status === 404) return "资源不存在 (404)";
      return e.message;
    }
    return String(e);
  }

  // ===== KB =====

  async listKBs(): Promise<KnowledgeBaseSummary[]> {
    try {
      const { data } = await this.http.get<KnowledgeBaseSummary[]>(
        "/admin/kbs"
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async getKB(kbId: string): Promise<KnowledgeBaseSummary> {
    try {
      const { data } = await this.http.get<KnowledgeBaseSummary>(
        `/admin/kbs/${encodeURIComponent(kbId)}`
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async createKB(payload: KnowledgeBaseCreate): Promise<KnowledgeBaseConfig> {
    try {
      const { data } = await this.http.post<KnowledgeBaseConfig>(
        "/admin/kbs",
        payload
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async updateKB(
    kbId: string,
    patch: KnowledgeBaseUpdate
  ): Promise<KnowledgeBaseConfig> {
    try {
      const { data } = await this.http.patch<KnowledgeBaseConfig>(
        `/admin/kbs/${encodeURIComponent(kbId)}`,
        patch
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async deleteKB(kbId: string): Promise<KnowledgeBaseConfig> {
    try {
      const { data } = await this.http.delete<KnowledgeBaseConfig>(
        `/admin/kbs/${encodeURIComponent(kbId)}`
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async listEmbedders(): Promise<EmbedderInfo[]> {
    try {
      const { data } = await this.http.get<EmbedderInfo[]>("/admin/embedders");
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async listRerankers(): Promise<RerankerInfo[]> {
    try {
      const { data } = await this.http.get<RerankerInfo[]>("/admin/rerankers");
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  // ===== Embedder 配置管理（CRUD）=====

  async listEmbedderConfigs(): Promise<EmbedderConfigItem[]> {
    try {
      const { data } = await this.http.get<EmbedderConfigItem[]>(
        "/admin/embedders/configs"
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async createEmbedderConfig(
    payload: EmbedderConfigCreate
  ): Promise<EmbedderConfigItem> {
    try {
      const { data } = await this.http.post<EmbedderConfigItem>(
        "/admin/embedders/configs",
        payload
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async updateEmbedderConfig(
    name: string,
    patch: EmbedderConfigUpdate
  ): Promise<EmbedderConfigItem> {
    try {
      const { data } = await this.http.put<EmbedderConfigItem>(
        `/admin/embedders/configs/${encodeURIComponent(name)}`,
        patch
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async deleteEmbedderConfig(name: string): Promise<void> {
    try {
      await this.http.delete(`/admin/embedders/configs/${encodeURIComponent(name)}`);
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  // ===== Reranker 配置管理（CRUD）=====

  async listRerankerConfigs(): Promise<RerankerConfigItem[]> {
    try {
      const { data } = await this.http.get<RerankerConfigItem[]>(
        "/admin/rerankers/configs"
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async createRerankerConfig(
    payload: RerankerConfigCreate
  ): Promise<RerankerConfigItem> {
    try {
      const { data } = await this.http.post<RerankerConfigItem>(
        "/admin/rerankers/configs",
        payload
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async updateRerankerConfig(
    name: string,
    patch: RerankerConfigUpdate
  ): Promise<RerankerConfigItem> {
    try {
      const { data } = await this.http.put<RerankerConfigItem>(
        `/admin/rerankers/configs/${encodeURIComponent(name)}`,
        patch
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async deleteRerankerConfig(name: string): Promise<void> {
    try {
      await this.http.delete(`/admin/rerankers/configs/${encodeURIComponent(name)}`);
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  // ===== Parser 配置管理（CRUD）=====

  async listParserConfigs(): Promise<ParserConfigItem[]> {
    try {
      const { data } = await this.http.get<ParserConfigItem[]>(
        "/admin/parsers/configs"
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async createParserConfig(
    payload: ParserConfigCreate
  ): Promise<ParserConfigItem> {
    try {
      const { data } = await this.http.post<ParserConfigItem>(
        "/admin/parsers/configs",
        payload
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async updateParserConfig(
    name: string,
    patch: ParserConfigUpdate
  ): Promise<ParserConfigItem> {
    try {
      const { data } = await this.http.put<ParserConfigItem>(
        `/admin/parsers/configs/${encodeURIComponent(name)}`,
        patch
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async deleteParserConfig(name: string): Promise<void> {
    try {
      await this.http.delete(`/admin/parsers/configs/${encodeURIComponent(name)}`);
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  // ===== Document =====

  async listDocuments(kbId: string): Promise<DocumentMeta[]> {
    try {
      const { data } = await this.http.get<DocumentMeta[]>(
        `/admin/kbs/${encodeURIComponent(kbId)}/documents`
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async getDocument(kbId: string, docId: string): Promise<DocumentMeta> {
    try {
      const { data } = await this.http.get<DocumentMeta>(
        `/admin/kbs/${encodeURIComponent(kbId)}/documents/${encodeURIComponent(docId)}`
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async addDocument(doc: DocumentCreate): Promise<DocumentMeta> {
    try {
      const { data } = await this.http.post<DocumentMeta>(
        `/admin/kbs/${encodeURIComponent(doc.kb_id)}/documents`,
        doc
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async addDocumentChunked(doc: DocumentChunkedCreate): Promise<DocumentMeta> {
    try {
      const { data } = await this.http.post<DocumentMeta>(
        `/admin/kbs/${encodeURIComponent(doc.kb_id)}/documents/chunked`,
        doc
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async listParsers(): Promise<ParserSpec[]> {
    try {
      const { data } = await this.http.get<ParserSpec[]>("/admin/parsers");
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async previewChunk(
    kbId: string,
    content: string,
    parser: string,
    title = "preview"
  ): Promise<ChunkPreviewResponse> {
    try {
      const { data } = await this.http.post<ChunkPreviewResponse>(
        `/admin/kbs/${encodeURIComponent(kbId)}/documents/preview-chunk`,
        { content, parser, title }
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async uploadDocument(
    kbId: string,
    form: FormData
  ): Promise<UploadDocumentResponse> {
    try {
      const { data } = await this.http.post<UploadDocumentResponse>(
        `/admin/kbs/${encodeURIComponent(kbId)}/documents/upload`,
        form,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  // ===== Jobs（异步任务） =====

  async getJob(jobId: string): Promise<JobInfo> {
    try {
      const { data } = await this.http.get<JobInfo>(
        `/admin/jobs/${encodeURIComponent(jobId)}`
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async listJobs(kbId?: string): Promise<JobInfo[]> {
    try {
      const { data } = await this.http.get<JobInfo[]>("/admin/jobs", {
        params: kbId ? { kb_id: kbId } : undefined,
      });
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async cancelJob(jobId: string): Promise<CancelJobResponse> {
    try {
      const { data } = await this.http.delete<CancelJobResponse>(
        `/admin/jobs/${encodeURIComponent(jobId)}`
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async deleteDocument(
    kbId: string,
    docId: string
  ): Promise<{ status: string; kb_id: string; doc_id: string }> {
    try {
      const { data } = await this.http.delete<{
        status: string;
        kb_id: string;
        doc_id: string;
      }>(
        `/admin/kbs/${encodeURIComponent(kbId)}/documents/${encodeURIComponent(docId)}`
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async addDocumentsBatch(
    kbId: string,
    payload: {
      documents: Array<{
        doc_id: string;
        title: string;
        content: string;
        source?: string | null;
        metadata?: Record<string, unknown>;
      }>;
      overwrite?: boolean;
    }
  ): Promise<{
    succeeded: string[];
    failed: Array<{ doc_id: string; error: string }>;
    counts: { ok: number; fail: number };
  }> {
    try {
      const { data } = await this.http.post<{
        succeeded: string[];
        failed: Array<{ doc_id: string; error: string }>;
        counts: { ok: number; fail: number };
      }>(
        `/admin/kbs/${encodeURIComponent(kbId)}/documents/batch`,
        payload
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async listDocumentChunks(
    kbId: string,
    docId: string,
    limit = 100,
    offset = 0
  ): Promise<ChunksListResponse> {
    try {
      const { data } = await this.http.get<ChunksListResponse>(
        `/admin/kbs/${encodeURIComponent(kbId)}/documents/${encodeURIComponent(docId)}/chunks`,
        { params: { limit, offset } }
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  // ===== Search =====

  async search(
    kbId: string,
    body: {
      query: string;
      /** 返回数量；null = 跟随 reranker 配置的 top_k */
      top_k?: number | null;
      use_rerank?: boolean;
      /** 覆盖 KB 默认 reranker；null/undefined 表示用 KB 配置的 */
      reranker_name?: string | null;
      filter_expr?: Record<string, unknown> | null;
    }
  ): Promise<SearchResponse> {
    try {
      const { data } = await this.http.post<SearchResponse>(
        `/admin/kbs/${encodeURIComponent(kbId)}/search`,
        body
      );
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  // ===== Health & Metrics =====

  async getHealthDetailed(): Promise<HealthDetailed> {
    try {
      const { data } = await this.http.get<HealthDetailed>("/health/detailed");
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  async getMetricsText(): Promise<string> {
    try {
      const { data } = await this.http.get<string>("/metrics", {
        responseType: "text",
        transformResponse: [(d) => d],
      });
      return data;
    } catch (e) {
      throw new Error(this.unwrapError(e));
    }
  }

  // ===== Raw HTTP (escape hatch) =====

  raw<T = unknown>(config: AxiosRequestConfig): Promise<T> {
    return this.http.request<T>(config).then((r) => r.data);
  }
}

// ============ React Context ============

interface ApiContextValue {
  client: ApiClient;
  token: string;
  setToken: (t: string) => void;
  baseURL: string;
  setBaseURL: (u: string) => void;
  hasToken: boolean;
  /** 服务端是否启用了鉴权（未配置时为 false，前端不要主动弹 token 框） */
  authEnabled: boolean | null;
  /** 重新探测服务端鉴权状态 */
  refreshAuthStatus: () => Promise<void>;
  /** 服务端真要鉴权但客户端没配 token 时，引导用户去「设置」 */
  promptAuth: () => Promise<boolean>;
  /**
   * 前端代理不通的错误（Vite proxy 失败 / 后端 5xx / 网络错）。
   * MainLayout 看到这个就显示顶部 Alert，避免用户对着空白 Dashboard 一脸懵。
   * null = 没出错（或已恢复）。
   */
  proxyError: string | null;
  clearProxyError: () => void;
}

const ApiContext = createContext<ApiContextValue | null>(null);

export function ApiProvider({ children }: { children: ReactNode }) {
  const navigate = useNavigate();
  const [client] = useState(() => new ApiClient());
  const [token, setTokenState] = useState(() => client.getToken());
  const [baseURL, setBaseURLState] = useState(() => client.getBaseURL());
  const [hasToken, setHasToken] = useState(() => client.hasToken());
  // null = 探测中；true/false = 已确认
  const [authEnabled, setAuthEnabled] = useState<boolean | null>(null);
  // 前端代理/网络层不通的错误（详见 ApiContextValue.proxyError）
  const [proxyError, setProxyError] = useState<string | null>(null);
  // 鉴权引导弹窗状态
  const [authPromptOpen, setAuthPromptOpen] = useState(false);
  const [authPromptValue, setAuthPromptValue] = useState("");
  const authPromptResolverRef = useRef<((v: boolean) => void) | null>(null);

  const setToken = useCallback(
    (t: string) => {
      client.setToken(t);
      setTokenState(t);
      setHasToken(client.hasToken());
    },
    [client]
  );

  const setBaseURL = useCallback(
    (u: string) => {
      client.setBaseURL(u);
      setBaseURLState(client.getBaseURL());
    },
    [client]
  );

  /** 探测服务端鉴权状态：/health/detailed 永远免鉴权 */
  const refreshAuthStatus = useCallback(async () => {
    try {
      const h = await client.getHealthDetailed();
      setAuthEnabled(h.auth_enabled);
      setProxyError(null); // 探测成功 → 清掉之前的代理错误
    } catch (e) {
      // 探测失败时保守当作未启用，不要骚扰用户去配 token
      setAuthEnabled(false);

      // 区分错误类型：网络/代理错（看不到后端） vs 业务错（看到后端但 4xx/5xx）
      //  - 看不到后端：典型场景是 Vite 代理配置错（localhost 解析到 ::1 失败、proxy target 端口错、uvicorn 没起）
      //  - 看到后端但 5xx：后端进程在但内部崩了
      //  - 4xx：通常是路径错（如 /admin/knowledge-bases），单独处理不算代理错
      let proxyMsg: string | null = null;
      if (axios.isAxiosError(e)) {
        const ax = e as AxiosError;
        if (!ax.response) {
          // 完全连不上（Vite proxy 拿到 500 空 body / 502 / network error）
          const baseHint =
            "无法连接到后端 API。" +
            " dev 模式下通常是 Vite 代理配置问题（请检查 frontend/vite.config.ts 的 proxy target）。";
          proxyMsg = `${baseHint}\n\naxios 错误：${ax.message}` +
            (ax.code ? ` (code=${ax.code})` : "");
        } else if (ax.response.status >= 500) {
          // 后端崩了或代理转发时后端挂掉
          proxyMsg =
            `后端返回 ${ax.response.status} ${ax.response.statusText || ""}。` +
            " 可能是后端进程崩溃或代理转发失败。";
        }
        // 4xx（如 404）不算代理错，留给具体页面的 error 处理
      } else {
        proxyMsg = `未知错误：${(e as Error).message}`;
      }
      setProxyError(proxyMsg);
      // eslint-disable-next-line no-console
      console.warn("refreshAuthStatus failed:", e);
    }
  }, [client]);

  const clearProxyError = useCallback(() => setProxyError(null), []);

  // mount 时探测一次；token 变化也重探（用户配了 token 之后看是否还 401）
  useEffect(() => {
    void refreshAuthStatus();
  }, [refreshAuthStatus, token]);

  /** 服务端真要鉴权但客户端没配 token 时弹的引导框（带跳设置页按钮） */
  const promptAuth = useCallback(async (): Promise<boolean> => {
    if (client.hasToken()) return true;
    return new Promise<boolean>((resolve) => {
      authPromptResolverRef.current = resolve;
      setAuthPromptValue("");
      setAuthPromptOpen(true);
    });
  }, [client]);

  /** AuthPromptDialog 回调：保存 token */
  const handleAuthSave = useCallback(() => {
    if (!authPromptValue) {
      toast.warning("Token 不能为空");
      authPromptResolverRef.current?.(false);
      authPromptResolverRef.current = null;
      setAuthPromptOpen(false);
      return;
    }
    setToken(authPromptValue);
    toast.success("Token 已保存");
    authPromptResolverRef.current?.(true);
    authPromptResolverRef.current = null;
    setAuthPromptOpen(false);
  }, [authPromptValue, setToken]);

  /** AuthPromptDialog 回调：取消 */
  const handleAuthCancel = useCallback(() => {
    authPromptResolverRef.current?.(false);
    authPromptResolverRef.current = null;
    setAuthPromptOpen(false);
  }, []);

  /** AuthPromptDialog 回调：跳到设置页 */
  const handleAuthGoToSettings = useCallback(() => {
    authPromptResolverRef.current?.(false);
    authPromptResolverRef.current = null;
    setAuthPromptOpen(false);
    navigate("/settings");
  }, [navigate]);

  // 全局 axios 错误 → toast + 401 弹引导框
  useEffect(() => {
    const id = axios.interceptors.response.use(
      (r) => r,
      (err: AxiosError) => {
        if (err.response?.status === 401) {
          // 免鉴权模式下服务端 401 通常是其他原因（路径错等），只提示不弹窗
          if (authEnabled) {
            void promptAuth();
          } else {
            toast.error("鉴权失败 (401)，但服务端未启用鉴权，请检查 API 路径");
          }
        } else if (err.response?.status === 429) {
          toast.warning("触发限流 (429)，请稍后再试");
        } else if (err.response && err.response.status >= 500) {
          toast.error(`服务异常 (${err.response.status})`);
        }
        return Promise.reject(err);
      }
    );
    return () => axios.interceptors.response.eject(id);
  }, [authEnabled, promptAuth]);

  const value = useMemo<ApiContextValue>(
    () => ({
      client,
      token,
      setToken,
      baseURL,
      setBaseURL,
      hasToken,
      authEnabled,
      refreshAuthStatus,
      promptAuth,
      proxyError,
      clearProxyError,
    }),
    [
      client,
      token,
      setToken,
      baseURL,
      setBaseURL,
      hasToken,
      authEnabled,
      refreshAuthStatus,
      promptAuth,
      proxyError,
      clearProxyError,
    ]
  );

  return (
    <ApiContext.Provider value={value}>
      {children}
      <AuthPromptDialog
        state={{
          open: authPromptOpen,
          value: authPromptValue,
          onSave: handleAuthSave,
          onCancel: handleAuthCancel,
          onGoToSettings: handleAuthGoToSettings,
        }}
      />
    </ApiContext.Provider>
  );
}

export function useApi(): ApiContextValue {
  const ctx = useContext(ApiContext);
  if (!ctx) throw new Error("useApi must be used within ApiProvider");
  return ctx;
}
