/** API 通用类型定义（对齐后端 Pydantic schema）。 */

export interface KnowledgeBaseSummary {
  kb_id: string;
  name: string;
  description?: string | null;
  department?: string;
  enabled: boolean;
  document_count: number;
  chunk_count: number;
  embedder?: string | null;
  reranker?: string | null;
  retrieval_strategy?: string | null;
  vector_store?: string | null;
  collection?: string | null;
}

export interface DocumentMeta {
  kb_id: string;
  doc_id: string;
  title: string;
  content?: string;
  source?: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  /** 文档在向量库的 chunk 数（list 时由后端实时统计覆盖） */
  chunk_count?: number;
  /** 文档内容总字符数（list 时由后端从 content 长度回填） */
  char_count?: number;
  /** 分块方式：sentence_512 / whole / ... */
  parser?: string | null;
}

export interface DocumentCreate {
  kb_id: string;
  doc_id: string;
  title: string;
  content: string;
  source?: string | null;
  metadata?: Record<string, unknown>;
}

export interface DocumentChunkedCreate {
  kb_id: string;
  doc_id: string;
  title: string;
  content: string;
  source?: string | null;
  metadata?: Record<string, unknown>;
  chunk_size: number;
  chunk_overlap: number;
  /** 命名 parser（如 'sentence_512'），写入 DocumentMeta.parser */
  parser_name?: string;
}

export interface SearchHit {
  doc_id: string;
  chunk_id: string;
  score: number;
  text: string;
  metadata: Record<string, unknown>;
}

export interface SearchResponse {
  kb_id: string;
  query: string;
  strategy?: string;
  hits: SearchHit[];
  duration_ms: number;
  meta?: Record<string, unknown>;
}

export interface RatelimitStats {
  name: string;
  capacity: number;
  refill_rate: number;
  tracked_keys: number;
}

export interface CircuitBreakerStats {
  name: string;
  state: "closed" | "open" | "half_open";
  failure_count: number;
  retry_after_s: number;
}

export interface HealthDetailed {
  status: string;
  kbs_total: number;
  kbs_enabled: number;
  embedders: number;
  rerankers: number;
  ratelimit: {
    admin: RatelimitStats;
    search: RatelimitStats;
  };
  circuit_breakers: Record<string, CircuitBreakerStats>;
  auth_enabled: boolean;
  registry_loaded: boolean;
}

export interface KnowledgeBaseConfig {
  kb_id: string;
  name: string;
  description: string;
  department: string;
  collection: string;
  embedding: string;
  reranker: string | null;
  enabled: boolean;
}

export interface KnowledgeBaseCreate {
  kb_id: string;
  name: string;
  description: string;
  department: string;
  collection?: string | null;
  embedding: string;
  reranker?: string | null;
  enabled?: boolean;
}

export interface KnowledgeBaseUpdate {
  name?: string;
  description?: string;
  department?: string;
  reranker?: string | null;
  enabled?: boolean;
}

export interface EmbedderInfo {
  name: string;
  provider: string;
  model_name: string;
  dimension: number;
  batch_size: number;
  loaded: boolean;
  description: string;
}

export interface RerankerInfo {
  name: string;
  provider: string;
  model_name: string;
  top_k: number;
  loaded: boolean;
  description: string;
}

// ===== 模型配置管理（CRUD）=====

export interface EmbedderConfigItem {
  name: string;
  provider: string;
  model_name: string;
  dimension: number;
  batch_size: number;
  description: string;
  extra: Record<string, unknown>;
  loaded: boolean;
}

export interface EmbedderConfigCreate {
  name: string;
  provider: string;
  model_name?: string;
  dimension?: number;
  batch_size?: number;
  description?: string;
  extra?: Record<string, unknown>;
}

export interface EmbedderConfigUpdate {
  provider?: string;
  model_name?: string;
  dimension?: number;
  batch_size?: number;
  description?: string;
  extra?: Record<string, unknown>;
}

export interface RerankerConfigItem {
  name: string;
  provider: string;
  protocol: string;
  model_name: string;
  top_k: number;
  description: string;
  extra: Record<string, unknown>;
  loaded: boolean;
}

export interface RerankerConfigCreate {
  name: string;
  provider: string;
  protocol?: string;
  model_name?: string;
  top_k?: number;
  description?: string;
  extra?: Record<string, unknown>;
}

export interface RerankerConfigUpdate {
  provider?: string;
  protocol?: string;
  model_name?: string;
  top_k?: number;
  description?: string;
  extra?: Record<string, unknown>;
}

// ===== Parser（分块配置）CRUD =====

export interface ParserConfigItem {
  name: string;
  strategy: string;
  chunk_size: number;
  chunk_overlap: number;
  paragraph_separator: string;
  buffer_size: number | null;
  breakpoint_percentile_threshold: number | null;
  extra: Record<string, unknown>;
}

export interface ParserConfigCreate {
  name: string;
  strategy: string;
  chunk_size?: number;
  chunk_overlap?: number;
  paragraph_separator?: string;
  buffer_size?: number | null;
  breakpoint_percentile_threshold?: number | null;
  extra?: Record<string, unknown>;
}

export interface ParserConfigUpdate {
  strategy?: string;
  chunk_size?: number;
  chunk_overlap?: number;
  paragraph_separator?: string;
  buffer_size?: number | null;
  breakpoint_percentile_threshold?: number | null;
  extra?: Record<string, unknown>;
}

export interface ParserSpec {
  name: string;
  strategy: "whole" | "sentence" | "token" | "semantic";
  chunk_size: number;
  chunk_overlap: number;
  paragraph_separator: string;
}

export interface ChunkPreviewItem {
  chunk_index: number;
  text: string;
  char_count: number;
  metadata: Record<string, unknown>;
}

export interface ChunkPreviewResponse {
  parser: string;
  chunks: ChunkPreviewItem[];
  total_chunks: number;
  total_chars: number;
}

export interface UploadDocumentResponse {
  job_id: string;
  status: string;
  kb_id: string;
  doc_id: string;
  format: string;
  size_bytes: number;
  parser: string;
}

export type JobStatus =
  | "pending"
  | "running"
  | "done"
  | "error"
  | "cancelled";

export type JobStage =
  | "queued"
  | "parsing"
  | "chunking"
  | "embedding"
  | "saving"
  | "done";

export interface JobInfo {
  job_id: string;
  type: string;
  status: JobStatus;
  stage: JobStage;
  current: number;
  total: number;
  /** 0.0 - 1.0 */
  progress: number;
  message: string;
  error: string | null;
  kb_id: string | null;
  doc_id: string | null;
  filename: string | null;
  created_at: string;
  updated_at: string;
}

export interface CancelJobResponse {
  status: "cancelling" | "not_cancellable" | "not_found";
}

export interface ChunkInfo {
  chunk_id: string;
  chunk_index: number;
  text: string;
  char_count: number;
  metadata: Record<string, unknown>;
}

export interface ChunksListResponse {
  kb_id: string;
  doc_id: string;
  total: number;
  limit: number;
  offset: number;
  chunks: ChunkInfo[];
}

// ===== 检索策略 =====

export interface RetrievalStrategyInfo {
  name: string;
  description: string;
  params: Record<string, unknown>;
  is_default: boolean;
}

export interface RetrievalConfig {
  default_strategy: string;
  dense: {
    over_fetch_factor: number;
  };
  sparse: {
    k1: number;
    b: number;
    use_jieba: boolean;
    stop_words: string[];
  };
  hybrid: {
    dense_weight: number;
    rrf_k: number;
    over_fetch_factor: number;
    k1: number;
    b: number;
    use_jieba: boolean;
  };
  milvus_bm25: {
    analyzer_type: string;
    dense_weight: number;
    rrf_k: number;
    over_fetch_factor: number;
  };
}
