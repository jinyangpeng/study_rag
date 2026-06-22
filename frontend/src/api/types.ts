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
  chunk_count?: number;
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
  hits: SearchHit[];
  duration_ms: number;
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
  kb_id: string;
  doc_id: string;
  title: string;
  chunks: number;
  format: string;
  size_bytes: number;
  parser: string;
}
