# 配置参考

study_rag 采用「YAML 配置文件 + 环境变量」双层配置。YAML 描述业务结构（KB 定义、模型注册表），环境变量注入 secrets 和运行时参数。

## 目录

- [配置加载机制](#配置加载机制)
- [配置文件总览](#配置文件总览)
- [knowledge_bases.yaml](#knowledge_basesyaml)
- [embeddings.yaml](#embeddingsyaml)
- [vector_store.yaml](#vector_storeyaml)
- [reranker.yaml](#rerankeryaml)
- [retrieval.yaml](#retrievalyaml)
- [llamaindex.yaml](#llamaindexyaml)
- [llm.yaml](#llmyaml)
- [环境变量参考](#环境变量参考)

---

## 配置加载机制

### 路径解析

配置路径由 `src/study_rag/settings.py` 的 `AppPaths` 定义，固定为 `项目根/configs/*.yaml`：

```
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # src/study_rag/ 的上两级
CONFIGS_DIR  = PROJECT_ROOT / "configs"
```

> 容器内 `PROJECT_ROOT=/app`，配置在 `/app/configs/`（Dockerfile `COPY configs ./configs`，compose 挂载覆盖）。

### `${VAR}` 占位符

YAML 内的 `${VAR_NAME}` 会在运行时被 `capabilities/embedding/base.py` 解析为环境变量值：

```yaml
extra:
  api_key: ${OPENAI_API_KEY}        # → os.environ["OPENAI_API_KEY"]
  base_url: http://10.0.0.5:8080/v1
```

- 变量不存在时保留原文 `${VAR_NAME}`（不报错，通常后续调用才失败）
- 适用于 embedding / reranker 配置中的 `api_key`、`base_url` 等敏感/环境相关字段

### pydantic-settings

服务运行时参数（端口、限流、熔断、鉴权）由 `ServerSettings` 通过 pydantic-settings 加载：

- 环境变量前缀：`STUDY_RAG_`（如 `STUDY_RAG_PORT`）
- 支持 `.env` 文件（`env_file=".env"`）
- 优先级：环境变量 > `.env` > 代码默认值

---

## 配置文件总览

| 文件 | 用途 | 可运行时修改 |
|---|---|---|
| `knowledge_bases.yaml` | 知识库定义（每 KB 独立 embedding/reranker） | 是（admin REST 写回） |
| `embeddings.yaml` | Embedding 注册表（多后端） | 是（admin REST 写回） |
| `vector_store.yaml` | 向量库 provider 切换 | 否（重启生效） |
| `reranker.yaml` | Reranker 注册表（多后端） | 是（admin REST 写回） |
| `retrieval.yaml` | 检索策略参数（dense/sparse/hybrid） | 否（重启生效） |
| `llamaindex.yaml` | NodeParser 切块策略注册表 | 是（admin REST 写回） |
| `llm.yaml` | LLM 配置（当前未启用） | 否 |

> 「可运行时修改」= admin REST 接口（`POST /admin/kbs` 等）会直接写回对应 YAML 文件，热生效于 admin 进程；mcp 进程需重启才能感知。

---

## knowledge_bases.yaml

定义所有知识库。每个 KB 可独立指定 embedding / reranker / 检索策略。

```yaml
knowledge_bases:
- kb_id: rd_frontend          # 全局唯一，命名规范 {dept}_{name}
  name: 前端技术文档
  description: 研发部前端技术栈文档...  # Agent 选 KB 的依据，建议详细
  department: rd              # 部门，用于权限分组
  collection: kb_rd_frontend  # Milvus collection 名（全局唯一）
  embedding: ollama_bge_large # 引用 embeddings.yaml 的 key
  reranker: none              # 引用 reranker.yaml 的 key；none=不重排
  retrieval_strategy: dense   # 可选：dense/sparse/hybrid/sparse_milvus/hybrid_milvus（省略=全局默认）
  enabled: true
  extra: {}                   # 预留扩展字段
```

**关键字段**：

| 字段 | 必填 | 说明 |
|---|---|---|
| `kb_id` | ✓ | 知识库 ID，命名规范 `{dept}_{name}` |
| `name` | ✓ | 显示名 |
| `description` | ✓ | 描述（Agent 据此判断查哪个 KB） |
| `department` | ✓ | 部门 |
| `collection` | ✓ | 向量库 collection 名 |
| `embedding` | ✓ | embeddings.yaml 中的配置名 |
| `reranker` | | reranker.yaml 中的配置名，默认 `none` |
| `retrieval_strategy` | | 检索策略，省略则用 `retrieval.yaml` 的 `default_strategy` |
| `enabled` | | 是否启用，默认 true |
| `extra` | | 预留扩展 |

---

## embeddings.yaml

Embedding 模型注册表。通过命名实体复用，类似 embedder 工厂。

```yaml
embeddings:
  mock:                       # 配置名（KB 引用此 key）
    provider: mock            # 实现类型
    model_name: mock-embedder
    dimension: 128
    batch_size: 32
    description: 基于 hash 的占位实现
  openai_small:
    provider: openai
    model_name: text-embedding-3-small
    dimension: 1536
    batch_size: 100
    extra:
      api_key: ${OPENAI_API_KEY}     # ${VAR} 占位符
      timeout: 30.0
      max_retries: 3
```

**支持的 provider**：

| provider | 依赖组 | 说明 |
|---|---|---|
| `mock` | *(核心)* | hash 占位，零依赖，开发用 |
| `openai` | `embedding-openai` | OpenAI / 兼容协议（通义/智谱/本地/Ollama） |
| `bge` | `embedding-bge` | FlagEmbedding 本地推理（BGE-M3） |
| `bge_zh` | `embedding-bge` | BGE 中文模型（自动加 query 前缀） |
| `azure_openai` | `embedding-openai` | Azure OpenAI |
| `fastembed` | `embedding-fastembed` | fastembed 轻量本地 |

**OpenAI 兼容协议**：通义千问、智谱、Ollama、本地 TEI 都用 `provider: openai` + `extra.base_url` 指向各自的 `/v1` 端点。

---

## vector_store.yaml

向量库 provider 配置，全局唯一（所有 KB 共享同一向量库实例，通过不同 collection 隔离）。

```yaml
vector_store:
  provider: milvus            # mock / milvus / qdrant / weaviate
  uri: ${MILVUS_URI:-./milvus.db}   # Milvus Standalone / Lite / Zilliz
  extra:
    token: ${MILVUS_TOKEN}    # Zilliz / 鉴权时必填
    db_name: ${MILVUS_DB:-default}
    batch_size: 1000
    index:
      type: AUTOINDEX         # AUTOINDEX / HNSW / IVF_FLAT
      metric_type: COSINE     # COSINE / L2 / IP
```

**provider 对照**：

| provider | uri 示例 | 说明 |
|---|---|---|
| `mock` | *(空)* | 内存版，重启丢数据 |
| `milvus` | `./milvus.db` | Milvus Lite（本地文件，单机） |
| `milvus` | `http://milvus:19530` | Milvus Standalone（docker compose） |
| `milvus` | `https://xxx.zillizcloud.com` | Zilliz Cloud（需 token） |
| `qdrant` | `http://localhost:6333` | Qdrant 轻量替代 |

> **切换向量库**：改 `provider` + `uri`，重启即可。collection 名保持不变，但不同向量库的数据不互通（需重新导入文档）。

---

## reranker.yaml

Reranker 模型注册表。

```yaml
rerankers:
  none:                       # 不重排
    provider: none
    top_k: 5
  bge_m3:                     # 本地 BGE reranker
    provider: bge
    model_name: BAAI/bge-reranker-v2-m3
    top_k: 5
    extra:
      use_fp16: true
      batch_size: 32
  cohere_v3:                  # Cohere API
    provider: cohere
    model_name: rerank-v3.5
    extra:
      api_key: ${COHERE_API_KEY}
  tei_bge_m3:                 # TEI 部署（HTTP /rerank 协议）
    provider: http
    protocol: tei
    model_name: BAAI/bge-reranker-v2-m3
    extra:
      base_url: http://127.0.0.1:8081
      truncate_input_tokens: 512
```

**支持的 provider**：

| provider | protocol | 依赖组 | 说明 |
|---|---|---|---|
| `none` | - | *(核心)* | 不重排，截断到 top_k |
| `mock` | - | *(核心)* | 同 none（兼容别名） |
| `bge` | - | `reranker-bge` | FlagEmbedding 本地推理 |
| `cohere` | - | `reranker-cohere` | Cohere API |
| `http` | `tei` | *(核心)* | TEI 部署（/rerank 协议） |
| `http` | `cohere_compat` | *(核心)* | Cohere 兼容的私有部署 |

**Top K 语义**（重要）：

- KB 配置的 `top_k` = reranker 最终返回数
- 启用 rerank 时，向量召回数 = `top_k × over_fetch_factor`（见 `retrieval.yaml`）
- admin UI / MCP `search_kb` 的 `top_k` 参数 = **召回数**（候选数），不是最终返回数

---

## retrieval.yaml

检索策略全局参数。每个 KB 可在 `knowledge_bases.yaml` 用 `retrieval_strategy` 覆盖。

```yaml
retrieval:
  default_strategy: dense     # dense / sparse / hybrid / sparse_milvus / hybrid_milvus

  dense:
    over_fetch_factor: 4      # 启用 rerank 时 candidate_k = top_k × 此值

  sparse:                     # 纯 Python BM25
    k1: 1.5
    b: 0.75
    use_jieba: true           # 中文分词（未装 jieba 自动降级为正则）
    stop_words: []

  hybrid:                     # Dense + Sparse RRF 融合
    dense_weight: 0.6         # 语义为主 0.7~0.8；关键词为主 0.3~0.4
    rrf_k: 60                 # Reciprocal Rank Fusion 常数
    over_fetch_factor: 4
    k1: 1.5
    b: 0.75
    use_jieba: true
    stop_words: []

  milvus_bm25:                # Milvus 2.5+ 原生 BM25（需 pymilvus>=2.5）
    analyzer_type: chinese    # chinese / english / standard
    dense_weight: 0.6         # 仅 hybrid_milvus
    rrf_k: 60
    over_fetch_factor: 4
```

**策略对照**：

| 策略 | 召回方式 | 依赖 | 适用 |
|---|---|---|---|
| `dense` | 向量语义 | embedding + vector store | 通用默认 |
| `sparse` | BM25 关键词 | 纯 Python 倒排索引 | 精确关键词匹配 |
| `hybrid` | Dense + Sparse RRF | 两者都有 | 综合（小规模） |
| `sparse_milvus` | Milvus 2.5+ 原生 BM25 | Milvus ≥2.5 | 大规模关键词 |
| `hybrid_milvus` | Milvus Dense + BM25 融合 | Milvus ≥2.5 | 大规模综合 |

> **参数三级合并**：`retrieval.yaml` 全局 < KB 配置 < 请求级 API 参数覆盖。

---

## llamaindex.yaml

NodeParser 切块策略注册表。

```yaml
parsers:
  whole:                      # 整篇一个 chunk
    strategy: whole
  sentence_512:               # 句子切块
    strategy: sentence
    chunk_size: 512
    chunk_overlap: 50
    paragraph_separator: "\n\n"
  token_512:                  # Token 切块
    strategy: token
    chunk_size: 512
    chunk_overlap: 50
  semantic:                   # 语义切块
    strategy: semantic
    chunk_size: 512
    buffer_size: 1
    breakpoint_percentile_threshold: 95

settings:
  enabled: true
  over_fetch: 4
  llm:
    provider: none            # 语义切块默认不依赖 LLM
```

**支持的 strategy**：

| strategy | 说明 | 适用 |
|---|---|---|
| `whole` | 整篇一个 chunk | 短文档 |
| `sentence` | 句子边界切块 | 通用 |
| `token` | Token 数切块 | 需精确控制长度 |
| `semantic` | 语义相似度切分 | 长文档、内容多样 |

---

## llm.yaml

LLM 配置（用于响应合成等场景，当前未启用，项目设计原则不引入 LLM）。

```yaml
llm:
  provider: mock
  model_name: mock-llm
```

---

## 环境变量参考

完整列表见 [`.env.example`](../.env.example)，分组说明：

### pydantic-settings（`STUDY_RAG_` 前缀）

| 变量 | 默认 | 说明 |
|---|---|---|
| `STUDY_RAG_HOST` | `0.0.0.0` | admin 监听地址 |
| `STUDY_RAG_PORT` | `8000` | admin 监听端口（容器内 entrypoint 用 8765） |
| `STUDY_RAG_LOG_LEVEL` | `INFO` | 日志级别 |
| `STUDY_RAG_WORKERS` | `1` | uvicorn workers |
| `STUDY_RAG_ADMIN_TOKEN` | *(空)* | admin Bearer Token |
| `STUDY_RAG_MCP_REQUIRE_API_KEY` | `false` | 强制 MCP api_key |
| `STUDY_RAG_ADMIN_RATELIMIT_CAPACITY` | `120` | admin 限流突发 |
| `STUDY_RAG_ADMIN_RATELIMIT_PER_SEC` | `2.0` | admin 限流 QPS |
| `STUDY_RAG_SEARCH_RATELIMIT_CAPACITY` | `30` | 检索限流突发 |
| `STUDY_RAG_SEARCH_RATELIMIT_PER_SEC` | `5.0` | 检索限流 QPS |
| `STUDY_RAG_OPENAI_BREAKER_THRESHOLD` | `5` | OpenAI 熔断阈值 |
| `STUDY_RAG_OPENAI_BREAKER_TIMEOUT_S` | `30.0` | OpenAI 熔断恢复 |
| `STUDY_RAG_SEARCH_BREAKER_THRESHOLD` | `10` | 检索熔断阈值 |
| `STUDY_RAG_SEARCH_BREAKER_TIMEOUT_S` | `20.0` | 检索熔断恢复 |

### 容器入口（entrypoint.sh）

| 变量 | 默认 | 说明 |
|---|---|---|
| `CONTAINER_ROLE` | `admin` | `admin` / `mcp` 选择启动进程 |
| `MCP_HOST` | `0.0.0.0` | MCP 监听地址 |
| `MCP_PORT` | `8001` | MCP 监听端口 |

### YAML 占位符（`embeddings.yaml` / `reranker.yaml` / `vector_store.yaml` 引用）

| 变量 | 引用处 |
|---|---|
| `OPENAI_API_KEY` | `openai_*` embedding 配置 |
| `AZURE_OPENAI_API_KEY` | `azure_openai` embedding |
| `DASHSCOPE_API_KEY` | `qwen_embedding` |
| `ZHIPUAI_API_KEY` | `zhipu_embedding` |
| `COHERE_API_KEY` | `cohere_*` reranker |
| `LOCAL_EMBED_API_KEY` | `local_*` embedding |
| `LOCAL_RERANK_API_KEY` | `local_*` reranker |
| `MILVUS_URI` | `vector_store.yaml` uri |
| `MILVUS_TOKEN` | `vector_store.yaml` extra.token |
| `MILVUS_DB` | `vector_store.yaml` extra.db_name |
| `RERANK_HTTP_BASE_URL` | HTTP reranker fallback（`impls_http.py`） |

### 代码直接读取

部分变量在代码中用 `os.environ.get` 直接读取（非 pydantic-settings）：

| 变量 | 读取位置 | 说明 |
|---|---|---|
| `STUDY_RAG_ADMIN_TOKEN` | `api/admin.py:_admin_token()` | admin 鉴权 |
| `MCP_PORT` / `MCP_HOST` | `mcp_standalone.py:main()` | CLI 入口端口 |
| `OPENAI_API_KEY` | `embedding/impls_openai.py` | OpenAI 兜底 |
| `COHERE_API_KEY` | `reranker/impls_cohere.py` | Cohere 兜底 |
| `RERANK_HTTP_BASE_URL` | `reranker/impls_http.py` | HTTP reranker 兜底 |
