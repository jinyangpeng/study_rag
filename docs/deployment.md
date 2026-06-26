# 部署指南

本文档介绍 study_rag 的多种部署方式：Docker Compose（推荐）、本地开发、生产注意事项。

## 目录

- [快速开始（Docker Compose）](#快速开始docker-compose)
- [部署形态](#部署形态)
- [环境变量](#环境变量)
- [Docker 镜像构建](#docker-镜像构建)
- [Docker Compose 详解](#docker-compose-详解)
- [Milvus 向量库部署](#milvus-向量库部署)
- [本地开发部署](#本地开发部署)
- [生产环境清单](#生产环境清单)
- [运维操作](#运维操作)

---

## 快速开始（Docker Compose）

```bash
# 1. 克隆 & 准备环境变量
git clone <repo-url> study_rag && cd study_rag
cp .env.example .env
#   按需编辑 .env：至少配置 OPENAI_API_KEY 或本地 embedding 端点

# 2. 构建并启动（admin + mcp）
docker compose -f docker/docker-compose.yml up -d --build
#   或: just docker-up

# 3. 验证
curl http://localhost:8765/health        # admin 存活探针
curl http://localhost:8001/health        # mcp 存活探针
#   浏览器打开管理 UI: http://localhost:8765/admin/ui/
#   OpenAPI 文档:      http://localhost:8765/docs
#   MCP 端点:          http://localhost:8001/mcp
```

带 Milvus 向量库的一体化部署：

```bash
docker compose -f docker/docker-compose.yml --profile vector up -d --build
#   或: just docker-up-vector
```

---

## 部署形态

study_rag 由两个独立进程组成，可独立扩缩容：

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  admin REST  │     │  MCP server  │     │   Milvus     │
│  :8765       │     │  :8001       │     │   :19530     │
│  - 管理 UI   │     │  - 10 Tools  │     │  (可选)      │
│  - KB CRUD   │     │  - Resources │     │              │
│  - 检索调试  │     │  - Prompts   │     │              │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                            │
                  ┌─────────▼─────────┐
                  │  configs/*.yaml   │  (共享配置)
                  │  data/            │  (docs_index.json)
                  └───────────────────┘
```

| 进程 | 端口 | 用途 | 扩缩容 |
|---|---|---|---|
| admin | 8765 | 管理面 REST + SPA、健康检查、Prometheus 指标 | 可水平扩展（无状态，data 共享卷） |
| mcp | 8001 | MCP streamable_http 端点（Agent 调用入口） | 可水平扩展 |
| milvus | 19530 | 向量库（可选，也可用 mock 或外部实例） | 独立运维 |

> **为什么 admin 和 mcp 分进程？**
> StreamableHTTP 的 session manager 需要显式管理 ASGI lifespan，mount 到 FastAPI 子应用时 session manager 不会启动。拆成独立进程后，FastMCP 自管生命周期。详见 `src/study_rag/mcp_standalone.py` 顶部注释。

---

## 环境变量

完整变量见 [`.env.example`](../.env.example)。核心变量：

### 服务运行

| 变量 | 默认 | 说明 |
|---|---|---|
| `STUDY_RAG_HOST` | `0.0.0.0` | admin 监听地址 |
| `STUDY_RAG_PORT` | `8765` | admin 监听端口 |
| `STUDY_RAG_LOG_LEVEL` | `INFO` | 日志级别 `DEBUG/INFO/WARNING/ERROR` |
| `STUDY_RAG_WORKERS` | `1` | uvicorn worker 数（MCP 不受影响） |
| `MCP_HOST` | `0.0.0.0` | MCP 监听地址 |
| `MCP_PORT` | `8001` | MCP 监听端口 |

### 鉴权

| 变量 | 默认 | 说明 |
|---|---|---|
| `STUDY_RAG_ADMIN_TOKEN` | *(空)* | admin REST Bearer Token；未设置=不鉴权（仅开发） |
| `STUDY_RAG_MCP_REQUIRE_API_KEY` | `false` | 是否强制 MCP Tool 的 api_key 非空 |

> 生产环境务必设置 `STUDY_RAG_ADMIN_TOKEN`：`openssl rand -hex 32`

### 限流 / 熔断

| 变量 | 默认 | 说明 |
|---|---|---|
| `STUDY_RAG_ADMIN_RATELIMIT_CAPACITY` | `120` | admin 突发上限 |
| `STUDY_RAG_ADMIN_RATELIMIT_PER_SEC` | `2.0` | admin 稳态 QPS |
| `STUDY_RAG_SEARCH_RATELIMIT_CAPACITY` | `30` | 检索突发上限 |
| `STUDY_RAG_SEARCH_RATELIMIT_PER_SEC` | `5.0` | 检索稳态 QPS |
| `STUDY_RAG_OPENAI_BREAKER_THRESHOLD` | `5` | OpenAI 连续失败次数→熔断 |
| `STUDY_RAG_OPENAI_BREAKER_TIMEOUT_S` | `30.0` | 熔断恢复时间(秒) |
| `STUDY_RAG_SEARCH_BREAKER_THRESHOLD` | `10` | 检索连续失败次数→熔断 |
| `STUDY_RAG_SEARCH_BREAKER_TIMEOUT_S` | `20.0` | 检索熔断恢复时间(秒) |

### 向量库 & API Keys

| 变量 | 说明 |
|---|---|
| `MILVUS_URI` | Milvus 地址（容器内用服务名 `http://milvus:19530`） |
| `MILVUS_TOKEN` | Milvus/Zilliz 鉴权 token |
| `MILVUS_DB` | Milvus 数据库名 |
| `OPENAI_API_KEY` | OpenAI embedding key |
| `COHERE_API_KEY` | Cohere reranker key |
| `DASHSCOPE_API_KEY` | 通义千问 embedding key |
| `ZHIPUAI_API_KEY` | 智谱 embedding key |
| `LOCAL_EMBED_API_KEY` | 本地自建 embedding 服务 key |
| `LOCAL_RERANK_API_KEY` | 本地自建 reranker 服务 key |

> YAML 配置内的 `${VAR}` 占位符由 app 运行时解析（见 `capabilities/embedding/base.py`），需对应变量在容器环境存在。

---

## Docker 镜像构建

Dockerfile 采用多阶段构建：

1. **frontend-builder**（Node 20）：构建 React SPA → `src/study_rag/web/dist`
2. **base**（Python 3.11-slim）：安装 Python 依赖 + 拷贝前端产物 + 非 root 用户运行

### 构建参数 `EXTRAS`

控制 `pip install -e ".[${EXTRAS}]"` 的可选依赖组，默认 `llamaindex,vector-milvus`：

```bash
# 默认（llamaindex + milvus）
docker build -f docker/Dockerfile -t study-rag:dev .

# 含 OpenAI embedding + BGE reranker
docker build -f docker/Dockerfile \
  --build-arg EXTRAS=llamaindex,vector-milvus,embedding-openai,reranker-bge \
  -t study-rag:dev .

# 全量（含所有推荐组合）
docker build -f docker/Dockerfile --build-arg EXTRAS=all -t study-rag:dev .

# 或用 justfile
just docker-build llamaindex,vector-milvus,embedding-openai,reranker-bge
```

可选依赖组（见 `pyproject.toml`）：

| 依赖组 | 内容 |
|---|---|
| `llamaindex` | llama-index-core（切块） |
| `vector-milvus` | pymilvus |
| `vector-qdrant` | qdrant-client |
| `embedding-openai` | openai SDK |
| `embedding-bge` | FlagEmbedding + torch（本地推理） |
| `embedding-bge-st` | sentence-transformers |
| `embedding-fastembed` | fastembed |
| `reranker-bge` | FlagEmbedding |
| `reranker-cohere` | cohere SDK |
| `reader-pdf` | pypdf |
| `reader-docx` | docx2txt |
| `reader-md` | markdown |
| `all` | 推荐组合一键安装 |
| `dev` | pytest + ruff + mypy（仅开发） |

> **注意**：`embedding-bge` / `embedding-bge-st` 含 PyTorch，镜像会显著变大（~2GB+）。生产建议用 OpenAI 兼容的本地推理服务（TEI / vLLM / Ollama），镜像只装 `embedding-openai`。

---

## Docker Compose 详解

### 服务清单

| 服务 | profile | 端口 | 说明 |
|---|---|---|---|
| `admin` | *(默认)* | 8765 | FastAPI 管理面 + SPA |
| `mcp` | *(默认)* | 8001 | MCP streamable_http |
| `milvus` | `vector` | 19530 | Milvus Standalone |
| `milvus-etcd` | `vector` | - | Milvus 元数据存储 |
| `milvus-minio` | `vector` | - | Milvus 对象存储 |
| `dev-tools` | `dev` | - | 调试容器（sleep infinity） |

### 配置挂载

```yaml
volumes:
  - ../configs:/app/configs      # admin: 可写（支持运行时改配置）
  - ../configs:/app/configs:ro   # mcp:   只读
  - rag-data:/app/data           # 共享 docs_index.json
```

- **admin 挂载可写**：admin REST 支持通过 UI/API 动态新增 KB、embedder、reranker，这些会写回 `configs/*.yaml`
- **mcp 挂载只读**：MCP 进程不写配置，只读
- **生产锁定配置**：把 admin 的 `:ro` 也改为只读，并禁用 admin 的配置写入接口（通过不设置写权限或网络层拦截）

### 环境变量注入

compose 不使用 `env_file` 指令（避免 `.env` 不存在时报错），而是：

1. docker compose **自动从命令执行目录加载 `.env`**（项目根）用于 `${VAR}` 插值
2. 每个服务的 `environment:` 显式列出需要传入容器的变量，用 `${VAR:-default}` 兜底
3. 未创建 `.env` 时所有变量取默认值 / 空串，服务仍可启动（用 mock 后端）

### 健康检查

| 服务 | 探针端点 | 含义 |
|---|---|---|
| admin | `GET /health` | 进程存活 |
| mcp | `GET /health` | 进程存活 + KB 已懒加载 |
| milvus | `GET :9091/healthz` | Milvus 就绪 |
| milvus-etcd | `etcdctl endpoint health` | etcd 健康 |
| milvus-minio | `GET :9000/minio/health/live` | MinIO 健康 |

`mcp` 服务 `depends_on: admin (service_healthy)`，确保 admin 先就绪。Milvus 各组件 `depends_on` 带健康条件，按序启动。

### 网络

所有服务在 `rag-net` bridge 网络内，通过服务名互访（如 `http://milvus:19530`）。

---

## Milvus 向量库部署

### 方式一：compose profile（推荐开发/测试）

```bash
docker compose -f docker/docker-compose.yml --profile vector up -d
```

启动 Milvus Standalone（含 etcd + minio），`MILVUS_URI` 自动指向 `http://milvus:19530`。

### 方式二：外部 Milvus

已有 Milvus 集群时，在 `.env` 设置：

```env
MILVUS_URI=https://your-milvus.example.com:19530
MILVUS_TOKEN=your-token
```

不启用 `--profile vector`，compose 只起 admin + mcp。

### 方式三：Milvus Lite（单机嵌入式）

适合小规模 / 单机测试，无需独立服务。在 `configs/vector_store.yaml` 配置：

```yaml
vector_store:
  provider: milvus
  uri: ./milvus.db    # 本地文件
```

### 方式四：mock（零依赖开发）

```yaml
vector_store:
  provider: mock       # 内存版，重启丢数据
```

---

## 本地开发部署

不用 Docker，直接 Python 运行（适合开发调试）：

```powershell
# 1. 创建 venv + 装依赖
just setup
#   或手动:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,llamaindex,vector-milvus]"

# 2. 构建前端（首次）
just ui-install
just ui-build

# 3. 启动服务
just admin    # admin REST (port 8765, --reload)
just mcp      # MCP standalone (port 8001, --reload)
just ui-dev   # Vite dev server (port 5173, HMR + proxy)

# 4. 调试 MCP
just inspector   # MCP Inspector UI
```

详见 [README.md](../README.md) 的「本地开发」章节。

---

## 生产环境清单

部署前逐项确认：

### 安全

- [ ] `STUDY_RAG_ADMIN_TOKEN` 已设置强随机串
- [ ] `STUDY_RAG_MCP_REQUIRE_API_KEY=true`（若接入真实鉴权）
- [ ] 所有 API Key 通过 `.env` / secret manager 注入，不进镜像
- [ ] HTTPS 终止（反向代理 / ingress）
- [ ] admin 的 configs 挂载按需设为 `:ro`

### 可靠性

- [ ] `restart: unless-stopped` 已配置
- [ ] 健康检查端点被负载均衡 / K8s 探针使用
- [ ] Milvus 数据卷已做备份策略
- [ ] `rag-data` 卷（docs_index.json）已持久化
- [ ] 日志收集（structlog JSON 输出，对接 ELK / Loki）

### 性能

- [ ] `STUDY_RAG_WORKERS` 按 CPU 调整（admin 多 worker）
- [ ] 限流参数按负载调整（`STUDY_RAG_*_RATELIMIT_*`）
- [ ] embedding / reranker 模型预热（首请求会触发加载）
- [ ] Milvus index 类型确认（AUTOINDEX / HNSW）

### 监控

- [ ] `GET /metrics`（admin + mcp）接入 Prometheus
- [ ] `GET /health/detailed`（admin）监控 KB / 限流 / 熔断状态
- [ ] 熔断器阈值合理（`STUDY_RAG_*_BREAKER_*`）

---

## 运维操作

### 日志

```bash
# 实时日志
just docker-logs
# 或
docker compose -f docker/docker-compose.yml logs -f

# 单个服务
docker compose -f docker/docker-compose.yml logs -f admin
docker compose -f docker/docker-compose.yml logs -f mcp
```

### 进入容器

```bash
docker compose -f docker/docker-compose.yml exec admin sh
docker compose -f docker/docker-compose.yml exec mcp sh
```

### 在容器内跑测试 / lint

```bash
just docker-test
# 或
docker compose -f docker/docker-compose.yml --profile dev run --rm dev-tools just verify
```

### 数据卷管理

```bash
# 停止服务但保留数据
just docker-down

# 停止并删除所有数据卷（Milvus + rag-data，谨慎！）
just docker-purge
```

### 重新构建

```bash
# 代码变更后重建镜像
docker compose -f docker/docker-compose.yml build

# 带自定义依赖组重建
just docker-build llamaindex,vector-milvus,embedding-openai,reranker-bge
docker compose -f docker/docker-compose.yml up -d
```

### 配置热更新

- 修改 `configs/*.yaml` 后需重启服务（admin 和 mcp 各自加载配置到内存）
- admin 通过 REST API 修改配置（新增 KB / embedder）会实时写回 YAML，但 mcp 进程不会自动感知——需重启 mcp
