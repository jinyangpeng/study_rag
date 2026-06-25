"""知识库数据模型。

每个 KnowledgeBase 对应一个向量库 collection。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class KnowledgeBaseConfig(BaseModel):
    """知识库定义（YAML schema）。

    kb_id 命名规范: {dept}_{name}，如 rd_frontend、hr_policies。
    这个命名会作为 Agent 选 Tool 的依据。
    """

    kb_id: str = Field(..., description="知识库唯一标识")
    name: str = Field(..., description="显示名")
    description: str = Field(..., description="知识库描述，供 Agent 选 KB")
    department: str = Field(..., description="所属部门")
    collection: str = Field(..., description="向量库 collection 名称")
    embedding: str = Field(..., description="使用的 embedding 配置名（对应 embeddings.yaml 中的 key）")
    reranker: str | None = Field(default=None, description="使用的 reranker 配置名（可选）")
    enabled: bool = Field(default=True, description="是否启用")
    extra: dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseCreate(BaseModel):
    """创建知识库请求体（来自管理前端）。

    字段语义同 KnowledgeBaseConfig，但 collection 可省略（默认 = "kb_" + kb_id）。
    """

    kb_id: str = Field(
        ...,
        min_length=2,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="KB 唯一标识（小写字母/数字/下划线，以字母开头）",
        examples=["rd_frontend"],
    )
    name: str = Field(..., min_length=1, max_length=128, description="显示名")
    description: str = Field(..., min_length=1, description="描述（Agent 选 KB 依据）")
    department: str = Field(..., min_length=1, max_length=64, description="所属部门")
    collection: str | None = Field(
        default=None,
        description="向量库 collection 名称（不填则用 'kb_' + kb_id）",
    )
    embedding: str = Field(..., description="embedding 配置名（见 GET /admin/embedders）")
    reranker: str | None = Field(default=None, description="reranker 配置名（见 GET /admin/rerankers）")
    enabled: bool = Field(default=True, description="是否启用")


class KnowledgeBaseUpdate(BaseModel):
    """更新知识库请求体（部分字段更新）。

    所有字段可选；只传需要改的。
    collection / embedding 改了需要重建 collection（当前未实现自动迁移，会拒绝）。
    """

    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = Field(default=None, min_length=1)
    department: str | None = Field(default=None, min_length=1, max_length=64)
    reranker: str | None = Field(default=None)
    enabled: bool | None = Field(default=None)


class EmbedderInfo(BaseModel):
    """前端下拉用：embedder 元信息。"""

    name: str = Field(..., description="配置名（=KnowledgeBaseConfig.embedding）")
    provider: str = Field(..., description="实现 provider: bge / openai / mock / ...")
    model_name: str
    dimension: int
    batch_size: int
    loaded: bool = Field(..., description="当前是否已加载（依赖是否装齐）")
    description: str = Field(default="", description="管理员备注")


class RerankerInfo(BaseModel):
    """前端下拉用：reranker 元信息。"""

    name: str = Field(..., description="配置名（=KnowledgeBaseConfig.reranker）")
    provider: str = Field(..., description="实现 provider: bge / cohere / http / ...")
    model_name: str
    top_k: int
    loaded: bool
    description: str = Field(default="")


class KnowledgeBase(BaseModel):
    """运行时知识库实例。"""

    config: KnowledgeBaseConfig
    document_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class KnowledgeBaseSummary(BaseModel):
    """知识库摘要（用于 list 接口，避免泄露内部信息）。"""

    kb_id: str = Field(..., description="知识库唯一标识", examples=["rd_frontend"])
    name: str = Field(..., description="显示名", examples=["前端技术文档"])
    description: str = Field(
        ...,
        description="知识库描述。Agent 选 KB 的关键依据，要写清楚'什么内容'、'适合什么问题'。",
    )
    department: str = Field(..., description="所属部门", examples=["rd"])
    enabled: bool = Field(..., description="是否启用")
    document_count: int = Field(..., description="已添加的文档数（运行时统计）", examples=[42])
    chunk_count: int = Field(default=0, description="切块后的 chunk 总数（运行时统计）")
    embedder: str | None = Field(default=None, description="当前 embedder 配置名")
    reranker: str | None = Field(default=None, description="当前 reranker 配置名")
    vector_store: str | None = Field(default=None, description="向量库 provider")
    collection: str | None = Field(default=None, description="向量库 collection 名称")


class DocumentMeta(BaseModel):
    """文档元信息。"""

    doc_id: str = Field(..., description="文档 ID（KB 内唯一）", examples=["react_perf_001"])
    kb_id: str = Field(..., description="所属知识库 ID")
    title: str = Field(..., description="标题", examples=["React 性能优化指南"])
    source: str | None = Field(default=None, description="来源（wiki/git/...）", examples=["wiki"])
    content: str | None = Field(default=None, description="完整正文（与向量库一致）")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="扩展 metadata；可在 search_kb 的 filter_expr 里按这里字段过滤",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="创建时间（ISO 8601）",
    )
    # 运行时统计：写入时设置；list_documents 会用 vector store 实时值覆盖
    chunk_count: int = Field(default=0, description="该文档在向量库的 chunk 数（list 时由后端实时统计覆盖）")
    char_count: int = Field(default=0, description="该文档内容总字符数（list 时由后端从 content 长度填）")
    # 分块方式（pipeline / chunked add 时记录；add_document 默认为 'whole'）
    parser: str | None = Field(
        default=None,
        description="分块方式（sentence_512 / whole / ...）",
        examples=["sentence_512"],
    )


class ChunkInfo(BaseModel):
    """文档切块信息（从向量库查出来的 chunk 元信息）。

    字段：
      chunk_id:        Milvus 主键（string）
      chunk_index:     块索引（0-based）
      text:            块完整文本
      char_count:      字符数（UI 友好）
      metadata:        原始 metadata（title/source/parser/...）
    """

    chunk_id: str = Field(..., description="chunk 唯一标识（Milvus 主键）")
    chunk_index: int = Field(..., description="块索引（0-based，按切块顺序）")
    text: str = Field(..., description="chunk 完整文本")
    char_count: int = Field(..., description="字符数（=len(text)）")
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentCreate(BaseModel):
    """创建文档请求。"""

    kb_id: str = Field(..., description="目标知识库 ID（必须与路径一致）")
    doc_id: str = Field(..., description="文档 ID（KB 内唯一）")
    title: str = Field(..., description="标题", examples=["React 性能优化指南"])
    content: str = Field(..., description="正文（建议长度 100-2000 字）")
    source: str | None = Field(default=None, description="来源；会存入 metadata['source']")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="扩展 metadata；后续 filter 可按这里字段过滤",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "kb_id": "rd_frontend",
                "doc_id": "react_perf_001",
                "title": "React 性能优化指南",
                "content": "React 应用常见的性能优化点包括：1) 用 React.memo 缓存组件...",
                "source": "wiki",
                "metadata": {"author": "alice", "year": 2025},
            }
        }
    }
