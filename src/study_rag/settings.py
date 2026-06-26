"""全局配置层。

使用 Pydantic Settings 加载环境变量 + YAML 配置文件。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"


class ServerSettings(BaseSettings):
    """服务运行时配置。"""

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    workers: int = 1

    # ---- 限流（admin REST + MCP 检索） ----
    # 桶容量（突发上限）+ 稳态 QPS
    admin_ratelimit_capacity: int = 120
    admin_ratelimit_per_sec: float = 2.0
    search_ratelimit_capacity: int = 30
    search_ratelimit_per_sec: float = 5.0

    # ---- 熔断（保护 OpenAI / 检索链路） ----
    # 连续失败次数 → OPEN；OPEN 持续时间 → HALF_OPEN
    openai_breaker_threshold: int = 5
    openai_breaker_timeout_s: float = 30.0
    search_breaker_threshold: int = 10
    search_breaker_timeout_s: float = 20.0

    # ---- MCP 鉴权（api_key 强制校验） ----
    # 默认 false：PermissionResolver 占位实现允许任意 api_key（含空串），本地开发零摩擦
    # 设为 true 时：非空 api_key 必须能在 resolve() 中命中（占位实现下也即非空即可）
    # 真实接入 JWT/OAuth 后此开关生效，未命中则 PermissionDenied
    mcp_require_api_key: bool = False

    model_config = SettingsConfigDict(
        env_prefix="STUDY_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class AppPaths:
    """项目内常用路径。"""

    ROOT = PROJECT_ROOT
    CONFIGS = CONFIGS_DIR
    KB_CONFIG = CONFIGS_DIR / "knowledge_bases.yaml"
    EMBEDDING_CONFIG = CONFIGS_DIR / "embeddings.yaml"
    VECTOR_STORE_CONFIG = CONFIGS_DIR / "vector_store.yaml"
    RERANKER_CONFIG = CONFIGS_DIR / "reranker.yaml"
    LLAMAINDEX_CONFIG = CONFIGS_DIR / "llamaindex.yaml"
    LLM_CONFIG = CONFIGS_DIR / "llm.yaml"
    RETRIEVAL_CONFIG = CONFIGS_DIR / "retrieval.yaml"
    # 运行时持久化数据（重启后从 Milvus 重建 DocumentMeta）
    DATA_DIR = PROJECT_ROOT / "data"
    DOCS_INDEX = DATA_DIR / "docs_index.json"


@lru_cache(maxsize=1)
def get_server_settings() -> ServerSettings:
    """获取全局服务配置（单例）。"""
    return ServerSettings()
