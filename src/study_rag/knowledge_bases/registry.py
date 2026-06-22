"""知识库注册表：从 YAML 配置加载知识库定义。

提供：
  - 全局单例（lru_cache）
  - kb_id -> KnowledgeBaseConfig 索引
  - list / get / add / update / delete 接口
  - 原子写回 YAML（os.replace）
"""

from __future__ import annotations

import builtins
import os
import tempfile
import threading
from functools import lru_cache

import yaml

from ..settings import AppPaths
from .models import KnowledgeBaseConfig, KnowledgeBaseCreate, KnowledgeBaseUpdate


class KnowledgeBaseRegistryError(Exception):
    """知识库注册表错误。"""


class KnowledgeBaseRegistry:
    """知识库注册表（内存态）。"""

    def __init__(self, configs: list[KnowledgeBaseConfig]):
        self._configs: dict[str, KnowledgeBaseConfig] = {}
        self._lock = threading.RLock()
        for cfg in configs:
            if cfg.kb_id in self._configs:
                raise KnowledgeBaseRegistryError(
                    f"Duplicate kb_id: {cfg.kb_id}"
                )
            self._configs[cfg.kb_id] = cfg

    def get(self, kb_id: str) -> KnowledgeBaseConfig | None:
        with self._lock:
            return self._configs.get(kb_id)

    def get_required(self, kb_id: str) -> KnowledgeBaseConfig:
        cfg = self.get(kb_id)
        if cfg is None:
            raise KnowledgeBaseRegistryError(f"Unknown kb_id: {kb_id}")
        return cfg

    def all_cfgs(self, enabled_only: bool = False) -> list[KnowledgeBaseConfig]:
        """返回所有 cfg（拷贝）；enabled_only=True 时过滤未启用的。

        命名避开了 ``list``，避免跟 builtin type 冲突。
        """
        with self._lock:
            cfgs: list[KnowledgeBaseConfig] = list(self._configs.values())
        if enabled_only:
            cfgs = [c for c in cfgs if c.enabled]
        return cfgs

    def list_ids(self) -> builtins.list[str]:
        with self._lock:
            return list(self._configs.keys())

    # ===== 变更接口（只改内存；调用方负责持久化） =====

    def add(self, cfg: KnowledgeBaseConfig) -> None:
        """新增 KB；已存在抛错。"""
        with self._lock:
            if cfg.kb_id in self._configs:
                raise KnowledgeBaseRegistryError(
                    f"kb_id already exists: {cfg.kb_id}"
                )
            self._configs[cfg.kb_id] = cfg

    def update(self, kb_id: str, patch: KnowledgeBaseUpdate) -> KnowledgeBaseConfig:
        """部分更新 KB；返回更新后的 cfg。"""
        with self._lock:
            current = self._configs.get(kb_id)
            if current is None:
                raise KnowledgeBaseRegistryError(f"Unknown kb_id: {kb_id}")
            data = current.model_dump()
            for k, v in patch.model_dump(exclude_unset=True).items():
                data[k] = v
            new_cfg = KnowledgeBaseConfig(**data)
            self._configs[kb_id] = new_cfg
            return new_cfg

    def delete(self, kb_id: str) -> KnowledgeBaseConfig:
        """删除 KB；返回被删除的 cfg。"""
        with self._lock:
            current = self._configs.get(kb_id)
            if current is None:
                raise KnowledgeBaseRegistryError(f"Unknown kb_id: {kb_id}")
            del self._configs[kb_id]
            return current


def _load_from_yaml(path) -> list[KnowledgeBaseConfig]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw_list = data.get("knowledge_bases", [])
    return [KnowledgeBaseConfig(**item) for item in raw_list]


def _dump_to_yaml(path, configs: list[KnowledgeBaseConfig]) -> None:
    """原子写回 YAML：先写 .tmp，再 os.replace 覆盖。"""
    payload = {
        "knowledge_bases": [c.model_dump(exclude_none=True) for c in configs]
    }
    # 用同一目录下的临时文件，避免跨 FS
    dirpath = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".kbs_", suffix=".yaml.tmp", dir=dirpath)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                payload, f, allow_unicode=True, sort_keys=False, default_flow_style=False
            )
        os.replace(tmp, path)
    except Exception:
        # 清理临时文件
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


@lru_cache(maxsize=1)
def get_registry() -> KnowledgeBaseRegistry:
    """获取全局知识库注册表（单例）。"""
    configs = _load_from_yaml(AppPaths.KB_CONFIG)
    return KnowledgeBaseRegistry(configs)


def reset_registry_cache() -> None:
    """测试用：重置注册表缓存。"""
    get_registry.cache_clear()


# ===== 持久化辅助（管理面 API 用） =====

_registry_lock = threading.Lock()


def create_kb(payload: KnowledgeBaseCreate) -> KnowledgeBaseConfig:
    """新建 KB + 持久化。"""
    with _registry_lock:
        reg = get_registry()
        cfg = KnowledgeBaseConfig(
            kb_id=payload.kb_id,
            name=payload.name,
            description=payload.description,
            department=payload.department,
            collection=payload.collection or f"kb_{payload.kb_id}",
            embedding=payload.embedding,
            reranker=payload.reranker,
            enabled=payload.enabled,
        )
        reg.add(cfg)
        _dump_to_yaml(AppPaths.KB_CONFIG, reg.all_cfgs())
        return cfg


def update_kb(kb_id: str, patch: KnowledgeBaseUpdate) -> KnowledgeBaseConfig:
    """更新 KB + 持久化。"""
    with _registry_lock:
        reg = get_registry()
        new_cfg = reg.update(kb_id, patch)
        _dump_to_yaml(AppPaths.KB_CONFIG, reg.all_cfgs())
        return new_cfg


def delete_kb(kb_id: str) -> KnowledgeBaseConfig:
    """删除 KB + 持久化。"""
    with _registry_lock:
        reg = get_registry()
        removed = reg.delete(kb_id)
        _dump_to_yaml(AppPaths.KB_CONFIG, reg.all_cfgs())
        return removed
