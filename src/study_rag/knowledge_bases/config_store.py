"""Embedder / Reranker 配置存储：原子读写 YAML。

镜像 registry.py 的原子写模式（先写 tmp 再 os.replace），保证配置文件不会因写中途崩溃而损坏。

两份配置文件的结构：
  - embeddings.yaml → { embeddings: { name: {provider, model_name, dimension, batch_size, description, extra} } }
  - reranker.yaml   → { rerankers:  { name: {provider, protocol, model_name, top_k, description, extra} } }

管理面 API 通过本模块做 CRUD；运行时加载（_load_embedding_configs 等）仍走原路径，互不干扰。
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import yaml

from ..settings import AppPaths

_lock = threading.Lock()


class ConfigNotFoundError(KeyError):
    """配置项不存在。"""


# ===== 底层读写 =====


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, data: dict[str, Any], root_key: str) -> None:
    """原子写回 YAML：先写 .tmp，再 os.replace 覆盖。"""
    payload = {root_key: data}
    dirpath = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=f".{root_key}_", suffix=".yaml.tmp", dir=dirpath)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                payload, f, allow_unicode=True, sort_keys=False, default_flow_style=False
            )
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ===== Embedder =====

_EMBEDDING_ROOT = "embeddings"


def list_embedder_configs_raw() -> dict[str, dict[str, Any]]:
    """返回 embeddings.yaml 中所有 embedder 配置（原始 dict）。"""
    return dict(_read_yaml(AppPaths.EMBEDDING_CONFIG).get(_EMBEDDING_ROOT, {}))


def get_embedder_config_raw(name: str) -> dict[str, Any]:
    cfgs = list_embedder_configs_raw()
    if name not in cfgs:
        raise ConfigNotFoundError(name)
    return cfgs[name]


def create_embedder_config(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        all_cfgs = list_embedder_configs_raw()
        if name in all_cfgs:
            raise ValueError(f"embedder config already exists: {name}")
        all_cfgs[name] = cfg
        _write_yaml(AppPaths.EMBEDDING_CONFIG, all_cfgs, _EMBEDDING_ROOT)
        return cfg


def update_embedder_config(name: str, patch: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        all_cfgs = list_embedder_configs_raw()
        if name not in all_cfgs:
            raise ConfigNotFoundError(name)
        merged = {**all_cfgs[name], **patch}
        all_cfgs[name] = merged
        _write_yaml(AppPaths.EMBEDDING_CONFIG, all_cfgs, _EMBEDDING_ROOT)
        return merged


def delete_embedder_config(name: str) -> dict[str, Any]:
    with _lock:
        all_cfgs = list_embedder_configs_raw()
        if name not in all_cfgs:
            raise ConfigNotFoundError(name)
        removed = all_cfgs.pop(name)
        _write_yaml(AppPaths.EMBEDDING_CONFIG, all_cfgs, _EMBEDDING_ROOT)
        return removed


# ===== Reranker =====

_RERANKER_ROOT = "rerankers"


def list_reranker_configs_raw() -> dict[str, dict[str, Any]]:
    """返回 reranker.yaml 中所有 reranker 配置（原始 dict）。"""
    return dict(_read_yaml(AppPaths.RERANKER_CONFIG).get(_RERANKER_ROOT, {}))


def get_reranker_config_raw(name: str) -> dict[str, Any]:
    cfgs = list_reranker_configs_raw()
    if name not in cfgs:
        raise ConfigNotFoundError(name)
    return cfgs[name]


def create_reranker_config(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        all_cfgs = list_reranker_configs_raw()
        if name in all_cfgs:
            raise ValueError(f"reranker config already exists: {name}")
        all_cfgs[name] = cfg
        _write_yaml(AppPaths.RERANKER_CONFIG, all_cfgs, _RERANKER_ROOT)
        return cfg


def update_reranker_config(name: str, patch: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        all_cfgs = list_reranker_configs_raw()
        if name not in all_cfgs:
            raise ConfigNotFoundError(name)
        merged = {**all_cfgs[name], **patch}
        all_cfgs[name] = merged
        _write_yaml(AppPaths.RERANKER_CONFIG, all_cfgs, _RERANKER_ROOT)
        return merged


def delete_reranker_config(name: str) -> dict[str, Any]:
    with _lock:
        all_cfgs = list_reranker_configs_raw()
        if name not in all_cfgs:
            raise ConfigNotFoundError(name)
        removed = all_cfgs.pop(name)
        _write_yaml(AppPaths.RERANKER_CONFIG, all_cfgs, _RERANKER_ROOT)
        return removed


# ===== Parser（分块配置，llamaindex.yaml）=====
#
# llamaindex.yaml 结构：
#   parsers: { name: {strategy, chunk_size, ...} }
#   settings: { enabled, over_fetch, llm }   ← 全局设置，CRUD 时必须保留
#
# 因此 parser CRUD 用「读全量 → 改 parsers 部分 → 写全量」的策略，
# 避免原子写时丢失 settings 部分。

_PARSER_ROOT = "parsers"


def _read_llamaindex_full() -> dict[str, Any]:
    """读取 llamaindex.yaml 全量（含 parsers + settings）。"""
    return _read_yaml(AppPaths.LLAMAINDEX_CONFIG)


def _write_llamaindex_full(data: dict[str, Any]) -> None:
    """全量写回 llamaindex.yaml（非原子，因需保留 settings；用锁保证线程安全）。"""
    with AppPaths.LLAMAINDEX_CONFIG.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def list_parser_configs_raw() -> dict[str, dict[str, Any]]:
    """返回 llamaindex.yaml 中所有 parser 配置（原始 dict）。"""
    return dict(_read_llamaindex_full().get(_PARSER_ROOT, {}))


def get_parser_config_raw(name: str) -> dict[str, Any]:
    cfgs = list_parser_configs_raw()
    if name not in cfgs:
        raise ConfigNotFoundError(name)
    return cfgs[name]


def create_parser_config(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        full = _read_llamaindex_full()
        parsers = full.get(_PARSER_ROOT, {})
        if name in parsers:
            raise ValueError(f"parser config already exists: {name}")
        parsers[name] = cfg
        full[_PARSER_ROOT] = parsers
        _write_llamaindex_full(full)
        return cfg


def update_parser_config(name: str, patch: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        full = _read_llamaindex_full()
        parsers = full.get(_PARSER_ROOT, {})
        if name not in parsers:
            raise ConfigNotFoundError(name)
        merged = {**parsers[name], **patch}
        parsers[name] = merged
        full[_PARSER_ROOT] = parsers
        _write_llamaindex_full(full)
        return merged


def delete_parser_config(name: str) -> dict[str, Any]:
    with _lock:
        full = _read_llamaindex_full()
        parsers = full.get(_PARSER_ROOT, {})
        if name not in parsers:
            raise ConfigNotFoundError(name)
        removed = parsers.pop(name)
        full[_PARSER_ROOT] = parsers
        _write_llamaindex_full(full)
        return removed
