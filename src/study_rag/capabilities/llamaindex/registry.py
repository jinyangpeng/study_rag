"""ParserRegistry：从 configs/llamaindex.yaml 加载命名 parser 配置。

设计对齐 Embedder/Reranker Registry：
  - 单例（模块级），启动时 warm up
  - by name 查；找不到抛 KeyError
  - summary() 返回 UI 用的元信息（strategy / chunk_size / chunk_overlap）
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ...observability.logging import get_logger
from .parser import NodeParserConfig, NodeParserFactory

logger = get_logger(__name__)

__all__ = ["ParserRegistry", "ParserSpec", "get_parser_registry"]


class ParserSpec:
    """单个 parser 配置（命名实体）。"""

    def __init__(self, name: str, config: NodeParserConfig):
        self.name = name
        self.config = config

    def to_summary(self) -> dict[str, Any]:
        """UI 用的元信息（不含 factory 实例）。"""
        cfg = self.config
        return {
            "name": self.name,
            "strategy": cfg.strategy,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "paragraph_separator": cfg.paragraph_separator,
            "buffer_size": cfg.buffer_size,
            "breakpoint_percentile_threshold": cfg.breakpoint_percentile_threshold,
        }


class ParserRegistry:
    """Parser 注册表。"""

    def __init__(self, specs: dict[str, ParserSpec]):
        self._specs = specs

    @classmethod
    def from_yaml(cls, path: str | Path) -> ParserRegistry:
        """从 yaml 加载。yaml 结构见 configs/llamaindex.yaml。"""
        p = Path(path)
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        parsers_raw = raw.get("parsers", {})
        specs: dict[str, ParserSpec] = {}
        for name, cfg_dict in parsers_raw.items():
            cfg = NodeParserConfig.from_dict(cfg_dict)
            specs[name] = ParserSpec(name=name, config=cfg)
        logger.info("parser_registry_loaded", count=len(specs), path=str(p))
        return cls(specs)

    def list(self) -> list[ParserSpec]:
        return list(self._specs.values())

    def get(self, name: str, embed_model: Any = None) -> NodeParserFactory:
        """按名字取 NodeParserFactory。

        Args:
            name: parser 名称。
            embed_model: 可选；``semantic`` 策略时必须传，包装为 LI Embedding 协议。
                          其它策略忽略。
        """
        if name not in self._specs:
            available = sorted(self._specs.keys())
            raise KeyError(
                f"parser '{name}' not registered. Available: {available}"
            )
        return NodeParserFactory(
            config=self._specs[name].config, embed_model=embed_model
        )

    def summary(self, name: str) -> dict[str, Any]:
        if name not in self._specs:
            raise KeyError(f"parser '{name}' not registered")
        return self._specs[name].to_summary()

    def list_summaries(self) -> list[dict[str, Any]]:  # type: ignore[valid-type]
        return [s.to_summary() for s in self._specs.values()]


# ---- 单例 ----
_registry_singleton: ParserRegistry | None = None
_lock = threading.Lock()


def get_parser_registry(force_reload: bool = False) -> ParserRegistry:
    """进程级单例。"""
    global _registry_singleton
    if _registry_singleton is not None and not force_reload:
        return _registry_singleton
    with _lock:
        if _registry_singleton is not None and not force_reload:
            return _registry_singleton
        from ...settings import AppPaths

        _registry_singleton = ParserRegistry.from_yaml(AppPaths.LLAMAINDEX_CONFIG)
        return _registry_singleton
