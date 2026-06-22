"""NodeParser 封装：把 LlamaIndex 的 NodeParser 包装为我们的接口。

支持的策略：
  - whole         整篇文档作为一个节点（最简单）
  - sentence      按句子/段落切分（SentenceSplitter）
  - semantic      按语义相似度切分（SemanticSplitterNodeParser，需 embed_model）
  - token         按 token 切分（TokenTextSplitter）
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["NodeParserFactory", "NodeParserConfig", "ParsedNode"]


class NodeParserConfig:
    """NodeParser 配置。

    字段：
      strategy:   "whole" | "sentence" | "semantic" | "token"
      chunk_size:        目标 chunk 大小（字符/token）
      chunk_overlap:     相邻 chunk 重叠
      separator:         句子切分分隔符（默认 "。!?；!?\n"）
      paragraph_separator: 段落切分分隔符（默认 "\n\n"）
    """

    def __init__(
        self,
        strategy: str = "sentence",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = "。!?；!?\n",
        paragraph_separator: str = "\n\n",
    ):
        if strategy not in ("whole", "sentence", "semantic", "token"):
            raise ValueError(
                f"Unknown parser strategy: {strategy}. "
                f"Expected: whole / sentence / semantic / token"
            )
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.paragraph_separator = paragraph_separator

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> NodeParserConfig:
        return cls(
            strategy=raw.get("strategy", "sentence"),
            chunk_size=raw.get("chunk_size", 512),
            chunk_overlap=raw.get("chunk_overlap", 50),
            separator=raw.get("separator", "。!?；!?\n"),
            paragraph_separator=raw.get("paragraph_separator", "\n\n"),
        )


class ParsedNode:
    """切块后的节点（轻量级，不依赖 LI 类型）。"""

    def __init__(
        self,
        node_id: str,
        text: str,
        doc_id: str,
        chunk_index: int,
        metadata: dict[str, Any] | None = None,
    ):
        self.node_id = node_id
        self.text = text
        self.doc_id = doc_id
        self.chunk_index = chunk_index
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "text": self.text,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


def _make_sentence_splitter(cfg: NodeParserConfig):
    """构造 LI SentenceSplitter。"""
    try:
        from llama_index.core.node_parser import SentenceSplitter
    except ImportError as e:
        raise ImportError(
            "SentenceSplitter 需要 llama-index-core. 安装: pip install llama-index-core"
        ) from e

    # SentenceSplitter 的 chunk_size 是 token 数（默认 1024）
    return SentenceSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        paragraph_separator=cfg.paragraph_separator,
    )


def _make_token_splitter(cfg: NodeParserConfig):
    """构造 LI TokenTextSplitter。"""
    from llama_index.core.node_parser import TokenTextSplitter

    return TokenTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )


def _make_semantic_splitter(cfg: NodeParserConfig, embed_model):
    """构造 LI SemanticSplitterNodeParser（需要 embed_model）。"""
    try:
        from llama_index.core.node_parser import SemanticSplitterNodeParser
    except ImportError as e:
        raise ImportError(
            "SemanticSplitterNodeParser 需要 llama-index-core. 安装: pip install llama-index-core"
        ) from e

    return SemanticSplitterNodeParser(
        buffer_size=1,  # 句子粒度
        breakpoint_percentile_threshold=95,
        embed_model=embed_model,
    )


class NodeParserFactory:
    """NodeParser 工厂。

    用法：
        parser = NodeParserFactory(NodeParserConfig(strategy="sentence", chunk_size=512))
        nodes = parser.parse(content="...", doc_id="doc-1", metadata={...})
    """

    def __init__(
        self,
        config: NodeParserConfig | None = None,
        embed_model: Any = None,  # 语义切块时需要
    ):
        self._config = config or NodeParserConfig()
        self._embed_model = embed_model
        self._splitter = None  # 懒构造
        logger.debug(
            "NodeParserFactory initialized: strategy=%s, chunk_size=%d",
            self._config.strategy,
            self._config.chunk_size,
        )

    @classmethod
    def from_raw(
        cls, raw: dict[str, Any], embed_model: Any = None
    ) -> NodeParserFactory:
        return cls(
            config=NodeParserConfig.from_dict(raw) if raw else None,
            embed_model=embed_model,
        )

    def _get_splitter(self):
        if self._splitter is not None:
            return self._splitter
        if self._config.strategy == "whole":
            self._splitter = None  # whole 不需要 splitter
        elif self._config.strategy == "sentence":
            self._splitter = _make_sentence_splitter(self._config)
        elif self._config.strategy == "token":
            self._splitter = _make_token_splitter(self._config)
        elif self._config.strategy == "semantic":
            if self._embed_model is None:
                raise ValueError(
                    "semantic strategy requires embed_model; "
                    "请传入 embed_model 或在 manager 中传入"
                )
            self._splitter = _make_semantic_splitter(self._config, self._embed_model)
        return self._splitter

    def parse(
        self,
        content: str,
        doc_id: str,
        title: str = "",
        source: str = "",
    ) -> list[ParsedNode]:
        """把整篇文档切成多个节点。"""
        if not content or not content.strip():
            return []

        # whole 策略：整篇作为一个节点
        if self._config.strategy == "whole":
            return [
                ParsedNode(
                    node_id=f"{doc_id}#0",
                    text=content,
                    doc_id=doc_id,
                    chunk_index=0,
                    metadata={"title": title, "source": source},
                )
            ]

        splitter = self._get_splitter()
        # 用 LI 的 Document 包装
        from llama_index.core.schema import Document

        doc = Document(
            text=content,
            doc_id=doc_id,
            metadata={"title": title, "source": source},
        )
        nodes = splitter.get_nodes_from_documents([doc])
        return [
            ParsedNode(
                node_id=f"{doc_id}#{i}",
                text=n.get_content(),
                doc_id=doc_id,
                chunk_index=i,
                metadata={**(n.metadata or {}), "node_ref_id": n.node_id},
            )
            for i, n in enumerate(nodes)
        ]
