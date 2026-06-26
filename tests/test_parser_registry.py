"""ParserRegistry 测试：加载 yaml → list / get / 错误处理 / semantic 配置驱动切块。"""
from __future__ import annotations

import pytest


def test_list_parsers_returns_all_yaml_entries():
    from study_rag.capabilities.llamaindex.registry import ParserRegistry

    reg = ParserRegistry.from_yaml("configs/llamaindex.yaml")
    names = [p.name for p in reg.list()]
    # 6 个基础预置
    assert "whole" in names
    assert "sentence_512" in names
    assert "sentence_1024" in names
    assert "short_paragraph" in names
    assert "token_512" in names
    assert "semantic" in names
    # 3 个 semantic 变体（Phase 9.0/9.1）
    assert "semantic_aggressive" in names
    assert "semantic_conservative" in names
    # 1 个 sentence 变体
    assert "sentence_256" in names
    assert len(names) == 9


def test_get_parser_returns_factory():
    from study_rag.capabilities.llamaindex.parser import NodeParserFactory
    from study_rag.capabilities.llamaindex.registry import ParserRegistry

    reg = ParserRegistry.from_yaml("configs/llamaindex.yaml")
    factory = reg.get("sentence_512")
    assert isinstance(factory, NodeParserFactory)


def test_get_parser_unknown_raises():
    from study_rag.capabilities.llamaindex.registry import ParserRegistry

    reg = ParserRegistry.from_yaml("configs/llamaindex.yaml")
    with pytest.raises(KeyError, match="not_in_yaml"):
        reg.get("not_in_yaml")


def test_parser_summary_includes_strategy_and_size():
    from study_rag.capabilities.llamaindex.registry import ParserRegistry

    reg = ParserRegistry.from_yaml("configs/llamaindex.yaml")
    summary = reg.summary("sentence_512")
    assert summary["name"] == "sentence_512"
    assert summary["strategy"] == "sentence"
    assert summary["chunk_size"] == 512
    assert summary["chunk_overlap"] == 50


# ----- Phase 9.0: NodeParserConfig 接受 buffer_size + threshold -----


def test_node_parser_config_default_no_semantic_params():
    """默认配置（不传 buffer/threshold）应该是 None。"""
    from study_rag.capabilities.llamaindex.parser import NodeParserConfig

    cfg = NodeParserConfig(strategy="semantic")
    assert cfg.buffer_size is None
    assert cfg.breakpoint_percentile_threshold is None


def test_node_parser_config_accepts_semantic_params():
    """显式传 buffer/threshold 应保存。"""
    from study_rag.capabilities.llamaindex.parser import NodeParserConfig

    cfg = NodeParserConfig(
        strategy="semantic",
        buffer_size=5,
        breakpoint_percentile_threshold=60,
    )
    assert cfg.buffer_size == 5
    assert cfg.breakpoint_percentile_threshold == 60


def test_node_parser_config_from_dict_reads_semantic_params():
    """from_dict 应读 yaml 的 buffer_size / breakpoint_percentile_threshold。"""
    from study_rag.capabilities.llamaindex.parser import NodeParserConfig

    cfg = NodeParserConfig.from_dict(
        {
            "strategy": "semantic",
            "buffer_size": 3,
            "breakpoint_percentile_threshold": 80,
        }
    )
    assert cfg.strategy == "semantic"
    assert cfg.buffer_size == 3
    assert cfg.breakpoint_percentile_threshold == 80


# ----- Phase 9.1: 3 个 semantic_* 变体的 yaml 配置 -----


@pytest.mark.parametrize(
    "name,expected_buffer,expected_threshold",
    [
        ("semantic", 1, 95),
        ("semantic_aggressive", 5, 60),
        ("semantic_conservative", 1, 99),
    ],
)
def test_semantic_preset_loads_correct_params(
    name, expected_buffer, expected_threshold
):
    """每个 semantic 变体的 buffer/threshold 应从 yaml 读到。"""
    from study_rag.capabilities.llamaindex.registry import ParserRegistry

    reg = ParserRegistry.from_yaml("configs/llamaindex.yaml")
    summary = reg.summary(name)
    assert summary["strategy"] == "semantic"
    assert summary["buffer_size"] == expected_buffer
    assert summary["breakpoint_percentile_threshold"] == expected_threshold


def test_semantic_splitter_factory_uses_config_params():
    """_make_semantic_splitter 应使用 cfg.buffer_size / threshold。"""
    from unittest.mock import patch, MagicMock

    from study_rag.capabilities.llamaindex.parser import (
        NodeParserConfig,
        _make_semantic_splitter,
    )

    cfg = NodeParserConfig(
        strategy="semantic",
        buffer_size=7,
        breakpoint_percentile_threshold=42,
    )

    with patch(
        "llama_index.core.node_parser.SemanticSplitterNodeParser"
    ) as mock_parser:
        mock_instance = MagicMock()
        mock_parser.return_value = mock_instance
        mock_embed = MagicMock()

        result = _make_semantic_splitter(cfg, mock_embed)

    # 验证传入的参数是 cfg 里的值（不是写死的 1/95）
    call_kwargs = mock_parser.call_args.kwargs
    assert call_kwargs["buffer_size"] == 7
    assert call_kwargs["breakpoint_percentile_threshold"] == 42
    assert call_kwargs["embed_model"] is mock_embed


def test_semantic_splitter_factory_falls_back_to_defaults():
    """None 时应回退到 LI 默认（1, 95）。"""
    from unittest.mock import patch, MagicMock

    from study_rag.capabilities.llamaindex.parser import (
        NodeParserConfig,
        _make_semantic_splitter,
    )

    cfg = NodeParserConfig(strategy="semantic")  # 不传 buffer/threshold

    with patch(
        "llama_index.core.node_parser.SemanticSplitterNodeParser"
    ) as mock_parser:
        mock_instance = MagicMock()
        mock_parser.return_value = mock_instance

        _make_semantic_splitter(cfg, MagicMock())

    call_kwargs = mock_parser.call_args.kwargs
    assert call_kwargs["buffer_size"] == 1
    assert call_kwargs["breakpoint_percentile_threshold"] == 95
