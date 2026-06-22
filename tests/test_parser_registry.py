"""ParserRegistry 测试：加载 yaml → list / get / 错误处理。"""
from __future__ import annotations

import pytest


def test_list_parsers_returns_all_yaml_entries():
    from study_rag.capabilities.llamaindex.registry import ParserRegistry

    reg = ParserRegistry.from_yaml("configs/llamaindex.yaml")
    names = [p.name for p in reg.list()]
    # yaml 里的 6 个预置
    assert "whole" in names
    assert "sentence_512" in names
    assert "sentence_1024" in names
    assert "short_paragraph" in names
    assert "token_512" in names
    assert "semantic" in names
    assert len(names) == 6


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
