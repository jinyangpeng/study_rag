"""DocumentReader 测试：纯文本 / Markdown / HTML / 未知格式 / 格式推断。"""
from __future__ import annotations

import pytest

from study_rag.capabilities.llamaindex.reader import (
    UnsupportedFormatError,
    read_document,
)


def test_plain_text_reader():
    content = b"Hello\n\nWorld"
    text, meta = read_document(content, filename="note.txt", mime="text/plain")
    assert "Hello" in text
    assert "World" in text
    assert meta["filename"] == "note.txt"
    assert meta["format"] == "txt"


def test_markdown_reader():
    md = b"# Title\n\nSome **bold** text."
    text, meta = read_document(md, filename="readme.md", mime="text/markdown")
    assert "Title" in text
    assert meta["format"] == "md"


def test_html_reader_strips_tags():
    html = b"<html><body><h1>X</h1><p>Hello <b>world</b></p></body></html>"
    text, meta = read_document(html, filename="page.html", mime="text/html")
    assert "Hello" in text
    assert "world" in text
    assert "<h1>" not in text
    assert meta["format"] == "html"


def test_unsupported_format_raises():
    with pytest.raises(UnsupportedFormatError):
        read_document(b"binary", filename="data.exe", mime="application/octet-stream")


def test_format_inference_from_filename():
    """mime 没传时，按后缀推断。"""
    text, meta = read_document(b"hello", filename="x.txt")
    assert meta["format"] == "txt"
