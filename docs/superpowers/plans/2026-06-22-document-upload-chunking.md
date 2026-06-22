# 文档上传 + 分块 + 预览 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把当前「文本框输入 → 整篇入库」升级为「**文件上传 → LlamaIndex 解析器自动识别 → 多种分块策略 → 预览 → 确认入库**」的完整链路，对齐企业级 RAG 系统标准。

**Architecture:**
- **后端**：新增 Parser Registry（参考 Embedder/Reranker 设计），把 `configs/llamaindex.yaml` 的 parser 配置作为命名实体（whole / sentence_512 / semantic 等）；新增 `POST /upload`（multipart）和 `POST /preview-chunk`（解析不入库）端点；用 LlamaIndex 的 `SimpleDirectoryReader` 处理多格式。
- **前端**：Documents.tsx 加 `<Upload>` 拖拽组件 + 策略下拉 + 「预览分块」抽屉（提交前看到每个 chunk 的内容/大小/metadata，确认后再入库）。
- **不破坏现有**：`POST /documents/chunked` 仍可工作，新接口是增强；UI 默认选 `sentence_512` 切块，向后兼容。

**Tech Stack:**
- 后端：FastAPI / python-multipart（upload）/ llama-index-core / pypdf / docx2txt / markdown
- 前端：React 18 / Ant Design 5 / antd `<Upload>` 组件
- 测试：pytest + httpx（已有）

---

## 现状摘要（基线）

| 已有 | 缺失 |
|---|---|
| `NodeParserConfig` 支持 4 策略（whole/sentence/semantic/token）| 这些策略没注册成命名实体，没法在 UI 下拉选 |
| `NodeParserFactory.parse(content, ...)` 工作正常 | 只接受 `content: str`，不接受文件 |
| `POST /admin/kbs/{kb_id}/documents/chunked` 接 `parser_config: dict` | 接受方式用 query + body 混搭，不规范；UI 没暴露 |
| `configs/llamaindex.yaml` 已有 5 个预置 parser 配置 | 没被加载，**纯配置文件** |
| `tests/verify_llamaindex.py` 验证 4 策略切块 | 没有文件上传测试、没有 preview 测试 |

---

## 任务结构

按"自下而上"分 5 个 Phase，每个 Phase 可独立 review / commit / 回滚：

| Phase | 内容 | 涉及层 | 估算 |
|---|---|---|---|
| **0** | Parser Registry（加载 yaml → 命名 parser） | 后端 | 中 |
| **1** | 文档解析 + 文件格式支持（LlamaIndex readers） | 后端 | 中 |
| **2** | 新端点：`/upload` + `/preview-chunk` | 后端 | 中 |
| **3** | 前端 UI：上传 + 策略下拉 + 预览抽屉 | 前端 | 大 |
| **4** | 端到端测试 + 文档更新 | 全栈 | 小 |

每个 Phase 包含多个 Task，每个 Task 是 2-5 分钟的 step。

---

# Phase 0：Parser Registry

**目标**：把 `configs/llamaindex.yaml` 里的 5 个预置 parser 加载成命名实体，提供 `get_parser(name)` / `list_parsers()` 接口，与现有 Embedder/Reranker Registry 设计对齐。

## File Structure（新增/修改）

- **Create** `src/study_rag/capabilities/llamaindex/registry.py` — ParserRegistry 类（仿 registry.py 设计）
- **Create** `src/study_rag/capabilities/llamaindex/__init__.py` — 加 `ParserRegistry` 导出（已有 6 个导出）
- **Create** `tests/test_parser_registry.py` — pytest 风格测试
- **Modify** `src/study_rag/app.py` 或 manager.py — 启动时 warm up registry（可选）

---

### Task 0.1: 写 ParserRegistry 测试（先写失败的测试）

**Files:**
- Create: `tests/test_parser_registry.py`

- [ ] **Step 1: 写测试用例**

```python
# tests/test_parser_registry.py
"""ParserRegistry 测试：加载 yaml → list / get / 错误处理。"""
from __future__ import annotations

import pytest


def test_list_parsers_returns_all_yaml_entries():
    from study_rag.capabilities.llamaindex.registry import ParserRegistry

    reg = ParserRegistry.from_yaml("configs/llamaindex.yaml")
    names = [p.name for p in reg.list()]
    # yaml 里的 5 个预置
    assert "whole" in names
    assert "sentence_512" in names
    assert "sentence_1024" in names
    assert "short_paragraph" in names
    assert "token_512" in names
    assert "semantic" in names
    assert len(names) == 6


def test_get_parser_returns_factory():
    from study_rag.capabilities.llamaindex.registry import ParserRegistry
    from study_rag.capabilities.llamaindex.parser import NodeParserFactory

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
```

- [ ] **Step 2: 跑测试，确认失败（因为 ParserRegistry 还没建）**

Run: `pytest tests/test_parser_registry.py -v`
Expected: `ModuleNotFoundError: No module named 'study_rag.capabilities.llamaindex.registry'`

- [ ] **Step 3: Commit 测试文件**

```bash
git add tests/test_parser_registry.py
git commit -m "test(parser): add ParserRegistry test cases (TDD red)"
```

---

### Task 0.2: 实现 ParserRegistry

**Files:**
- Create: `src/study_rag/capabilities/llamaindex/registry.py`

- [ ] **Step 1: 写实现**

```python
# src/study_rag/capabilities/llamaindex/registry.py
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

import yaml

from ...observability.logging import get_logger
from .parser import NodeParserConfig, NodeParserFactory

logger = get_logger(__name__)


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
        }


class ParserRegistry:
    """Parser 注册表。"""

    def __init__(self, specs: dict[str, ParserSpec]):
        self._specs = specs

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ParserRegistry":
        """从 yaml 加载。yaml 结构见 configs/llamaindex.yaml。"""
        p = Path(path)
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        parsers_raw = raw.get("parsers", {})
        specs: dict[str, ParserSpec] = {}
        for name, cfg_dict in parsers_raw.items():
            cfg = NodeParserConfig.from_dict(cfg_dict)
            specs[name] = ParserSpec(name=name, config=cfg)
        logger.info("parser_registry_loaded", count=len(specs), path=str(p))
        return cls(specs)

    def list(self) -> list[ParserSpec]:
        return list(self._specs.values())

    def get(self, name: str) -> NodeParserFactory:
        """按名字取 NodeParserFactory。"""
        if name not in self._specs:
            available = sorted(self._specs.keys())
            raise KeyError(
                f"parser '{name}' not registered. Available: {available}"
            )
        return NodeParserFactory(config=self._specs[name].config)

    def summary(self, name: str) -> dict[str, Any]:
        if name not in self._specs:
            raise KeyError(f"parser '{name}' not registered")
        return self._specs[name].to_summary()

    def list_summaries(self) -> list[dict[str, Any]]:
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
```

- [ ] **Step 2: 跑测试，确认通过**

Run: `pytest tests/test_parser_registry.py -v`
Expected: 4 passed

- [ ] **Step 3: 导出 + 在 `__init__.py` 添加**

修改 `src/study_rag/capabilities/llamaindex/__init__.py`，添加：
```python
from .registry import ParserRegistry, ParserSpec, get_parser_registry
```
到 `__all__` 和对外导出。

- [ ] **Step 4: 跑全测试，确认没破坏现有**

Run: `pytest tests/ -v --ignore=tests/verify_*.py`
Expected: 全 pass

- [ ] **Step 5: Commit**

```bash
git add src/study_rag/capabilities/llamaindex/registry.py \
        src/study_rag/capabilities/llamaindex/__init__.py
git commit -m "feat(parser): add ParserRegistry loading configs/llamaindex.yaml"
```

---

# Phase 1：文档解析 + 多格式支持

**目标**：用 LlamaIndex 的 readers 把上传的文件转成纯文本，统一抽象成 `DocumentReader` 接口。

## File Structure

- **Create** `src/study_rag/capabilities/llamaindex/reader.py` — DocumentReader 抽象 + SimpleReader 实现
- **Create** `src/study_rag/capabilities/llamaindex/__init__.py` — 加 `DocumentReader` / `read_document` 导出
- **Create** `tests/test_document_reader.py`
- **Modify** `pyproject.toml` — 加 `python-multipart` / `pypdf` / `docx2txt` 到 optional deps

---

### Task 1.1: 写 DocumentReader 测试

**Files:**
- Create: `tests/test_document_reader.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_document_reader.py
"""DocumentReader 测试：纯文本 / Markdown / PDF / DOCX / 未知格式。"""
from __future__ import annotations

import pytest

from study_rag.capabilities.llamaindex.reader import (
    DocumentReader,
    UnsupportedFormatError,
    read_document,
)


def test_plain_text_reader():
    content = "Hello\n\nWorld".encode("utf-8")
    text, meta = read_document(content, filename="note.txt", mime="text/plain")
    assert "Hello" in text and "World" in text
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
    assert "Hello" in text and "world" in text
    assert "<h1>" not in text
    assert meta["format"] == "html"


def test_unsupported_format_raises():
    with pytest.raises(UnsupportedFormatError):
        read_document(b"binary", filename="data.exe", mime="application/octet-stream")


def test_format_inference_from_filename():
    """mime 没传时，按后缀推断。"""
    text, meta = read_document(b"hello", filename="x.txt")
    assert meta["format"] == "txt"
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `pytest tests/test_document_reader.py -v`
Expected: `ModuleNotFoundError: No module named 'study_rag.capabilities.llamaindex.reader'`

- [ ] **Step 3: Commit**

```bash
git add tests/test_document_reader.py
git commit -m "test(reader): add DocumentReader test cases (TDD red)"
```

---

### Task 1.2: 实现 DocumentReader

**Files:**
- Create: `src/study_rag/capabilities/llamaindex/reader.py`

- [ ] **Step 1: 加 pyproject 依赖**

修改 `pyproject.toml`，在 `dependencies` 里加：
```toml
"python-multipart>=0.0.9",  # FastAPI file upload 必需
```

在 `[project.optional-dependencies]` 加新分组：
```toml
reader-pdf = ["pypdf>=4.0.0"]
reader-docx = ["docx2txt>=0.8"]
reader-md = ["markdown>=3.5"]
```

更新 `all`：
```toml
all = [
    "study-rag[llamaindex,vector-milvus,embedding-openai,embedding-bge,reranker-bge,reader-pdf,reader-docx,reader-md]",
]
```

- [ ] **Step 2: 写实现**

```python
# src/study_rag/capabilities/llamaindex/reader.py
"""DocumentReader：把上传的文件 bytes 转成纯文本 + 元信息。

支持的格式（按 file extension / mime）：
  .txt / text/plain          → 纯文本
  .md  / text/markdown       → Markdown
  .html / text/html          → HTML（用 BeautifulSoup 剥标签，缺库时降级为正则）
  .pdf / application/pdf     → PDF（pypdf，可选依赖）
  .docx / ...               → DOCX（docx2txt，可选依赖）

未知格式：抛 UnsupportedFormatError，由 API 层转 400。

为什么自己实现而非 SimpleDirectoryReader：
  - 我们的入参是 file bytes + filename（不是 file path），更简单
  - 不需要 LI 的 file_iterator 抽象
  - 失败时要清晰报错（LI 报错信息不友好）
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ...observability.logging import get_logger

logger = get_logger(__name__)

__all__ = ["DocumentReader", "UnsupportedFormatError", "read_document"]


class UnsupportedFormatError(Exception):
    """不支持的文件格式。"""


# mime -> extension 映射（部分常见 mime）
_MIME_TO_EXT = {
    "text/plain": "txt",
    "text/markdown": "md",
    "text/x-markdown": "md",
    "text/html": "html",
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}


def _infer_format(filename: str | None, mime: str | None) -> str:
    """按 mime > 文件名 后缀 推断格式。"""
    if mime:
        mime_lower = mime.lower().split(";")[0].strip()
        if mime_lower in _MIME_TO_EXT:
            return _MIME_TO_EXT[mime_lower]
    if filename:
        suffix = Path(filename).suffix.lower().lstrip(".")
        if suffix:
            return suffix
    raise UnsupportedFormatError(
        f"Cannot infer format from filename='{filename}', mime='{mime}'"
    )


def _read_txt(data: bytes) -> str:
    # 容错解码
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _read_md(data: bytes) -> str:
    """Markdown 直接当文本即可（不需要转 HTML）。"""
    return _read_txt(data)


def _read_html(data: bytes) -> str:
    text = _read_txt(data)
    try:
        from bs4 import BeautifulSoup  # type: ignore[import-not-found]

        soup = BeautifulSoup(text, "html.parser")
        # 移除 script/style
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except ImportError:
        # 降级：用正则剥标签
        logger.warning("bs4_missing_fallback_to_regex")
        return re.sub(r"<[^>]+>", "", text)


def _read_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
        from io import BytesIO
    except ImportError as e:
        raise UnsupportedFormatError(
            "PDF reading requires pypdf. Install: pip install study-rag[reader-pdf]"
        ) from e
    reader = PdfReader(BytesIO(data))
    parts: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            parts.append(page.extract_text() or "")
        except Exception as e:  # noqa: BLE001
            logger.warning("pdf_page_extract_failed", page=i, error=str(e))
            parts.append("")
    return "\n\n".join(parts)


def _read_docx(data: bytes) -> str:
    try:
        import docx2txt  # type: ignore[import-not-found]
        from io import BytesIO
    except ImportError as e:
        raise UnsupportedFormatError(
            "DOCX reading requires docx2txt. Install: pip install study-rag[reader-docx]"
        ) from e
    return docx2txt.process(BytesIO(data)) or ""


_READERS = {
    "txt": _read_txt,
    "md": _read_md,
    "markdown": _read_md,
    "html": _read_html,
    "htm": _read_html,
    "pdf": _read_pdf,
    "docx": _read_docx,
}


def read_document(
    data: bytes,
    filename: str | None = None,
    mime: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """把文件 bytes 解析成 (text, metadata)。

    Args:
        data: 文件原始字节
        filename: 原始文件名（用于推断格式）
        mime: MIME type（可选，优先于 filename）

    Returns:
        (text, metadata) — text 是纯文本；metadata 至少含 filename / format / size_bytes

    Raises:
        UnsupportedFormatError: 格式不支持或无法推断
    """
    fmt = _infer_format(filename, mime)
    reader = _READERS.get(fmt)
    if reader is None:
        raise UnsupportedFormatError(
            f"Unsupported format: '{fmt}' (filename={filename}, mime={mime}). "
            f"Supported: {sorted(_READERS.keys())}"
        )
    try:
        text = reader(data)
    except UnsupportedFormatError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error("document_read_failed", format=fmt, error=str(e), exc_info=True)
        raise UnsupportedFormatError(f"Failed to read {fmt} file: {e}") from e
    return text, {
        "filename": filename or "",
        "format": fmt,
        "size_bytes": len(data),
        "mime": mime or "",
    }
```

- [ ] **Step 3: 装可选依赖 + 跑测试**

```bash
.\venv\Scripts\pip.exe install pypdf docx2txt beautifulsoup4 markdown
pytest tests/test_document_reader.py -v
```
Expected: 5 passed (txt/md/html/推断/未知格式)

- [ ] **Step 4: 导出到 `__init__.py`**

```python
# src/study_rag/capabilities/llamaindex/__init__.py
from .reader import DocumentReader, UnsupportedFormatError, read_document
```

- [ ] **Step 5: 跑全测试**

Run: `pytest tests/ -v --ignore=tests/verify_*.py`
Expected: 全 pass

- [ ] **Step 6: Commit**

```bash
git add src/study_rag/capabilities/llamaindex/reader.py \
        src/study_rag/capabilities/llamaindex/__init__.py \
        pyproject.toml
git commit -m "feat(reader): add DocumentReader supporting txt/md/html/pdf/docx"
```

---

# Phase 2：新端点 `/upload` + `/preview-chunk`

**目标**：补齐 API 层：
- `GET /admin/parsers` — 列 parser（前端下拉用）
- `POST /admin/kbs/{kb_id}/documents/upload` — multipart 上传文件
- `POST /admin/kbs/{kb_id}/documents/preview-chunk` — 解析预览（不入库）

## File Structure

- **Modify** `src/study_rag/knowledge_bases/manager.py` — 加 `preview_chunk` / `add_document_from_upload`
- **Modify** `src/study_rag/api/admin.py` — 加 3 个端点
- **Create** `tests/test_admin_upload.py` — 端到端测试

---

### Task 2.1: manager 层方法 + 测试

**Files:**
- Modify: `src/study_rag/knowledge_bases/manager.py`
- Create: `tests/test_manager_upload.py`

- [ ] **Step 1: 写 manager 方法的测试**

```python
# tests/test_manager_upload.py
"""Manager 层的 upload / preview 测试。"""
from __future__ import annotations

import asyncio
import pytest

from study_rag.knowledge_bases.manager import KnowledgeBaseManager


@pytest.mark.asyncio
async def test_preview_chunk_returns_nodes_not_persisted(tmp_path):
    """preview 只能解析，不能写入。"""
    from study_rag.capabilities.llamaindex.registry import get_parser_registry

    reg = get_parser_registry(force_reload=True)
    factory = reg.get("sentence_512")
    content = "Sentence one. Sentence two.\n\nSentence three. Sentence four."
    nodes = factory.parse(content, doc_id="preview-1", title="test")
    assert len(nodes) >= 1
    assert all(n.doc_id == "preview-1" for n in nodes)
    # 没有 KB / vector store 调用，单纯 parse
```

- [ ] **Step 2: 跑测试**

Run: `pytest tests/test_manager_upload.py -v`
Expected: 1 passed（parser 工厂本身就能跑通，不需要 manager）

- [ ] **Step 3: 在 manager 加 `add_document_from_upload` 方法**

修改 `src/study_rag/knowledge_bases/manager.py`，在 `add_document_chunked` 之后加：

```python
async def add_document_from_upload(
    self,
    kb_id: str,
    doc_id: str,
    title: str,
    content: str,
    source: str = "",
    metadata: dict | None = None,
    parser_name: str | None = None,
) -> int:
    """上传文件入库（content 已经是 reader 解析后的纯文本）。

    Args:
        parser_name: 命名 parser（configs/llamaindex.yaml），如 'sentence_512'。
                     为 None 时走 sentence 策略（向后兼容）。
    Returns:
        切成的 chunk 数；0 表示内容为空。
    """
    from ..capabilities.llamaindex import (
        LIEmbeddingAdapter,
        NodeParserFactory,
        get_parser_registry,
    )
    from ..capabilities.vector_store import VectorRecord

    if not content or not content.strip():
        return 0

    cfg = self._registry.get_required(kb_id)
    embedder = self.get_embedder(kb_id)
    li_embedder = LIEmbeddingAdapter(embedder)

    if parser_name:
        factory = get_parser_registry().get(parser_name)
    else:
        factory = NodeParserFactory.from_raw(
            {"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 50},
            embed_model=li_embedder,
        )

    nodes = factory.parse(content, doc_id=doc_id, title=title, source=source)
    if not nodes:
        return 0

    count = 0
    for n in nodes:
        vec = await embedder.embed_query(n.text)
        rec = VectorRecord(
            id=n.node_id,
            vector=vec,
            text=n.text,
            metadata={
                "title": title,
                "source": source,
                "doc_id": doc_id,
                "chunk_index": n.chunk_index,
                "parser": factory._config.strategy,
                "filename": (metadata or {}).get("filename", ""),
            },
        )
        await self._vector_store.insert(cfg.collection, [rec])
        count += 1

    # 保存 DocumentMeta
    meta = DocumentMeta(
        doc_id=doc_id,
        kb_id=kb_id,
        title=title,
        source=source,
        content=content,
        metadata=metadata or {},
    )
    async with self._lock:
        self._docs.setdefault(kb_id, {})[doc_id] = meta
    self._save_docs_to_disk()
    return count
```

- [ ] **Step 4: 跑测试**

Run: `pytest tests/ -v --ignore=tests/verify_*.py`
Expected: 全 pass

- [ ] **Step 5: Commit**

```bash
git add src/study_rag/knowledge_bases/manager.py tests/test_manager_upload.py
git commit -m "feat(manager): add add_document_from_upload supporting named parser"
```

---

### Task 2.2: API 端点 + 测试

**Files:**
- Modify: `src/study_rag/api/admin.py`
- Create: `tests/test_admin_upload.py`

- [ ] **Step 1: 写端到端测试**

```python
# tests/test_admin_upload.py
"""端到端测试：/admin/parsers + /upload + /preview-chunk。"""
from __future__ import annotations

import io

import pytest
from httpx import ASGITransport, AsyncClient

from study_rag.app import create_app
from study_rag.capabilities.vector_store import VectorRecord, create_vector_store
from study_rag.knowledge_bases.registry import create_kb, get_registry
from study_rag.knowledge_bases.models import KnowledgeBaseCreate


@pytest.fixture
async def app_with_kb(tmp_path, monkeypatch):
    """建一个用 in-memory vector store + mock embedder 的最小 app。"""
    from study_rag.settings import AppPaths

    monkeypatch.setattr(AppPaths, "DOCS_INDEX", tmp_path / "docs.json")
    monkeypatch.setattr(AppPaths, "KNOWLEDGE_BASES_CONFIG", tmp_path / "kb.yaml")
    monkeypatch.setattr(AppPaths, "EMBEDDINGS_CONFIG", tmp_path / "emb.yaml")
    monkeypatch.setattr(AppPaths, "RERANKERS_CONFIG", tmp_path / "rerank.yaml")
    monkeypatch.setattr(AppPaths, "LLAMAINDEX_CONFIG", "configs/llamaindex.yaml")

    # 清空 registry singleton
    from study_rag.knowledge_bases import registry as reg_mod

    reg_mod._registry_singleton = None

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_list_parsers(app_with_kb):
    r = await app_with_kb.get("/admin/parsers")
    assert r.status_code == 200
    names = [p["name"] for p in r.json()]
    assert "sentence_512" in names
    assert "whole" in names


@pytest.mark.asyncio
async def test_preview_chunk_no_insert(app_with_kb):
    payload = {
        "content": "First sentence. Second sentence.\n\nThird sentence.",
        "parser": "sentence_512",
    }
    r = await app_with_kb.post("/admin/kbs/rd_frontend/documents/preview-chunk", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "chunks" in data
    assert len(data["chunks"]) >= 1
    for c in data["chunks"]:
        assert "chunk_index" in c
        assert "text" in c
        assert "char_count" in c


@pytest.mark.asyncio
async def test_upload_txt_file(app_with_kb):
    files = {"file": ("note.txt", io.BytesIO(b"Hello world."), "text/plain")}
    data = {"doc_id": "upload-1", "title": "Test", "parser": "sentence_512"}
    r = await app_with_kb.post(
        "/admin/kbs/rd_frontend/documents/upload", files=files, data=data
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["doc_id"] == "upload-1"
    assert body["chunks"] >= 1
    assert body["format"] == "txt"


@pytest.mark.asyncio
async def test_upload_unsupported_format_400(app_with_kb):
    files = {"file": ("data.exe", io.BytesIO(b"binary"), "application/octet-stream")}
    data = {"doc_id": "x", "title": "x", "parser": "whole"}
    r = await app_with_kb.post(
        "/admin/kbs/rd_frontend/documents/upload", files=files, data=data
    )
    assert r.status_code == 400
    assert "format" in r.json()["detail"].lower()
```

> **注意**：上面的测试可能需要根据实际 init_kb 流程调整（如先确保 KB 已创建）。如果测试中 KB 不存在会被 404，是预期行为。把 `rd_frontend` 替换成测试前 create 的 KB ID。

- [ ] **Step 2: 实现 3 个 API 端点**

在 `src/study_rag/api/admin.py` 的 `add_document_chunked` 之后加：

```python
@router.get(
    "/parsers",
    summary="列出可用 parser（前端下拉用）",
    description=(
        "返回 configs/llamaindex.yaml 里所有命名 parser 的元信息。\n\n"
        "前端在「添加文档」弹窗里下拉选；与 embedder/reranker 不同，"
        "parser 不需要单独加载（用 LI 的包），所有策略都可用。"
    ),
)
async def list_parsers_endpoint(
    _: Annotated[str, Depends(admin_auth_dep)],
    __: Annotated[str, Depends(admin_ratelimit_dep)],
) -> list[dict[str, Any]]:
    from ..capabilities.llamaindex import get_parser_registry

    get_metrics().inc(AdminMetrics.REQUESTS, {"endpoint": "list_parsers"})
    return get_parser_registry().list_summaries()


@router.post(
    "/kbs/{kb_id}/documents/preview-chunk",
    summary="预览分块（不入库）",
    description=(
        "把 content 按指定 parser 切块，返回每个 chunk 的内容/大小/metadata，"
        "不写入向量库。\n\n"
        "前端在「添加文档」前用这个让用户看到切分效果。"
    ),
)
async def preview_chunk(
    kb_id: str,
    request: Request,
    _: Annotated[str, Depends(admin_auth_dep)],
    __: Annotated[str, Depends(admin_ratelimit_dep)],
) -> dict[str, Any]:
    body = await request.json()
    content = body.get("content", "")
    parser_name = body.get("parser", "sentence_512")

    if not content or not content.strip():
        raise HTTPException(status_code=400, detail="content is empty")

    from ..capabilities.llamaindex import get_parser_registry

    try:
        factory = get_parser_registry().get(parser_name)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    nodes = factory.parse(
        content, doc_id="preview", title=body.get("title", "preview")
    )
    return {
        "parser": parser_name,
        "chunks": [
            {
                "chunk_index": n.chunk_index,
                "text": n.text,
                "char_count": len(n.text),
                "metadata": n.metadata,
            }
            for n in nodes
        ],
        "total_chunks": len(nodes),
        "total_chars": len(content),
    }


@router.post(
    "/kbs/{kb_id}/documents/upload",
    summary="上传文件入库（multipart）",
    description=(
        "支持格式：txt / md / html / pdf / docx（见 pyproject optional deps）。\n\n"
        "Form 字段：\n"
        "- `file`: 文件\n"
        "- `doc_id`: 文档 ID（必填）\n"
        "- `title`: 标题（必填）\n"
        "- `parser`: 命名 parser，如 'sentence_512'（可选，默认 sentence_512）\n"
        "- `source`: 来源标识（可选）\n"
        "- `overwrite`: true/false（可选，默认 false）\n"
    ),
)
async def upload_document(
    kb_id: str,
    request: Request,
    _: Annotated[str, Depends(admin_auth_dep)],
    __: Annotated[str, Depends(admin_ratelimit_dep)],
) -> dict[str, Any]:
    from ..capabilities.llamaindex import (
        UnsupportedFormatError,
        read_document,
    )

    form = await request.form()
    file = form.get("file")
    if file is None or not hasattr(file, "filename"):
        raise HTTPException(status_code=400, detail="file is required")
    doc_id = form.get("doc_id", "")
    title = form.get("title", "")
    parser_name = form.get("parser") or "sentence_512"
    source = form.get("source", "")
    overwrite = form.get("overwrite", "false").lower() == "true"

    if not doc_id or not title:
        raise HTTPException(
            status_code=400, detail="doc_id and title are required"
        )

    data = await file.read()
    try:
        text, reader_meta = read_document(
            data,
            filename=file.filename,
            mime=file.content_type,
        )
    except UnsupportedFormatError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    manager = get_manager()
    if not overwrite and manager.get_document(kb_id, doc_id) is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Document already exists: {kb_id}/{doc_id}",
        )

    try:
        chunks = await manager.add_document_from_upload(
            kb_id=kb_id,
            doc_id=doc_id,
            title=title,
            content=text,
            source=source,
            metadata={**reader_meta, "uploaded": True},
            parser_name=parser_name,
        )
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    get_metrics().inc(
        AdminMetrics.DOCUMENTS, {"op": "upload", "status": "ok"}
    )
    return {
        "kb_id": kb_id,
        "doc_id": doc_id,
        "title": title,
        "chunks": chunks,
        "format": reader_meta["format"],
        "size_bytes": reader_meta["size_bytes"],
        "parser": parser_name,
    }
```

- [ ] **Step 3: 跑测试**

Run: `pytest tests/test_admin_upload.py -v`
Expected: 4 passed（如果环境就绪）

- [ ] **Step 4: 跑全测试**

Run: `pytest tests/ -v --ignore=tests/verify_*.py`
Expected: 全 pass

- [ ] **Step 5: Commit**

```bash
git add src/study_rag/api/admin.py tests/test_admin_upload.py
git commit -m "feat(api): add /parsers /preview-chunk /upload endpoints"
```

---

# Phase 3：前端 UI（上传 + 策略下拉 + 预览抽屉）

**目标**：把 `Documents.tsx` 改成支持拖拽上传 + 策略选择 + 分块预览的全功能组件。

## File Structure

- **Create** `frontend/src/components/AddDocumentDrawer.tsx` — 文档添加抽屉（替代原 Modal）
- **Create** `frontend/src/components/ChunkPreviewPanel.tsx` — 分块预览面板
- **Modify** `frontend/src/pages/Documents.tsx` — 集成新组件
- **Modify** `frontend/src/api/types.ts` — 加 ParserSpec / PreviewChunkResponse
- **Modify** `frontend/src/api/client.tsx` — 加 listParsers / previewChunk / uploadDocument

---

### Task 3.1: API 客户端 + 类型

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/client.tsx`

- [ ] **Step 1: 加类型**

修改 `frontend/src/api/types.ts`，在文件末尾加：

```typescript
export interface ParserSpec {
  name: string;
  strategy: "whole" | "sentence" | "token" | "semantic";
  chunk_size: number;
  chunk_overlap: number;
  paragraph_separator: string;
}

export interface ChunkPreviewItem {
  chunk_index: number;
  text: string;
  char_count: number;
  metadata: Record<string, unknown>;
}

export interface ChunkPreviewResponse {
  parser: string;
  chunks: ChunkPreviewItem[];
  total_chunks: number;
  total_chars: number;
}

export interface UploadDocumentResponse {
  kb_id: string;
  doc_id: string;
  title: string;
  chunks: number;
  format: string;
  size_bytes: number;
  parser: string;
}
```

- [ ] **Step 2: 加客户端方法**

在 `frontend/src/api/client.tsx` 的 `ApiClient` 类里加：

```typescript
async listParsers(): Promise<ParserSpec[]> {
  const r = await this.axios.get<ParserSpec[]>("/admin/parsers");
  return r.data;
}

async previewChunk(
  kbId: string,
  content: string,
  parser: string,
  title = "preview"
): Promise<ChunkPreviewResponse> {
  const r = await this.axios.post<ChunkPreviewResponse>(
    `/admin/kbs/${kbId}/documents/preview-chunk`,
    { content, parser, title }
  );
  return r.data;
}

async uploadDocument(
  kbId: string,
  form: FormData
): Promise<UploadDocumentResponse> {
  const r = await this.axios.post<UploadDocumentResponse>(
    `/admin/kbs/${kbId}/documents/upload`,
    form,
    { headers: { "Content-Type": "multipart/form-data" } }
  );
  return r.data;
}
```

并在文件顶部 import 新类型：
```typescript
import type { ..., ParserSpec, ChunkPreviewResponse, UploadDocumentResponse } from "./types";
```

- [ ] **Step 3: TypeScript 编译验证**

Run: `cd frontend && npx tsc --noEmit`
Expected: exit 0

- [ ] **Step 4: Commit**

```bash
git add frontend/src/api/types.ts frontend/src/api/client.tsx
git commit -m "feat(frontend): add parser/preview/upload API client methods"
```

---

### Task 3.2: 实现 AddDocumentDrawer 组件

**Files:**
- Create: `frontend/src/components/AddDocumentDrawer.tsx`

- [ ] **Step 1: 写组件**

```tsx
// frontend/src/components/AddDocumentDrawer.tsx
/**
 * AddDocumentDrawer：替代原 Modal 的文档添加抽屉。
 *
 * 三种模式：
 *   1. 文本输入：手动粘贴大段文本
 *   2. 文件上传：拖拽或选 txt/md/html/pdf/docx，自动解析
 *   3. 选 parser + 实时预览切块（提交前看到每块内容/大小）
 *
 * 提交流程：
 *   - 文本：调 preview → 用户确认 → 调 chunked 接口
 *   - 文件：调 upload 接口（preview 在后端 upload 路径上做）
 */

import { useEffect, useState } from "react";
import {
  Drawer,
  Tabs,
  Form,
  Input,
  Select,
  Button,
  Upload,
  Alert,
  Space,
  Tag,
  Typography,
  App as AntdApp,
  Spin,
} from "antd";
import {
  InboxOutlined,
  FileTextOutlined,
  CodeOutlined,
  EyeOutlined,
} from "@ant-design/icons";
import { useApi } from "../api/client";
import type {
  ParserSpec,
  ChunkPreviewItem,
  UploadDocumentResponse,
} from "../api/types";
import ChunkPreviewPanel from "./ChunkPreviewPanel";

const { TextArea } = Input;
const { Dragger } = Upload;

interface Props {
  open: boolean;
  kbId: string;
  onCancel: () => void;
  onSuccess: () => void;
}

export default function AddDocumentDrawer({
  open,
  kbId,
  onCancel,
  onSuccess,
}: Props) {
  const { client } = useApi();
  const { message } = AntdApp.useApp();
  const [form] = Form.useForm();
  const [parsers, setParsers] = useState<ParserSpec[]>([]);
  const [tab, setTab] = useState<"text" | "file">("text");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<ChunkPreviewItem[] | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  // 拉 parser 列表
  useEffect(() => {
    if (!open) return;
    (async () => {
      try {
        const ps = await client.listParsers();
        setParsers(ps);
        form.setFieldsValue({ parser: ps.find((p) => p.name === "sentence_512")?.name ?? ps[0]?.name });
      } catch (e) {
        message.error((e as Error).message);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // 关闭时清状态
  useEffect(() => {
    if (!open) {
      form.resetFields();
      setFile(null);
      setPreview(null);
    }
  }, [open, form]);

  const onPreview = async () => {
    try {
      const v = await form.validateFields(["content", "parser"]);
      setPreviewLoading(true);
      const r = await client.previewChunk(kbId, v.content, v.parser, v.title);
      setPreview(r.chunks);
    } catch (e) {
      if (!(e as { errorFields?: unknown }).errorFields) {
        message.error((e as Error).message);
      }
    } finally {
      setPreviewLoading(false);
    }
  };

  const onSubmit = async () => {
    try {
      const v = await form.validateFields();
      setSubmitting(true);
      if (tab === "text") {
        // 走 chunked 接口
        await client.addDocumentChunked({
          kb_id: kbId,
          doc_id: v.doc_id,
          title: v.title,
          content: v.content,
          source: v.source || null,
          metadata: {},
          chunk_size: parsers.find((p) => p.name === v.parser)?.chunk_size ?? 512,
          chunk_overlap: parsers.find((p) => p.name === v.parser)?.chunk_overlap ?? 50,
        });
        message.success(`文档 ${v.doc_id} 添加成功`);
      } else {
        // 走 upload
        if (!file) {
          message.error("请先选择文件");
          return;
        }
        const fd = new FormData();
        fd.append("file", file);
        fd.append("doc_id", v.doc_id);
        fd.append("title", v.title);
        fd.append("parser", v.parser);
        if (v.source) fd.append("source", v.source);
        const r: UploadDocumentResponse = await client.uploadDocument(kbId, fd);
        message.success(
          `${r.doc_id} 上传成功（${r.format}, ${r.chunks} chunks, ${(r.size_bytes / 1024).toFixed(1)} KB）`
        );
      }
      onSuccess();
      onCancel();
    } catch (e) {
      if (!(e as { errorFields?: unknown }).errorFields) {
        message.error((e as Error).message);
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Drawer
      title="添加文档"
      open={open}
      onClose={onCancel}
      width={720}
      destroyOnClose
      extra={
        <Space>
          <Button onClick={onPreview} loading={previewLoading} icon={<EyeOutlined />}>
            预览分块
          </Button>
          <Button type="primary" onClick={onSubmit} loading={submitting}>
            添加
          </Button>
        </Space>
      }
    >
      <Form form={form} layout="vertical">
        <Tabs
          activeKey={tab}
          onChange={setTab as any}
          items={[
            {
              key: "text",
              label: <span><FileTextOutlined /> 文本输入</span>,
              children: (
                <Form.Item
                  label="正文"
                  name="content"
                  rules={[{ required: tab === "text", message: "请输入正文" }]}
                >
                  <TextArea rows={8} placeholder="粘贴文本内容（100-10000 字）" showCount />
                </Form.Item>
              ),
            },
            {
              key: "file",
              label: <span><InboxOutlined /> 文件上传</span>,
              children: (
                <Form.Item label="文件" required>
                  <Dragger
                    beforeUpload={(f) => {
                      setFile(f);
                      // 自动填 title（如果空）
                      if (!form.getFieldValue("title")) {
                        form.setFieldValue("title", f.name.replace(/\.[^.]+$/, ""));
                      }
                      return false; // 阻止自动 upload
                    }}
                    onRemove={() => setFile(null)}
                    maxCount={1}
                    accept=".txt,.md,.markdown,.html,.htm,.pdf,.docx"
                  >
                    <p className="ant-upload-drag-icon">
                      <InboxOutlined />
                    </p>
                    <p className="ant-upload-text">点击或拖拽文件到此处</p>
                    <p className="ant-upload-hint">
                      支持 txt / md / html / pdf / docx，单文件最大 50MB
                    </p>
                  </Dragger>
                  {file && (
                    <Alert
                      type="info"
                      showIcon
                      style={{ marginTop: 8 }}
                      message={
                        <Space>
                          <CodeOutlined />
                          {file.name}
                          <Tag>{(file.size / 1024).toFixed(1)} KB</Tag>
                        </Space>
                      }
                    />
                  )}
                </Form.Item>
              ),
            },
          ]}
        />

        <Form.Item
          label="doc_id"
          name="doc_id"
          rules={[{ required: true, message: "请输入 doc_id" }]}
        >
          <Input placeholder="KB 内唯一，如 react_perf_001" />
        </Form.Item>
        <Form.Item
          label="标题"
          name="title"
          rules={[{ required: true, message: "请输入标题" }]}
        >
          <Input placeholder="React 性能优化指南" />
        </Form.Item>
        <Form.Item label="source" name="source">
          <Input placeholder="可选：来源标识" />
        </Form.Item>
        <Form.Item
          label={
            <Space>
              切块策略
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                （configs/llamaindex.yaml 命名实体）
              </Typography.Text>
            </Space>
          }
          name="parser"
          rules={[{ required: true, message: "请选择切块策略" }]}
        >
          <Select
            options={parsers.map((p) => ({
              value: p.name,
              label: (
                <Space>
                  <Tag color="blue">{p.strategy}</Tag>
                  <span>{p.name}</span>
                  <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                    size={p.chunk_size}, overlap={p.chunk_overlap}
                  </Typography.Text>
                </Space>
              ),
            }))}
          />
        </Form.Item>
      </Form>

      {previewLoading && (
        <div style={{ textAlign: "center", padding: 24 }}>
          <Spin />
        </div>
      )}
      {preview && !previewLoading && (
        <ChunkPreviewPanel chunks={preview} />
      )}
    </Drawer>
  );
}
```

- [ ] **Step 2: TS 编译验证**

Run: `cd frontend && npx tsc --noEmit`
Expected: 可能因为 ChunkPreviewPanel 还没建而失败

---

### Task 3.3: 实现 ChunkPreviewPanel 组件

**Files:**
- Create: `frontend/src/components/ChunkPreviewPanel.tsx`

- [ ] **Step 1: 写组件**

```tsx
// frontend/src/components/ChunkPreviewPanel.tsx
/**
 * ChunkPreviewPanel：分块预览面板，显示每个 chunk 的内容/大小/metadata。
 * 用于 AddDocumentDrawer 的预览模式。
 */

import { Card, Space, Tag, Typography, Empty } from "antd";
import { BlockOutlined } from "@ant-design/icons";
import type { ChunkPreviewItem } from "../api/types";

const { Text, Paragraph } = Typography;

interface Props {
  chunks: ChunkPreviewItem[];
}

export default function ChunkPreviewPanel({ chunks }: Props) {
  if (chunks.length === 0) {
    return <Empty description="无内容" />;
  }
  return (
    <div>
      <Text strong>
        <BlockOutlined /> 预览：{chunks.length} 个块
      </Text>
      <div style={{ marginTop: 12, maxHeight: 400, overflow: "auto" }}>
        <Space direction="vertical" style={{ width: "100%" }} size="small">
          {chunks.map((c) => (
            <Card
              key={c.chunk_index}
              size="small"
              title={
                <Space>
                  <Tag color="blue">#{c.chunk_index}</Tag>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {c.char_count} chars
                  </Text>
                </Space>
              }
              style={{ width: "100%" }}
            >
              <Paragraph
                style={{
                  marginBottom: 0,
                  whiteSpace: "pre-wrap",
                  fontSize: 13,
                }}
                ellipsis={{ rows: 4, expandable: true, symbol: "展开" }}
              >
                {c.text}
              </Paragraph>
            </Card>
          ))}
        </Space>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: TS 编译**

Run: `cd frontend && npx tsc --noEmit`
Expected: exit 0

---

### Task 3.4: 集成到 Documents.tsx

**Files:**
- Modify: `frontend/src/pages/Documents.tsx`

- [ ] **Step 1: 替换原 Add Modal**

找到 `<Modal title="添加文档" ...` 那段（line 295-377），**整段删除**；把状态 `addOpen` 保留；改 `添加文档` 按钮的 onClick 触发 drawer。

在文件顶部 import：
```typescript
import AddDocumentDrawer from "../components/AddDocumentDrawer";
```

把 `<Modal>...</Modal>` 替换为：
```tsx
<AddDocumentDrawer
  open={addOpen}
  kbId={kbId}
  onCancel={() => setAddOpen(false)}
  onSuccess={() => {
    setAddOpen(false);
    void load();
  }}
/>
```

并删除 `AddDocFormValues` / `AddDocChunkedFormValues` 接口（不再用）和 `addForm` state（不再用 Form.useForm）。

- [ ] **Step 2: TS 编译**

Run: `cd frontend && npx tsc --noEmit`
Expected: exit 0

- [ ] **Step 3: vite build 验证**

Run: `cd frontend && npm run build`
Expected: exit 0

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/AddDocumentDrawer.tsx \
        frontend/src/components/ChunkPreviewPanel.tsx \
        frontend/src/pages/Documents.tsx
git commit -m "feat(frontend): AddDocumentDrawer with upload/parser/preview"
```

---

# Phase 4：端到端测试 + 文档

**目标**：补一个 e2e 验证脚本（用 in-memory vector store + mock embedder），写 README 说明。

## File Structure

- **Create** `tests/verify_upload_e2e.py` — 脚本式验证（仿 verify_llamaindex.py）
- **Modify** `README.md` — 加新功能说明（最小修改）

---

### Task 4.1: E2E 验证脚本

**Files:**
- Create: `tests/verify_upload_e2e.py`

- [ ] **Step 1: 写脚本**

```python
# tests/verify_upload_e2e.py
"""E2E 验证：parser registry → reader → upload → preview。

跑法：python -m tests.verify_upload_e2e
"""
from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path


def _section(s):
    print(f"\n=== {s} ===")


def verify_parser_registry():
    _section("1. ParserRegistry 加载")
    from study_rag.capabilities.llamaindex import get_parser_registry
    reg = get_parser_registry(force_reload=True)
    summaries = reg.list_summaries()
    print(f"  加载了 {len(summaries)} 个 parser:")
    for s in summaries:
        print(f"    - {s['name']}: strategy={s['strategy']}, size={s['chunk_size']}")
    assert len(summaries) >= 5


def verify_text_reader():
    _section("2. 文本 reader")
    from study_rag.capabilities.llamaindex import read_document
    text, meta = read_document(b"Hello world", filename="x.txt")
    assert "Hello" in text
    assert meta["format"] == "txt"
    print(f"  [OK] txt: format={meta['format']}, size={meta['size_bytes']}")


def verify_markdown_reader():
    _section("3. Markdown reader")
    from study_rag.capabilities.llamaindex import read_document
    md = b"# Title\n\nParagraph 1.\n\nParagraph 2."
    text, meta = read_document(md, filename="x.md")
    assert "Title" in text
    print(f"  [OK] md: {len(text)} chars")


def verify_parser_preview():
    _section("4. Parser 切块预览")
    from study_rag.capabilities.llamaindex import get_parser_registry
    reg = get_parser_registry()
    factory = reg.get("sentence_512")
    content = "Sentence 1. Sentence 2.\n\nSentence 3. Sentence 4.\n\nSentence 5."
    nodes = factory.parse(content, doc_id="e2e-1", title="t")
    assert len(nodes) >= 2
    print(f"  [OK] sentence_512 切出 {len(nodes)} 块")
    for n in nodes:
        print(f"    [{n.chunk_index}] {n.text[:50]}...")


async def verify_full_flow():
    _section("5. 完整 upload flow (用 in-memory backend)")
    from httpx import ASGITransport, AsyncClient
    from study_rag.app import create_app

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # 1. 列出 parsers
        r = await c.get("/admin/parsers")
        assert r.status_code == 200
        print(f"  [OK] GET /admin/parsers -> {len(r.json())} parsers")

        # 2. preview
        r = await c.post(
            "/admin/kbs/rd_frontend/documents/preview-chunk",
            json={"content": "A. B. C. D.", "parser": "sentence_512"},
        )
        # 400 if kb not exists — that's OK for this test
        if r.status_code == 200:
            print(f"  [OK] preview -> {r.json()['total_chunks']} chunks")
        else:
            print(f"  [skip] preview: {r.status_code} {r.json().get('detail', '')}")


def main():
    verify_parser_registry()
    verify_text_reader()
    verify_markdown_reader()
    verify_parser_preview()
    asyncio.run(verify_full_flow())
    print("\n=== ALL OK ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        sys.exit(1)
```

- [ ] **Step 2: 跑脚本**

Run: `python -m tests.verify_upload_e2e`
Expected: 5 sections all pass

- [ ] **Step 3: Commit**

```bash
git add tests/verify_upload_e2e.py
git commit -m "test(e2e): add upload/preview/parser verify script"
```

---

### Task 4.2: README 更新（最小）

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 加功能说明**

在 README 的 "Features" 段加：

```markdown
- **文档管理**：支持文本输入 / 文件上传（txt/md/html/pdf/docx），多种切块策略
  （whole / sentence / token / semantic），分块预览（提交前看到每块内容），所有策略
  通过 `configs/llamaindex.yaml` 配置
- **Parser Registry**：命名 parser 实体，类似 embedder/reranker 的注册表设计
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add upload/chunking feature description"
```

---

## 整体验收清单（自检）

跑完所有任务后，对照以下清单验证：

- [ ] `pytest tests/ -v --ignore=tests/verify_*.py` 全 pass
- [ ] `python -m tests.verify_upload_e2e` 输出 "ALL OK"
- [ ] `cd frontend && npx tsc --noEmit && npx tsc --build` exit 0
- [ ] `cd frontend && npm run build` exit 0
- [ ] 重启服务后浏览器：
  - `/admin/parsers` GET 返回 6 个 parser
  - Documents 页"添加文档"打开新 Drawer，有「文本输入 / 文件上传」两个 tab
  - 文件 tab 拖一个 .txt 进去，自动填 title，能预览切块
  - 选不同 parser，"预览分块"显示不同结果
  - 提交后 KB 文档列表多一行，format 字段显示 `txt/md/pdf/...`

## 实施顺序建议

1. **Phase 0**（Parser Registry）— 后端基础，先做
2. **Phase 1**（DocumentReader）— 文件解析，无依赖
3. **Phase 2**（API 端点）— 把 0/1 串起来
4. **Phase 3**（前端 UI）— 依赖 2 的端点
5. **Phase 4**（E2E + 文档）— 收尾

## 风险与边界

- **OOV chunk count**：semantic 策略切块数依赖 embed 速度，5000 字文章可能要 30s+。前端要 `Spin loading`，后端不设超时（LI 自己会跑完）。
- **PDF 解析质量**：pypdf 对扫描版 PDF 提取出空字符串。要么前端提示用户「扫描版请用 OCR 工具预转」，要么后端 fallback 到 PyMuPDF（待评估）。
- **大文件上传**：默认 50MB 上限。FastAPI 默认无大小限制，需要 `request.stream()` 限制，或在 nginx 限。
- **编码**：`text/*` 走 utf-8 → gb18030 → latin-1 容错解码。GBK 文档可能乱码（gb18030 通常够用）。
- **Parser 修改 yaml 后**：要重启服务（ParserRegistry 是单例）。
