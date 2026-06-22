"""DocumentReader：把上传的文件 bytes 转成纯文本 + 元信息。

支持的格式（按 file extension / mime）：
  .txt / text/plain          → 纯文本（容错解码 utf-8 → utf-8-sig → gb18030 → latin-1）
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
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

from ...observability.logging import get_logger

logger = get_logger(__name__)

__all__ = ["DocumentReader", "UnsupportedFormatError", "read_document"]


class UnsupportedFormatError(Exception):
    """不支持的文件格式。"""


class DocumentReader:
    """文档读取器抽象基类。

    实际格式分发由模块级 `read_document` 函数 + `_READERS` 字典完成；
    此处保留类形式以便未来做配置化扩展（例如按 KB 配置不同的 reader 链）。
    """

    def read(
        self,
        data: bytes,
        filename: str | None = None,
        mime: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """读取文件 bytes，返回 (text, metadata)。"""
        return read_document(data, filename=filename, mime=mime)


# mime -> extension 映射（部分常见 mime）
_MIME_TO_EXT: dict[str, str] = {
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
        from bs4 import BeautifulSoup  # type: ignore[import-untyped]

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
        import docx2txt  # type: ignore[import-untyped]
    except ImportError as e:
        raise UnsupportedFormatError(
            "DOCX reading requires docx2txt. Install: pip install study-rag[reader-docx]"
        ) from e
    return docx2txt.process(BytesIO(data)) or ""


_READERS: dict[str, Callable[[bytes], str]] = {
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
