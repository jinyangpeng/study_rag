"""MCP 错误定义。"""

from __future__ import annotations


class MCPError(Exception):
    """MCP Tool 通用错误基类。"""

    code: str = "internal_error"

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code


class KBNotFoundError(MCPError):
    code = "kb_not_found"


class DocumentNotFoundError(MCPError):
    code = "document_not_found"


class InvalidParameterError(MCPError):
    code = "invalid_parameter"


class DocumentAlreadyExistsError(MCPError):
    code = "document_already_exists"
