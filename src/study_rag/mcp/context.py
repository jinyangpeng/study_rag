"""MCP Tool 共享上下文。

由 server.py 在 Tool 注册时注入，Tool 函数通过此上下文访问 manager/registry/auth 等。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..auth.permissions import PermissionResolver, get_permission_resolver
from ..knowledge_bases.manager import KnowledgeBaseManager, build_default_manager


@dataclass
class MCPContext:
    """MCP Tool 共享上下文。"""

    manager: KnowledgeBaseManager
    auth: PermissionResolver

    @classmethod
    def default(cls) -> MCPContext:
        """从默认 manager/auth 构建上下文（单例）。"""
        return cls(
            manager=build_default_manager(),
            auth=get_permission_resolver(),
        )
