"""权限解析（占位实现）。

后续接入：
  - JWT 解析
  - 静态配置（user → KB 列表）
  - 第三方 IDP（OIDC / OAuth2）
  - 单独的权限服务

当前占位实现的行为：
  - api_key 为 None/空串：返回「匿名用户」（占位 = 拥有所有 KB 的读写权限）
  - api_key 非空：返回「命名用户」（user_id = api_key，同样拥有所有 KB）
  - 真实接入鉴权后，把这里替换为 JWT 解析 / 配置查表 / OIDC 远程调用

是否强制要求 api_key 由 ServerSettings.mcp_require_api_key 控制：
  - 默认 false：允许匿名（api_key 可空，MCP Inspector 不填也能调）
  - 设为 true：空 api_key → 抛 PermissionDenied
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..knowledge_bases.registry import get_registry
from ..settings import get_server_settings


@dataclass
class UserContext:
    """用户上下文。"""

    user_id: str
    display_name: str = ""
    accessible_kbs: list[str] = field(default_factory=list)
    # 可写权限 KB 列表（管理类操作需要此权限）
    # 默认与 accessible_kbs 一致；真实接入鉴权后按 role / scope 区分
    writable_kbs: list[str] = field(default_factory=list)
    # 后续扩展：scopes, roles, tenant_id 等

    def can_write(self, kb_id: str) -> bool:
        return kb_id in self.writable_kbs


class PermissionDenied(Exception):  # noqa: N818
    """权限不足。"""


class PermissionResolver:
    """权限解析器。"""

    async def resolve(self, api_key: str | None) -> UserContext:
        """根据 api_key 解析用户上下文。

        参数:
          api_key: 用户凭证；为 None 或空串时按匿名用户处理（除非
            ServerSettings.mcp_require_api_key=True 强制要求非空）。

        行为:
          - 空 api_key + 不强制：匿名用户，user_id='anonymous'
          - 空 api_key + 强制：抛 PermissionDenied
          - 非空 api_key：命名用户，user_id=api_key（占位实现；真实接入后这里查表/JWT）

        返回:
          UserContext（占位实现下永远拥有所有 KB 的读写权限）
        """
        # 配置驱动的强制校验
        require = get_server_settings().mcp_require_api_key
        if not api_key or not api_key.strip():
            if require:
                raise PermissionDenied(
                    "api_key is required (set STUDY_RAG_MCP_REQUIRE_API_KEY=false to disable)"
                )
            user_id = "anonymous"
            display_name = "Anonymous User (auth not enforced)"
        else:
            user_id = api_key
            display_name = f"User {api_key} (auth placeholder)"

        registry = get_registry()
        all_kb_ids = registry.list_ids()
        return UserContext(
            user_id=user_id,
            display_name=display_name,
            accessible_kbs=list(all_kb_ids),
            writable_kbs=list(all_kb_ids),
        )

    def check_kb_access(self, user: UserContext, kb_id: str) -> None:
        """检查用户是否有权访问指定 KB，无权则抛异常。"""
        if kb_id not in user.accessible_kbs:
            raise PermissionDenied(
                f"User '{user.user_id}' has no access to kb '{kb_id}'"
            )

    def check_kb_write_access(self, user: UserContext, kb_id: str) -> None:
        """检查用户是否有写入（管理）权限，无权则抛异常。"""
        if not user.can_write(kb_id):
            raise PermissionDenied(
                f"User '{user.user_id}' has no write access to kb '{kb_id}'"
            )


# 单例
_default_resolver: PermissionResolver | None = None


def get_permission_resolver() -> PermissionResolver:
    """获取全局权限解析器。"""
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = PermissionResolver()
    return _default_resolver
