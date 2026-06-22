"""权限解析（占位实现）。

后续接入：
  - JWT 解析
  - 静态配置（user → KB 列表）
  - 第三方 IDP（OIDC / OAuth2）
  - 单独的权限服务
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..knowledge_bases.registry import get_registry


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

    async def resolve(self, api_key: str) -> UserContext:
        """根据 api_key 解析用户上下文。

        当前占位实现：返回包含所有 KB 的匿名用户。
        """
        registry = get_registry()
        all_kb_ids = registry.list_ids()
        return UserContext(
            user_id="anonymous",
            display_name="Anonymous User (auth not enabled)",
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
