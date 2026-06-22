"""鉴权层（占位实现）。

当前版本不做实际鉴权，仅做结构占位，方便后续接入。

设计：
  - 所有 Tool 都接受 `api_key` 参数
  - PermissionResolver.resolve(api_key) -> UserContext
  - 当前实现：永远返回包含所有 KB 的用户
  - 后续接入：JWT / 静态配置 / 第三方 IDP
"""
