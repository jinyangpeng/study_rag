"""Provider 中立的 metadata filter 表达 + 转换器。

设计目标：
  - 用 dict 作为"中性表达"（输入）
  - mock provider 直接用 Python 表达式求值
  - milvus provider 转为 Milvus expr 字符串
  - 后续可扩展 qdrant / chroma 等

支持的 key 后缀操作符（与 Django ORM 风格一致）：
  - `key`             : == value
  - `key__eq`         : == value
  - `key__ne`         : != value
  - `key__in`         : field in [v1, v2, ...]
  - `key__nin`        : field not in [...]
  - `key__gt`         : > value
  - `key__gte`        : >= value
  - `key__lt`         : < value
  - `key__lte`        : <= value
  - `key__contains`   : substring (str only)
  - `key__exists`     : key in metadata (bool)

Example:
    {"department": "rd", "year__gte": 2024, "tag__in": ["api", "db"]}
"""

from __future__ import annotations

import re
from typing import Any

# 字段名只能含字母/数字/下划线（防止 Milvus expr 注入）
_FIELD_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# 支持的操作符集合
_OPS_EQ = {"eq"}
_OPS_NE = {"ne"}
_OPS_IN = {"in", "nin"}  # 包含 / 不包含
_OPS_CMP = {"gt", "gte", "lt", "lte"}
_OPS_STR = {"contains"}
_OPS_BOOL = {"exists"}

# Milvus collection schema 中存于 JSON `metadata` 字段里的子字段白名单。
# 任何命中此集合的 filter 字段会被翻译为 metadata["<field>"] JSON path 语法。
# 顶层（非 metadata）字段保持原样引用，保留向后兼容。
_METADATA_FIELDS: frozenset[str] = frozenset({
    "doc_id",
    "chunk_index",
    "title",
    "source",
    "parser",
    "filename",
    "uploaded",
    "author",
    "tags",
    "url",
    "year",
    # Phase 6.7: LI adapter 写入 chunks 时把 ref_doc_id 存到 metadata["ref_doc_id"]。
    # 删除时按 ref_doc_id 过滤；不加入此白名单会生成顶层 `ref_doc_id == X`，
    # Milvus 顶层无此字段，会报 "field not exist"。
    "ref_doc_id",
})


def _to_field_ref(field: str) -> str:
    """把 field 翻译为 Milvus expr 字段引用。

    - 在 ``_METADATA_FIELDS`` 白名单中的字段生成 JSON path: ``metadata["<field>"]``
    - 其它字段保持顶层引用（向后兼容：例如 ``department``）
    """
    if field in _METADATA_FIELDS:
        return f'metadata["{field}"]'
    return field


def parse_key(raw_key: str) -> tuple[str, str]:
    """从 "key__op" 拆出 (field, op)；无后缀时 op="eq"。

    Raises:
        ValueError: 字段名不合法（防止注入 / 拼写错误）
    """
    if "__" in raw_key:
        field, op = raw_key.rsplit("__", 1)
    else:
        field, op = raw_key, "eq"

    if not _FIELD_RE.match(field):
        raise ValueError(f"Invalid filter field name: {field!r}")

    all_ops = _OPS_EQ | _OPS_NE | _OPS_IN | _OPS_CMP | _OPS_STR | _OPS_BOOL
    if op not in all_ops:
        raise ValueError(
            f"Unknown filter operator: {op!r}. "
            f"Supported: eq, ne, in, nin, gt, gte, lt, lte, contains, exists"
        )
    return field, op


def format_value(value: Any) -> str:
    """把 Python 值格式化为 Milvus expr 字面量。

    - str  -> "value"（转义内部双引号）
    - bool -> true / false
    - None -> null
    - 其它 -> str(value)（数字等）
    """
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def to_milvus_expr(filter_expr: dict | None) -> str | None:
    """把中性 dict 转成 Milvus expr 字符串。

    - 空/None -> None
    - 多个条件用 and 连接
    - 在 ``_METADATA_FIELDS`` 白名单中的字段自动翻译为 ``metadata["<field>"]``
      JSON path 语法，匹配 Milvus collection schema 中 JSON metadata 字段。
    - 全部 op 在 parse_key 中已校验
    """
    if not filter_expr:
        return None

    parts: list[str] = []
    for raw_key, value in filter_expr.items():
        field, op = parse_key(raw_key)
        ref = _to_field_ref(field)

        if op in _OPS_EQ:
            parts.append(f"{ref} == {format_value(value)}")
        elif op in _OPS_NE:
            parts.append(f"{ref} != {format_value(value)}")
        elif op == "in":
            if not isinstance(value, (list, tuple)):
                raise ValueError(
                    f"__in operator requires list value, got {type(value).__name__}"
                )
            formatted = ", ".join(format_value(v) for v in value)
            parts.append(f"{ref} in [{formatted}]")
        elif op == "nin":
            if not isinstance(value, (list, tuple)):
                raise ValueError(
                    f"__nin operator requires list value, got {type(value).__name__}"
                )
            formatted = ", ".join(format_value(v) for v in value)
            parts.append(f"not ({ref} in [{formatted}])")
        elif op in _OPS_CMP:
            sym = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}[op]
            parts.append(f"{ref} {sym} {format_value(value)}")
        elif op == "contains":
            if not isinstance(value, str):
                raise ValueError(
                    f"__contains operator requires str value, got {type(value).__name__}"
                )
            parts.append(f'{ref} like "%{value}%"')
        elif op == "exists":
            if not isinstance(value, bool):
                raise ValueError(
                    f"__exists operator requires bool value, got {type(value).__name__}"
                )
            # Milvus JSON 字段的 exists 等价语义：``ref is not null`` (或 ``is null``)
            parts.append(f"{ref} is null" if not value else f"{ref} is not null")
        else:  # pragma: no cover - parse_key 已校验
            raise ValueError(f"Unhandled operator: {op}")

    return " and ".join(parts) if parts else None


def matches_filter(metadata: dict, filter_expr: dict | None) -> bool:
    """用中性 filter 表达判断 metadata 是否命中。

    供 mock / in-memory vector store / 任何想直接应用 filter 的场景使用。
    单一条件 -> 直接判断；多条件 -> 全部 AND。
    """
    if not filter_expr:
        return True

    md = metadata or {}
    for raw_key, expected in filter_expr.items():
        field, op = parse_key(raw_key)
        actual = md.get(field)

        if op == "eq":
            if actual != expected:
                return False
        elif op == "ne":
            if actual == expected:
                return False
        elif op == "in":
            if not isinstance(expected, (list, tuple)):
                raise ValueError("__in requires list value")
            if actual not in expected:
                return False
        elif op == "nin":
            if not isinstance(expected, (list, tuple)):
                raise ValueError("__nin requires list value")
            if actual in expected:
                return False
        elif op == "gt":
            if not (actual is not None and actual > expected):
                return False
        elif op == "gte":
            if not (actual is not None and actual >= expected):
                return False
        elif op == "lt":
            if not (actual is not None and actual < expected):
                return False
        elif op == "lte":
            if not (actual is not None and actual <= expected):
                return False
        elif op == "contains":
            if not isinstance(expected, str):
                raise ValueError("__contains requires str value")
            if not (isinstance(actual, str) and expected in actual):
                return False
        elif op == "exists":
            if not isinstance(expected, bool):
                raise ValueError("__exists requires bool value")
            has_key = field in md
            if has_key != expected:
                return False
        else:  # pragma: no cover - parse_key 已校验
            raise ValueError(f"Unhandled operator: {op}")
    return True
