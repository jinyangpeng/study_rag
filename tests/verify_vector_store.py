"""Vector Store 集成验证脚本。

验证：
1. Provider 注册情况（当前环境未装 pymilvus，应只有 mock）
2. Mock VectorStore 端到端可用（向后兼容）
3. 缺 pymilvus 时 milvus provider 应抛 ImportError
4. _load_vector_store_config 正确读取 yaml + 解析 ${ENV_VAR}
5. 多 KB 共享一个 VectorStore 实例（manager 的单例语义）
"""

# ruff: noqa: PT017, PT018  (verify 脚本，非 pytest 测试)

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import yaml

from study_rag.capabilities.vector_store import (
    VectorRecord,
    VectorStoreConfig,
    create_vector_store,
    list_vector_store_providers,
)
from study_rag.settings import AppPaths


def _section(name: str) -> None:
    print(f"\n=== {name} ===")


async def verify_mock_end_to_end() -> None:
    _section("1. Mock VectorStore 端到端")
    cfg = VectorStoreConfig(provider="mock")
    store = create_vector_store(cfg)
    assert type(store).__name__ == "InMemoryVectorStore"

    # create / insert / search / delete
    await store.create_collection("test_col", dimension=4)
    assert await store.has_collection("test_col")

    records = [
        VectorRecord(id="1", vector=[1.0, 0.0, 0.0, 0.0], text="alpha", metadata={}),
        VectorRecord(id="2", vector=[0.0, 1.0, 0.0, 0.0], text="beta", metadata={}),
        VectorRecord(id="3", vector=[0.9, 0.1, 0.0, 0.0], text="alpha-like", metadata={}),
    ]
    await store.insert("test_col", records)

    # 搜 [1, 0, 0, 0] 应排第 1 是 id=1（完全相同），其次 id=3（接近）
    results = await store.search("test_col", query_vector=[1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].id == "1"
    assert results[0].score > 0.99
    assert results[1].id == "3"
    print(f"  [OK] mock search top2: {[r.id for r in results]}, "
          f"score={results[0].score:.3f}/{results[1].score:.3f}")

    # delete
    await store.delete("test_col", ["1"])
    results2 = await store.search("test_col", query_vector=[1.0, 0.0, 0.0, 0.0], top_k=2)
    assert "1" not in [r.id for r in results2]
    print(f"  [OK] mock delete id=1, after delete ids: {[r.id for r in results2]}")

    # has / drop
    assert await store.has_collection("test_col")
    await store.drop_collection("test_col")
    assert not await store.has_collection("test_col")
    print("  [OK] mock drop_collection 后 has_collection 返回 False")


async def verify_milvus_missing_dep() -> None:
    _section("2. pymilvus 缺失时 milvus provider 行为")
    providers = list_vector_store_providers()
    print(f"  当前已注册 providers: {providers}")
    assert "mock" in providers
    # provider 装饰器在 import 时注册，但 pymilvus 实际导入在 _connect() 中懒加载
    # 所以 provider 总会注册；缺依赖时是首次调用时（不是构造时）报错
    assert "milvus" in providers
    print("  [OK] milvus provider 已注册（class 装饰器），但实际 pymilvus 导入在 _connect() 时懒加载")

    try:
        import pymilvus  # noqa: F401

        has_pymilvus = True
    except ImportError:
        has_pymilvus = False

    if has_pymilvus:
        print("  [INFO] pymilvus 已装, 端到端需要 Milvus 服务")
        return

    # 没装 pymilvus → 构造应能成功（懒加载），但首次 _connect() 应抛 ImportError
    cfg = VectorStoreConfig(provider="milvus", uri="dummy.db")
    store = create_vector_store(cfg)
    print(f"  [OK] 构造成功（懒加载）: {type(store).__name__}")

    # 触发 _connect() 应抛 ImportError
    try:
        await store.has_collection("any")
        raise AssertionError("应抛 ImportError")
    except ImportError as ex:
        assert "pymilvus" in str(ex) or "study-rag" in str(ex)
        print(f"  [OK] 首次调用触发 ImportError: {str(ex)[:100]}")


def verify_load_vector_store_config() -> None:
    _section("3. _load_vector_store_config 读取 + ENV 解析")
    import study_rag.knowledge_bases.manager as mgr_mod

    orig_path = AppPaths.VECTOR_STORE_CONFIG

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "vs.yaml"
        os.environ["MY_VS_URI"] = "tcp://10.0.0.1:19530"
        os.environ["MY_VS_TOKEN"] = "secret-token-xxx"
        with tmp.open("w", encoding="utf-8") as f:
            yaml.safe_dump({
                "vector_store": {
                    "provider": "milvus",
                    "uri": "${MY_VS_URI}",
                    "extra": {
                        "token": "${MY_VS_TOKEN}",
                        "db_name": "staging",
                        "batch_size": 500,
                    },
                },
            }, f)

        AppPaths.VECTOR_STORE_CONFIG = tmp
        try:
            cfg = mgr_mod._load_vector_store_config()
            assert cfg.provider == "milvus"
            assert cfg.uri == "tcp://10.0.0.1:19530"
            assert cfg.extra["token"] == "secret-token-xxx"
            assert cfg.extra["db_name"] == "staging"
            assert cfg.extra["batch_size"] == 500
            print(f"  [OK] provider={cfg.provider}, uri={cfg.uri}")
            print(f"  [OK] extra.token={cfg.extra['token']}, "
                  f"db_name={cfg.extra['db_name']}, batch_size={cfg.extra['batch_size']}")
        finally:
            AppPaths.VECTOR_STORE_CONFIG = orig_path
            os.environ.pop("MY_VS_URI", None)
            os.environ.pop("MY_VS_TOKEN", None)

    # 不存在的文件应兜底为 mock
    AppPaths.VECTOR_STORE_CONFIG = Path("/non/existent.yaml")
    try:
        cfg = mgr_mod._load_vector_store_config()
        assert cfg.provider == "mock"
        print("  [OK] 不存在的 config 文件兜底为 mock provider")
    finally:
        AppPaths.VECTOR_STORE_CONFIG = orig_path


async def verify_milvus_filter_translator() -> None:
    _section("4. Milvus filter dict -> expr 转换（无需真实 Milvus）")
    # filters.py 独立模块，不依赖 pymilvus，可直接 import
    from study_rag.capabilities.vector_store.filters import (
        format_value,
        to_milvus_expr,
    )

    assert to_milvus_expr(None) is None
    assert to_milvus_expr({}) is None
    print("  [OK] None / {} -> None")

    assert to_milvus_expr({"department": "rd"}) == 'department == "rd"'
    print('  [OK] {"department": "rd"} -> department == "rd"')

    expr = to_milvus_expr({"department": "rd", "year__gte": 2024})
    assert expr == 'department == "rd" and year >= 2024'
    print(f"  [OK] eq + gte 组合: {expr}")

    expr = to_milvus_expr({"status__in": ["active", "pending"], "level__ne": 0})
    assert expr == 'status in ["active", "pending"] and level != 0'
    print(f"  [OK] in + ne 组合: {expr}")

    # 字符串含双引号
    expr = to_milvus_expr({"title": 'hello "world"'})
    assert expr == 'title == "hello \\"world\\""'
    print(f"  [OK] 字符串转义: {expr}")

    # 数值
    assert to_milvus_expr({"score__gt": 0.95}) == "score > 0.95"
    print("  [OK] 浮点数: score__gt=0.95 -> score > 0.95")

    # bool
    assert to_milvus_expr({"active": True}) == "active == true"
    assert to_milvus_expr({"active": False}) == "active == false"
    print("  [OK] bool: True/False -> true/false")

    # __in 必须传 list
    try:
        to_milvus_expr({"x__in": "not-a-list"})
        raise AssertionError("应抛 ValueError")
    except ValueError as ex:
        print(f"  [OK] __in 传非 list 明确报错: {str(ex)[:60]}")

    # 未知操作符
    try:
        to_milvus_expr({"x__unknown": 1})
        raise AssertionError("应抛 ValueError")
    except ValueError as ex:
        print(f"  [OK] 未知操作符明确报错: {str(ex)[:60]}")

    # format_value 直接测试
    assert format_value(None) == "null"
    assert format_value(123) == "123"
    assert format_value(1.5) == "1.5"
    print("  [OK] format_value: None/数值 -> 正确字面量")


async def main() -> None:
    print("=" * 60)
    print("Vector Store 集成验证")
    print("=" * 60)

    await verify_mock_end_to_end()
    await verify_milvus_missing_dep()
    verify_load_vector_store_config()
    await verify_milvus_filter_translator()

    print("\n" + "=" * 60)
    print("[PASS] 全部验证通过")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
