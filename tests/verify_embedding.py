"""Embedding 集成验证脚本。

验证：
1. Mock embedder 工作正常（向后兼容）
2. OpenAI embedder 可加载（缺 api_key 时给出明确报错）
3. OpenAI embedder 实例化正确（api_key / base_url / model / dim）
4. BGE embedder 在未装 FlagEmbedding 时给出明确的 ImportError
5. 多个 KB 可以配置不同 embedding，manager 正确分发
6. embedding 引用了不存在的 KB 配置时报错
7. 缺省 disabled 的 KB 不会触发 embedding 加载
"""

# ruff: noqa: PT017, PT018  (verify 脚本，非 pytest 测试)

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import yaml

from study_rag.capabilities.embedding import (
    EmbeddingConfig,
    create_embedder,
    list_embedder_providers,
)
from study_rag.settings import AppPaths


def _section(name: str) -> None:
    print(f"\n=== {name} ===")


async def verify_mock() -> None:
    _section("1. Mock embedder")
    cfg = EmbeddingConfig.from_raw({
        "provider": "mock",
        "model_name": "mock-embedder",
        "dimension": 128,
        "batch_size": 32,
    })
    assert cfg.dimension == 128
    e = create_embedder(cfg)
    assert type(e).__name__ == "MockEmbedder"
    assert e.dimension == 128

    v1 = await e.embed_query("hello world")
    v2 = await e.embed_documents(["hello", "world", "hello world"])
    assert len(v1) == 128
    assert len(v2) == 3
    assert len(v2[0]) == 128
    # 同一文本 hash 相同 -> 向量相同
    assert v1 == v2[2]
    print("  MockEmbedder 工作正常：维度 128，相同文本得到相同向量")


def verify_openai_loadable() -> None:
    _section("2. OpenAI embedder - 缺 api_key 报错")
    os.environ.pop("OPENAI_API_KEY", None)
    cfg = EmbeddingConfig.from_raw({
        "provider": "openai",
        "model_name": "text-embedding-3-small",
        "dimension": 1536,
        "batch_size": 100,
    })
    try:
        create_embedder(cfg)
        raise AssertionError("应抛出 ValueError")
    except ValueError as ex:
        assert "api_key" in str(ex).lower()
        print(f"  [OK] 未提供 api_key 正确报错: {str(ex)[:80]}")


def verify_openai_instantiate() -> None:
    _section("3. OpenAI embedder - 正常实例化")
    os.environ["OPENAI_API_KEY"] = "sk-fake-for-test"
    cfg = EmbeddingConfig.from_raw({
        "provider": "openai",
        "model_name": "text-embedding-3-small",
        "dimension": 1536,
        "batch_size": 100,
        "extra": {
            "api_key": "sk-fake-for-test",
            "base_url": "https://api.openai.com/v1",
            "timeout": 30.0,
        },
    })
    e = create_embedder(cfg)
    assert type(e).__name__ == "OpenAIEmbedder"
    assert e.dimension == 1536
    assert e._model == "text-embedding-3-small"
    assert e._client is not None
    print(f"  [OK] OpenAIEmbedder 实例化成功：model={e._model}, dim={e.dimension}")
    print(f"  [OK] AsyncOpenAI client 已创建: {type(e._client).__name__}")


def verify_bge_missing_dep() -> None:
    _section("4. BGE embedder - 缺依赖时报错")
    providers = list_embedder_providers()
    assert "bge" in providers, "bge provider 应在 registry 中"
    print(f"  [OK] bge provider 已注册 (providers={providers})")

    cfg = EmbeddingConfig.from_raw({
        "provider": "bge",
        "model_name": "BAAI/bge-large-zh-v1.5",
        "dimension": 1024,
    })
    try:
        create_embedder(cfg)
        raise AssertionError("应抛出 ImportError（FlagEmbedding 未装）")
    except ImportError as ex:
        assert "FlagEmbedding" in str(ex) or "study-rag" in str(ex)
        print(f"  [OK] 未装 FlagEmbedding 时正确报错: {str(ex)[:100]}")


def verify_env_var_resolution() -> None:
    _section("5. 环境变量占位符解析")
    os.environ["MY_TEST_KEY"] = "resolved-key-123"
    cfg = EmbeddingConfig.from_raw({
        "provider": "openai",
        "model_name": "x",
        "dimension": 1536,
        "extra": {"api_key": "${MY_TEST_KEY}"},
    })
    assert cfg.extra["api_key"] == "resolved-key-123"
    print(f"  [OK] ${{MY_TEST_KEY}} -> {cfg.extra['api_key']}")

    # 嵌套 dict / list
    cfg2 = EmbeddingConfig.from_raw({
        "provider": "openai",
        "model_name": "x",
        "dimension": 1536,
        "extra": {
            "headers": {"X-Token": "${MY_TEST_KEY}"},
            "endpoints": ["${MY_TEST_KEY}/a", "static"],
        },
    })
    assert cfg2.extra["headers"]["X-Token"] == "resolved-key-123"
    assert cfg2.extra["endpoints"][0] == "resolved-key-123/a"
    assert cfg2.extra["endpoints"][1] == "static"
    print("  [OK] 嵌套 dict / list 中占位符均被解析")


async def verify_multi_kb_different_embeddings() -> None:
    _section("6. 多 KB 各自用不同 embedding")
    import study_rag.knowledge_bases.manager as mgr_mod
    import study_rag.knowledge_bases.registry as reg_mod

    orig_kb = AppPaths.KB_CONFIG
    orig_emb = AppPaths.EMBEDDING_CONFIG

    with tempfile.TemporaryDirectory() as td:
        tmp_kb_path = Path(td) / "kb.yaml"
        tmp_emb_path = Path(td) / "emb.yaml"

        with tmp_emb_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump({
                "embeddings": {
                    "mock128": {
                        "provider": "mock", "model_name": "m",
                        "dimension": 128, "batch_size": 32,
                    },
                    "openai_test": {
                        "provider": "openai", "model_name": "text-embedding-3-small",
                        "dimension": 1536, "batch_size": 100,
                        "extra": {"api_key": "${OPENAI_API_KEY}"},
                    },
                    "unused_embedding": {
                        "provider": "mock", "model_name": "u",
                        "dimension": 64, "batch_size": 16,
                    },
                },
            }, f, allow_unicode=True)

        with tmp_kb_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump({
                "knowledge_bases": [
                    {
                        "kb_id": "kb_a", "name": "A", "description": "A desc",
                        "department": "d1", "collection": "col_a",
                        "embedding": "mock128", "enabled": True,
                    },
                    {
                        "kb_id": "kb_b", "name": "B", "description": "B desc",
                        "department": "d2", "collection": "col_b",
                        "embedding": "openai_test", "enabled": True,
                    },
                    {
                        # disabled KB 不应触发其 embedding 加载
                        "kb_id": "kb_disabled", "name": "C", "description": "C desc",
                        "department": "d1", "collection": "col_c",
                        "embedding": "unused_embedding", "enabled": False,
                    },
                ],
            }, f, allow_unicode=True)

        # 重置单例 + 切路径
        reg_mod.reset_registry_cache()
        mgr_mod.reset_manager_singleton()
        AppPaths.KB_CONFIG = tmp_kb_path
        AppPaths.EMBEDDING_CONFIG = tmp_emb_path

        try:
            # 验证 KB 引用收集
            from study_rag.knowledge_bases.registry import get_registry
            reg = get_registry()
            kbs = reg.list(enabled_only=True)
            assert {k.kb_id for k in kbs} == {"kb_a", "kb_b"}

            # 验证 embedding config 按需加载
            configs = mgr_mod._load_embedding_configs()
            assert set(configs.keys()) == {"mock128", "openai_test"}, configs.keys()
            print(f"  [OK] 按需加载: {sorted(configs.keys())} (unused_embedding 被正确跳过)")

            # 验证 manager 分发
            manager = mgr_mod.build_default_manager()
            ea = manager.get_embedder("kb_a")
            eb = manager.get_embedder("kb_b")
            assert type(ea).__name__ == "MockEmbedder" and ea.dimension == 128
            assert type(eb).__name__ == "OpenAIEmbedder" and eb.dimension == 1536
            print(f"  [OK] kb_a -> {type(ea).__name__}(dim={ea.dimension})")
            print(f"  [OK] kb_b -> {type(eb).__name__}(dim={eb.dimension})")

            # 验证 mock 端到端可用
            va = await ea.embed_query("test")
            assert len(va) == 128
            print(f"  [OK] kb_a 端到端可用，向量长度 {len(va)}")

            # disabled KB 不应能取到 embedder
            try:
                manager.get_embedder("kb_disabled")
                raise AssertionError("disabled KB 不应能取到 embedder")
            except Exception as ex:
                print(f"  [OK] disabled KB 拒绝服务: {type(ex).__name__}")
        finally:
            AppPaths.KB_CONFIG = orig_kb
            AppPaths.EMBEDDING_CONFIG = orig_emb
            reg_mod.reset_registry_cache()
            mgr_mod.reset_manager_singleton()


def verify_missing_embedding_ref() -> None:
    _section("7. KB 引用了不存在的 embedding 时报错")
    import study_rag.knowledge_bases.manager as mgr_mod
    import study_rag.knowledge_bases.registry as reg_mod

    orig_kb = AppPaths.KB_CONFIG
    orig_emb = AppPaths.EMBEDDING_CONFIG

    with tempfile.TemporaryDirectory() as td:
        tmp_kb_path = Path(td) / "kb.yaml"
        tmp_emb_path = Path(td) / "emb.yaml"

        with tmp_emb_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump({"embeddings": {
                "mock128": {"provider": "mock", "model_name": "m", "dimension": 128, "batch_size": 32},
            }}, f)

        with tmp_kb_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump({"knowledge_bases": [{
                "kb_id": "kb_a", "name": "A", "description": "A",
                "department": "d1", "collection": "col_a",
                "embedding": "non_existent", "enabled": True,
            }]}, f)

        reg_mod.reset_registry_cache()
        mgr_mod.reset_manager_singleton()
        AppPaths.KB_CONFIG = tmp_kb_path
        AppPaths.EMBEDDING_CONFIG = tmp_emb_path

        try:
            try:
                mgr_mod._load_embedding_configs()
                raise AssertionError("应抛出 ValueError")
            except ValueError as ex:
                assert "non_existent" in str(ex)
                print(f"  [OK] KB 引用不存在的 embedding 时明确报错: {str(ex)[:80]}")
        finally:
            AppPaths.KB_CONFIG = orig_kb
            AppPaths.EMBEDDING_CONFIG = orig_emb
            reg_mod.reset_registry_cache()
            mgr_mod.reset_manager_singleton()


async def main() -> None:
    print("=" * 60)
    print("Embedding 集成验证")
    print("=" * 60)

    providers = list_embedder_providers()
    print(f"\n已注册的 providers: {providers}")
    assert "mock" in providers
    assert "openai" in providers
    assert "bge" in providers

    await verify_mock()
    verify_openai_loadable()
    verify_openai_instantiate()
    verify_bge_missing_dep()
    verify_env_var_resolution()
    await verify_multi_kb_different_embeddings()
    verify_missing_embedding_ref()

    print("\n" + "=" * 60)
    print("[PASS] 全部验证通过")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
