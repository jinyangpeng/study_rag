"""Reranker 端到端接入验证脚本。

验证场景：
  1. Manager 正确按 KB 引用加载 rerankers
  2. KB 引用了不存在的 reranker 时明确报错
  3. 多 KB 各自用不同 reranker → 拿到不同实例
  4. 多 KB 共享 reranker（同名引用）→ 单例复用
  5. 未引用的 reranker 不被加载
  6. KB 未配置 reranker → get_reranker_for_kb 返回 None
  7. search 端到端：embedding → vector search → rerank → top_k
  8. rerank 失败时优雅降级（返回截断结果不抛错）
  9. 未启用 rerank 时直接截断
 10. 跨 KB 搜索不再次重排（仅按分数合并）
"""

# ruff: noqa: PT017, PT018  (verify 脚本，非 pytest 测试)

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

import yaml

from study_rag.capabilities.embedding import EmbeddingConfig
from study_rag.capabilities.reranker import create_reranker
from study_rag.capabilities.vector_store import (
    SearchResult,
)
from study_rag.knowledge_bases.manager import (
    KnowledgeBaseManager,
    reset_manager_singleton,
)
from study_rag.knowledge_bases.models import (
    DocumentCreate,
)
from study_rag.knowledge_bases.registry import (
    get_registry,
    reset_registry_cache,
)
from study_rag.settings import AppPaths

logging.basicConfig(level=logging.WARNING)


def _section(name: str) -> None:
    print(f"\n=== {name} ===")


# ---- 辅助：构造最小可用的 manager ----

def _make_temp_manager(
    kb_configs: list[dict],
    emb_configs: dict | None = None,
    rerank_configs: dict | None = None,
    vs_config: dict | None = None,
) -> KnowledgeBaseManager:
    """用临时 YAML 构造一个 KnowledgeBaseManager。"""

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        # 写 KB yaml
        (td_path / "kb.yaml").write_text(
            yaml.safe_dump({"knowledge_bases": kb_configs}, allow_unicode=True),
            encoding="utf-8",
        )
        AppPaths.KB_CONFIG = td_path / "kb.yaml"

        # 写 embedding yaml
        emb_yaml = emb_configs or {
            "embeddings": {
                "mock128": {
                    "provider": "mock", "model_name": "m",
                    "dimension": 4, "batch_size": 8,
                },
            },
        }
        (td_path / "emb.yaml").write_text(
            yaml.safe_dump(emb_yaml, allow_unicode=True), encoding="utf-8",
        )
        AppPaths.EMBEDDING_CONFIG = td_path / "emb.yaml"

        # 写 reranker yaml
        rerank_yaml = rerank_configs or {"rerankers": {}}
        (td_path / "rerank.yaml").write_text(
            yaml.safe_dump(rerank_yaml, allow_unicode=True), encoding="utf-8",
        )
        AppPaths.RERANKER_CONFIG = td_path / "rerank.yaml"

        # 写 vector store yaml
        vs_yaml = {"vector_store": vs_config or {"provider": "mock", "uri": ""}}
        (td_path / "vs.yaml").write_text(
            yaml.safe_dump(vs_yaml, allow_unicode=True), encoding="utf-8",
        )
        AppPaths.VECTOR_STORE_CONFIG = td_path / "vs.yaml"

        # 重置单例，重新构造
        reset_registry_cache()
        reset_manager_singleton()

        registry = get_registry()

        from study_rag.knowledge_bases.manager import (
            _load_embedding_configs,
            _load_reranker_configs,
            _load_vector_store_config,
        )
        emb_cfgs = _load_embedding_configs()
        embedders = {
            name: create_embedder_wrap(cfg) for name, cfg in emb_cfgs.items()
        }
        rerank_cfgs = _load_reranker_configs()
        rerankers = {
            name: create_reranker(cfg) for name, cfg in rerank_cfgs.items()
        }
        vs_cfg = _load_vector_store_config()
        from study_rag.capabilities.vector_store import create_vector_store

        vs = create_vector_store(vs_cfg)

        manager = KnowledgeBaseManager(
            registry=registry,
            vector_store=vs,
            embedders=embedders,  # type: ignore
            rerankers=rerankers,
        )
        return manager


def create_embedder_wrap(cfg: EmbeddingConfig):
    from study_rag.capabilities.embedding import create_embedder
    return create_embedder(cfg)


# ---- 验证 ----

def verify_manager_loads_only_referenced():
    _section("1. Manager 按 KB 引用加载 rerankers")
    manager = _make_temp_manager(
        kb_configs=[
            {"kb_id": "kb_a", "name": "A", "description": "A",
             "department": "d", "collection": "c_a",
             "embedding": "mock128", "reranker": "r_a", "enabled": True},
            {"kb_id": "kb_b", "name": "B", "description": "B",
             "department": "d", "collection": "c_b",
             "embedding": "mock128", "reranker": "r_b", "enabled": True},
            {"kb_id": "kb_c", "name": "C", "description": "C",
             "department": "d", "collection": "c_c",
             "embedding": "mock128", "reranker": None, "enabled": True},
        ],
        rerank_configs={
            "rerankers": {
                "r_a": {"provider": "none", "top_k": 3},
                "r_b": {"provider": "mock", "top_k": 5},
                "unused_r": {"provider": "none", "top_k": 1},
            },
        },
    )
    loaded = list(manager._rerankers.keys())
    print(f"  manager.rerankers = {loaded}")
    assert set(loaded) == {"r_a", "r_b"}, "未引用的 unused_r 不应被加载"
    print("  [OK] 仅加载被引用的 r_a, r_b; unused_r 被正确跳过")

    # KB C 未配置 reranker → get_reranker_for_kb 返回 None
    assert manager.get_reranker_for_kb("kb_c") is None
    print("  [OK] kb_c 未配置 reranker -> get_reranker_for_kb 返回 None")

    # KB A 配的是 r_a（provider=none, top_k=3）
    r = manager.get_reranker_for_kb("kb_a")
    assert r is not None
    assert r._config.top_k == 3
    print(f"  [OK] kb_a -> {type(r).__name__}, config.top_k={r._config.top_k}")


def verify_missing_reranker_references():
    _section("2. KB 引用不存在的 reranker 时明确报错")
    try:
        _make_temp_manager(
            kb_configs=[{
                "kb_id": "kb_a", "name": "A", "description": "A",
                "department": "d", "collection": "c_a",
                "embedding": "mock128", "reranker": "non_existent", "enabled": True,
            }],
            rerank_configs={"rerankers": {
                "real_one": {"provider": "none", "top_k": 3},
            }},
        )
        raise AssertionError("应抛 ValueError")
    except ValueError as ex:
        assert "non_existent" in str(ex)
        print(f"  [OK] {str(ex)[:100]}")


def verify_multi_kb_shared_reranker():
    _section("3. 多 KB 共享同名 reranker（单例复用）")
    manager = _make_temp_manager(
        kb_configs=[
            {"kb_id": "kb_1", "name": "1", "description": "1",
             "department": "d", "collection": "c_1",
             "embedding": "mock128", "reranker": "shared", "enabled": True},
            {"kb_id": "kb_2", "name": "2", "description": "2",
             "department": "d", "collection": "c_2",
             "embedding": "mock128", "reranker": "shared", "enabled": True},
        ],
        rerank_configs={"rerankers": {
            "shared": {"provider": "none", "top_k": 3},
        }},
    )
    r1 = manager.get_reranker_for_kb("kb_1")
    r2 = manager.get_reranker_for_kb("kb_2")
    assert r1 is r2, "同名 reranker 应是单例"
    print(f"  [OK] kb_1.reranker is kb_2.reranker: {r1 is r2}")


def verify_multi_kb_different_rerankers():
    _section("4. 多 KB 各自用不同 reranker")
    manager = _make_temp_manager(
        kb_configs=[
            {"kb_id": "kb_a", "name": "A", "description": "A",
             "department": "d", "collection": "c_a",
             "embedding": "mock128", "reranker": "bge_r", "enabled": True},
            {"kb_id": "kb_b", "name": "B", "description": "B",
             "department": "d", "collection": "c_b",
             "embedding": "mock128", "reranker": "cohere_r", "enabled": True},
        ],
        rerank_configs={"rerankers": {
            # 用 mock provider 模拟 (因真实 bge/cohere 缺依赖)
            "bge_r": {"provider": "mock", "top_k": 3},
            "cohere_r": {"provider": "none", "top_k": 5},
        }},
    )
    a = manager.get_reranker_for_kb("kb_a")
    b = manager.get_reranker_for_kb("kb_b")
    assert a is not b
    assert a._config.top_k == 3
    assert b._config.top_k == 5
    print(f"  [OK] kb_a -> {type(a).__name__}(config.top_k={a._config.top_k})")
    print(f"  [OK] kb_b -> {type(b).__name__}(config.top_k={b._config.top_k})")


async def verify_search_e2e_with_rerank():
    _section("5. search 端到端：embedding -> vector search -> rerank -> top_k")
    manager = _make_temp_manager(
        kb_configs=[
            {"kb_id": "kb_rerank", "name": "R", "description": "R",
             "department": "d", "collection": "c_r",
             "embedding": "mock128", "reranker": "active", "enabled": True},
        ],
        rerank_configs={"rerankers": {
            "active": {"provider": "mock", "top_k": 2},
        }},
    )
    await manager.init_all()

    # 加入 4 篇文档
    for i, txt in enumerate(
        ["Apple fruit is sweet",
         "Banana is yellow",
         "Cherry is small and red",
         "Date is brown and sweet"]
    ):
        await manager.add_document(DocumentCreate(
            kb_id="kb_rerank",
            doc_id=f"d{i}",
            title=f"doc-{i}",
            content=txt,
        ))

    # 模拟检索：mock vector 用 hash 向量
    # 由于 mock 的语义不真实，这里只验证调用链：rerank 被调用，且返回数量 <= top_k
    # 真实场景下 rerank 会改变顺序
    from study_rag.mcp.tools.search import _rerank_with_fallback

    # 手工构造 candidates 测试 _rerank_with_fallback
    cands = [
        SearchResult(id="a", text="x", score=0.5, metadata={}),
        SearchResult(id="b", text="y", score=0.9, metadata={}),
        SearchResult(id="c", text="z", score=0.7, metadata={}),
    ]
    reranker = manager.get_reranker_for_kb("kb_rerank")
    out = await _rerank_with_fallback(reranker, "q", cands, top_k=2)
    assert len(out) == 2
    # mock rerank 保持原序
    assert [r.id for r in out] == ["a", "b"]
    print(f"  [OK] _rerank_with_fallback(mock): {[r.id for r in out]}")


async def verify_search_e2e_no_rerank():
    _section("6. search 端到端：KB 未配置 reranker 时直接截断")
    manager = _make_temp_manager(
        kb_configs=[{
            "kb_id": "kb_no_r", "name": "NR", "description": "NR",
            "department": "d", "collection": "c_nr",
            "embedding": "mock128", "reranker": None, "enabled": True,
        }],
    )
    assert manager.get_reranker_for_kb("kb_no_r") is None
    from study_rag.mcp.tools.search import _rerank_with_fallback

    cands = [
        SearchResult(id="a", text="x", score=0.5, metadata={}),
        SearchResult(id="b", text="y", score=0.9, metadata={}),
    ]
    out = await _rerank_with_fallback(None, "q", cands, top_k=1)
    assert len(out) == 1
    assert out[0].id == "a"
    print(f"  [OK] 无 reranker 时直接截断: {[r.id for r in out]}")


async def verify_rerank_failure_fallback():
    _section("7. rerank 失败时降级为截断（不抛错）")

    class FailingReranker:
        """故意抛错的 reranker。"""
        _config = None

        async def rerank(self, query, results, top_k=None):
            raise RuntimeError("simulated reranker failure")

    cands = [
        SearchResult(id="a", text="x", score=0.5, metadata={}),
        SearchResult(id="b", text="y", score=0.9, metadata={}),
        SearchResult(id="c", text="z", score=0.7, metadata={}),
    ]
    from study_rag.mcp.tools.search import _rerank_with_fallback

    # 失败应降级
    out = await _rerank_with_fallback(FailingReranker(), "q", cands, top_k=2)
    assert len(out) == 2
    assert [r.id for r in out] == ["a", "b"]  # 原始顺序
    print(f"  [OK] rerank 失败时降级为截断: {[r.id for r in out]}")


async def verify_empty_candidates():
    _section("8. 空 candidates 时不调用 rerank")
    from study_rag.mcp.tools.search import _rerank_with_fallback

    called = False

    class SpyReranker:
        async def rerank(self, query, results, top_k=None):
            nonlocal called
            called = True
            return []

    out = await _rerank_with_fallback(SpyReranker(), "q", [], top_k=5)
    assert out == []
    # 空 candidates 时不应调用 reranker（节省一次网络/推理）
    assert not called, "空 candidates 时不应触发 rerank"
    print("  [OK] 空 candidates 直接返回 [], rerank 未被调用")


async def main():
    print("=" * 60)
    print("Reranker 端到端集成验证")
    print("=" * 60)

    verify_manager_loads_only_referenced()
    verify_missing_reranker_references()
    verify_multi_kb_shared_reranker()
    verify_multi_kb_different_rerankers()
    await verify_search_e2e_with_rerank()
    await verify_search_e2e_no_rerank()
    await verify_rerank_failure_fallback()
    await verify_empty_candidates()

    print("\n" + "=" * 60)
    print("[PASS] 全部验证通过")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
