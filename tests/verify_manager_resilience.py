"""容错回归测试：KB 引用的 embedder 不可用时，lifespan 不应崩。

场景：
  1. KB config 引用了一个未加载的 embedder
  2. init_kb 应该跳过该 KB 并 warning，不抛异常
  3. init_all 应该统计成功/跳过数量
  4. FastAPI lifespan 整体应正常完成

这是 Study 中典型的"部分依赖缺失"场景：装 partial extras（如只装 OpenAI
不装 BGE）时，不应让服务起不来。
"""

# ruff: noqa: T201, PT017
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    print("=" * 60)
    print("Verify: manager 容错（embedder 缺失时 skip）")
    print("=" * 60)

    import asyncio

    from study_rag.knowledge_bases.manager import KnowledgeBaseManager

    # ---- 准备 mock ----
    class MockEmbedder:
        dimension = 768

    class MockVectorStore:
        def __init__(self) -> None:
            self.created: list[tuple[str, int]] = []

        async def create_collection(self, name: str, dimension: int) -> None:
            self.created.append((name, dimension))

    class MockRegistry:
        def __init__(self) -> None:
            self.cfgs = [
                type("C", (), {"kb_id": "kb1", "embedding": "openai_small", "collection": "c1"}),
                type("C", (), {"kb_id": "kb2", "embedding": "bge_missing", "collection": "c2"}),
                type("C", (), {"kb_id": "kb3", "embedding": "openai_small", "collection": "c3"}),
            ]

        def list(self, enabled_only: bool = True) -> list:
            return self.cfgs

        def get_required(self, kb_id: str):
            for c in self.cfgs:
                if c.kb_id == kb_id:
                    return c
            raise KeyError(kb_id)

    # ---- 1. init_kb 在 embedder 缺失时 skip ----
    print("\n[1] init_kb 缺失 embedder → skip (无异常)")
    vs = MockVectorStore()
    mgr = KnowledgeBaseManager(
        registry=MockRegistry(),  # type: ignore[arg-type]
        vector_store=vs,  # type: ignore[arg-type]
        embedders={"openai_small": MockEmbedder()},  # bge_missing 不在
    )

    async def t1() -> None:
        # kb1 正常 → True
        result1 = await mgr.init_kb("kb1")
        assert result1 is True, f"expected True, got {result1}"
        assert len(vs.created) == 1, vs.created
        assert vs.created[0] == ("c1", 768)
        print("    PASS: kb1 正常创建 collection → True")

        # kb2 缺失 embedder → False（不抛异常）
        result2 = await mgr.init_kb("kb2")
        assert result2 is False, f"expected False, got {result2}"
        assert len(vs.created) == 1, f"kb2 不应创建 collection, got {vs.created}"
        print("    PASS: kb2 缺失 embedder → False (skip)")

    asyncio.run(t1())

    # ---- 2. init_all 统计成功/跳过 ----
    print("\n[2] init_all 统计 succeeded=2, skipped=0（无抛错）")
    vs2 = MockVectorStore()
    mgr2 = KnowledgeBaseManager(
        registry=MockRegistry(),  # type: ignore[arg-type]
        vector_store=vs2,  # type: ignore[arg-type]
        embedders={"openai_small": MockEmbedder()},
    )

    async def t2() -> None:
        await mgr2.init_all()
        # 2 个 collection 被创建（kb1 + kb3），kb2 被 skip
        assert len(vs2.created) == 2, f"expected 2, got {vs2.created}"
        names = sorted(c[0] for c in vs2.created)
        assert names == ["c1", "c3"], names
        print(f"    PASS: 创建了 {names}")

    asyncio.run(t2())

    # ---- 3. create_collection 抛错时，init_all 不应崩整体 ----
    print("\n[3] init_all 单个 KB 失败 → 不影响其他 KB")

    class FlakyVectorStore:
        def __init__(self) -> None:
            self.created: list[str] = []

        async def create_collection(self, name: str, dimension: int) -> None:
            if name == "c1":
                raise RuntimeError("simulated Qdrant connection error")
            self.created.append(name)

    mgr3 = KnowledgeBaseManager(
        registry=MockRegistry(),  # type: ignore[arg-type]
        vector_store=FlakyVectorStore(),  # type: ignore[arg-type]
        embedders={"openai_small": MockEmbedder()},
    )

    async def t3() -> None:
        # 不应抛异常
        await mgr3.init_all()
        vs = mgr3._vector_store  # type: ignore[attr-defined]
        assert vs.created == ["c3"], f"expected ['c3'], got {vs.created}"
        print(f"    PASS: c1 失败被捕获，c3 仍正常创建 → {vs.created}")

    asyncio.run(t3())

    # ---- 4. FastAPI lifespan 整体不崩 ----
    print("\n[4] FastAPI app lifespan 启动不抛异常")
    from fastapi.testclient import TestClient

    with patch.dict("os.environ", {"STUDY_RAG_ADMIN_TOKEN": ""}, clear=False):
        from study_rag.app import create_app

        app = create_app()
        # TestClient 的 __enter__ 会触发 lifespan；如不崩则通过
        with TestClient(app) as client:
            r = client.get("/health/ready")
            assert r.status_code == 200
            print(f"    PASS: /health/ready → {r.status_code}")

    print("\n" + "=" * 60)
    print("ALL PASS: manager 容错回归")
    print("=" * 60)


if __name__ == "__main__":
    main()
