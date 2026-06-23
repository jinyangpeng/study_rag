"""run_chunking_pipeline 测试。"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from study_rag.jobs.manager import JobManager
from study_rag.jobs.models import JobStage, JobStatus
from study_rag.jobs.pipeline import run_chunking_pipeline


# ---- Fakes ----


class FakeEmbedder:
    """最小 Embedder 替身：返回固定维度的伪向量。"""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    async def embed_query(self, text: str) -> list[float]:
        return [0.1] * self.dim

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dim for _ in texts]


class FakeEmbedderRegistry:
    def __init__(self, embedder: FakeEmbedder) -> None:
        self._embedder = embedder

    def get(self, name: str) -> FakeEmbedder:
        return self._embedder


class FakeParserFactory:
    """模拟 NodeParserFactory.parse。"""

    def __init__(self, n_chunks: int = 3) -> None:
        self.n_chunks = n_chunks
        # 暴露 config 字段供 pipeline 读
        class _Cfg:
            strategy = "sentence"

        self._config = _Cfg()  # type: ignore[attr-defined]

    def parse(self, text: str, **kwargs: Any) -> list[Any]:
        # 把 text 切成 n_chunks 个伪 chunk
        size = max(1, len(text) // self.n_chunks)
        return [
            _FakeNode(
                text=text[i : i + size] or text,
                metadata={"title": kwargs.get("title", ""), "doc_id": kwargs.get("doc_id", "")},
            )
            for i in range(0, max(len(text), 1), size)
        ][: self.n_chunks]


class _FakeNode:
    def __init__(self, text: str, metadata: dict[str, Any]) -> None:
        self.text = text
        self.metadata = metadata
        self.node_id = f"node-{id(self)}"
        self.chunk_index = 0


class FakeParserRegistry:
    def __init__(self, factory: FakeParserFactory) -> None:
        self._factory = factory

    def get(self, name: str, **kwargs: Any) -> FakeParserFactory:
        return self._factory


class FakeKBConfig:
    def __init__(self, collection: str, embedder_name: str) -> None:
        self.collection = collection
        self.embedder_name = embedder_name


class FakeVectorStore:
    def __init__(self) -> None:
        self.inserted: list[tuple[str, list[Any]]] = []

    async def insert(self, collection: str, records: list[Any]) -> None:
        self.inserted.append((collection, records))


class FakeKBManager:
    def __init__(self, cfg: FakeKBConfig, vs: FakeVectorStore) -> None:
        self._cfg = cfg
        self._registry = FakeRegistry(cfg)
        self._vector_store = vs
        self._docs: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()


class FakeRegistry:
    def __init__(self, cfg: FakeKBConfig) -> None:
        self._cfg = cfg

    def get(self, kb_id: str) -> FakeKBConfig | None:
        return self._cfg if kb_id == "kb1" else None

    def get_required(self, kb_id: str) -> FakeKBConfig:
        cfg = self.get(kb_id)
        if cfg is None:
            raise KeyError(f"KB '{kb_id}' not found")
        return cfg


# ---- Tests ----


@pytest.mark.asyncio
async def test_pipeline_calls_progress_stages():
    """Pipeline 走过 parsing → chunking → embedding → saving。"""
    embedder = FakeEmbedder()
    vs = FakeVectorStore()
    cfg = FakeKBConfig(collection="c1", embedder_name="emb1")
    kb_manager = FakeKBManager(cfg, vs)
    parser_factory = FakeParserFactory(n_chunks=3)

    progress_calls: list[tuple[JobStage, int, int, str]] = []

    async def on_progress(
        stage: JobStage, current: int, total: int, message: str
    ) -> None:
        progress_calls.append((stage, current, total, message))

    def is_cancelled() -> bool:
        return False

    await run_chunking_pipeline(
        job_id="j1",
        on_progress=on_progress,
        is_cancelled=is_cancelled,
        file_content=b"Hello world. " * 100,
        filename="test.txt",
        doc_id="d1",
        title="T",
        parser_name="sentence_512",
        kb_id="kb1",
        embedder_registry=FakeEmbedderRegistry(embedder),  # type: ignore[arg-type]
        parser_registry=FakeParserRegistry(parser_factory),  # type: ignore[arg-type]
        kb_manager=kb_manager,  # type: ignore[arg-type]
    )

    # 验证 stage 序列包含所有四个
    stages_called = [p[0] for p in progress_calls]
    assert JobStage.PARSING in stages_called
    assert JobStage.CHUNKING in stages_called
    assert JobStage.EMBEDDING in stages_called
    assert JobStage.SAVING in stages_called

    # 验证 vector store 真的写入了
    assert len(vs.inserted) >= 1
    collection, records = vs.inserted[-1]
    assert collection == "c1"
    assert len(records) >= 1


@pytest.mark.asyncio
async def test_pipeline_respects_cancellation():
    """is_cancelled() == True 时应抛 CancelledError。"""
    embedder = FakeEmbedder()
    vs = FakeVectorStore()
    cfg = FakeKBConfig(collection="c1", embedder_name="emb1")
    kb_manager = FakeKBManager(cfg, vs)
    parser_factory = FakeParserFactory(n_chunks=3)

    async def on_progress(stage, current, total, message):
        pass

    cancel_flag = True

    def is_cancelled() -> bool:
        return cancel_flag

    with pytest.raises(asyncio.CancelledError):
        await run_chunking_pipeline(
            job_id="j1",
            on_progress=on_progress,
            is_cancelled=is_cancelled,
            file_content=b"Hello world. " * 100,
            filename="test.txt",
            doc_id="d1",
            title="T",
            parser_name="sentence_512",
            kb_id="kb1",
            embedder_registry=FakeEmbedderRegistry(embedder),  # type: ignore[arg-type]
            parser_registry=FakeParserRegistry(parser_factory),  # type: ignore[arg-type]
            kb_manager=kb_manager,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_pipeline_progress_callback_signature():
    """on_progress 必须接受 (stage, current, total, message) 四个参数（修 args shadowing bug）。"""
    embedder = FakeEmbedder()
    vs = FakeVectorStore()
    cfg = FakeKBConfig(collection="c1", embedder_name="emb1")
    kb_manager = FakeKBManager(cfg, vs)
    parser_factory = FakeParserFactory(n_chunks=2)

    captured = []

    async def on_progress(stage, current, total, message):
        # 验证参数被正确接收，不是被 *args 吞了
        captured.append(
            {
                "stage": stage,
                "current": current,
                "total": total,
                "message": message,
            }
        )
        # 验证 type 正确
        assert isinstance(stage, JobStage)
        assert isinstance(current, int)
        assert isinstance(total, int)
        assert isinstance(message, str)

    await run_chunking_pipeline(
        job_id="j1",
        on_progress=on_progress,
        is_cancelled=lambda: False,
        file_content=b"Hello. " * 50,
        filename="x.txt",
        doc_id="d1",
        title="T",
        parser_name="p1",
        kb_id="kb1",
        embedder_registry=FakeEmbedderRegistry(embedder),  # type: ignore[arg-type]
        parser_registry=FakeParserRegistry(parser_factory),  # type: ignore[arg-type]
        kb_manager=kb_manager,  # type: ignore[arg-type]
    )

    # 至少应该有 embedding 阶段的几条 progress
    embedding_progress = [c for c in captured if c["stage"] == JobStage.EMBEDDING]
    assert len(embedding_progress) >= 1
    # 最后一条 embedding progress 的 current 应等于 total
    last = embedding_progress[-1]
    assert last["current"] == last["total"]
    assert last["current"] > 0


@pytest.mark.asyncio
async def test_pipeline_integration_with_job_manager():
    """Pipeline 在 JobManager 调度下能跑完并置 DONE。"""
    mgr = JobManager()
    embedder = FakeEmbedder()
    vs = FakeVectorStore()
    cfg = FakeKBConfig(collection="c1", embedder_name="emb1")
    kb_manager = FakeKBManager(cfg, vs)
    parser_factory = FakeParserFactory(n_chunks=2)

    captured_progress: list[tuple[JobStage, int, int, str]] = []

    async def runner(job_id, on_progress, is_cancelled):
        captured_progress.append(("before", 0, 0, ""))
        await run_chunking_pipeline(
            job_id=job_id,
            on_progress=on_progress,
            is_cancelled=is_cancelled,
            file_content=b"content " * 20,
            filename="x.txt",
            doc_id="d1",
            title="T",
            parser_name="p1",
            kb_id="kb1",
            embedder_registry=FakeEmbedderRegistry(embedder),  # type: ignore[arg-type]
            parser_registry=FakeParserRegistry(parser_factory),  # type: ignore[arg-type]
            kb_manager=kb_manager,  # type: ignore[arg-type]
        )
        captured_progress.append(("after", 0, 0, ""))

    jid = await mgr.submit("upload_doc", runner)
    # 等待任务完成
    for _ in range(50):
        await asyncio.sleep(0.05)
        info = await mgr.get(jid)
        if info and info.status in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED):
            break
    info = await mgr.get(jid)
    assert info is not None
    assert info.status == JobStatus.DONE
    assert info.progress == 1.0
    # 验证 ran 前后的捕获
    assert captured_progress[0][0] == "before"
    assert captured_progress[-1][0] == "after"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
