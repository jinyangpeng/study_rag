"""检索链路 reranker_name 覆盖逻辑测试。

验证 MCP search_kb 的 reranker 选择优先级：
  - use_rerank=False → 不重排，reranker_name 被忽略
  - use_rerank=True + reranker_name=None → 用 KB 默认绑定的 reranker
  - use_rerank=True + reranker_name="xxx" → 用指定 reranker（覆盖 KB 默认）
  - reranker_name 指定不存在的配置名 → InvalidParameterError
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from study_rag.capabilities.vector_store.base import SearchResult as VSResult
from study_rag.knowledge_bases.manager import ComponentUnavailableError
from study_rag.mcp.errors import InvalidParameterError
from study_rag.mcp.tools.search import search_kb


def _make_ctx(
    reranker_by_name=None,
    kb_reranker=None,
    embedder=None,
    vector_store=None,
) -> MagicMock:
    """构造 mock MCPContext。"""
    ctx = MagicMock()
    ctx.auth.resolve = AsyncMock(return_value=MagicMock(accessible_kbs=set()))
    ctx.auth.check_kb_access = MagicMock()

    manager = MagicMock()
    manager.get_embedder = MagicMock(return_value=embedder)
    manager.get_vector_store = MagicMock(return_value=vector_store)
    manager.get_reranker = MagicMock(side_effect=reranker_by_name)
    manager.get_reranker_for_kb = MagicMock(return_value=kb_reranker)
    ctx.manager = manager
    return ctx


def _make_kb_cfg(reranker: str | None = "kb_default_reranker") -> MagicMock:
    cfg = MagicMock()
    cfg.enabled = True
    cfg.collection = "col_test"
    cfg.reranker = reranker
    return cfg


@pytest.fixture
def mock_embedder() -> MagicMock:
    e = MagicMock()
    e.embed_query = AsyncMock(return_value=[0.1] * 4)
    return e


@pytest.fixture
def mock_vector_store() -> MagicMock:
    vs = MagicMock()
    vs.search = AsyncMock(
        return_value=[
            VSResult(id="c1", text="文本1", score=0.5, metadata={"doc_id": "d1"}),
            VSResult(id="c2", text="文本2", score=0.4, metadata={"doc_id": "d2"}),
        ]
    )
    return vs


@pytest.fixture
def mock_reranker() -> MagicMock:
    """reranker mock：重排后返回带标记的结果，便于断言「确实走了 reranker」。"""
    r = MagicMock()
    r._top_k = 3  # 模拟 reranker 配置的 top_k
    r.rerank = AsyncMock(
        return_value=[
            VSResult(
                id="c1",
                text="文本1",
                score=0.91,
                metadata={"doc_id": "d1", "reranked_by": "mock"},
            )
        ]
    )
    return r


@pytest.mark.asyncio
async def test_reranker_name_overrides_kb_default(
    mock_embedder, mock_vector_store, mock_reranker
):
    """显式 reranker_name 时用指定 reranker，不调用 KB 默认。"""
    ctx = _make_ctx(
        reranker_by_name=lambda n: mock_reranker if n == "tei_bge_m3" else None,
        kb_reranker=MagicMock(),  # KB 默认，不应被用到
        embedder=mock_embedder,
        vector_store=mock_vector_store,
    )
    kb_cfg = _make_kb_cfg()

    with patch("study_rag.knowledge_bases.registry.get_registry") as mock_reg:
        mock_reg.return_value.get = MagicMock(return_value=kb_cfg)
        hits = await search_kb(
            api_key="admin",
            kb_id="kb1",
            query="测试",
            top_k=5,
            use_rerank=True,
            reranker_name="tei_bge_m3",
            ctx=ctx,
        )

    # 用了按名取的 reranker，而非 KB 默认
    ctx.manager.get_reranker.assert_called_once_with("tei_bge_m3")
    ctx.manager.get_reranker_for_kb.assert_not_called()
    assert len(hits) == 1
    assert hits[0].metadata.get("reranked_by") == "mock"


@pytest.mark.asyncio
async def test_reranker_name_none_uses_kb_default(
    mock_embedder, mock_vector_store, mock_reranker
):
    """reranker_name=None 时用 KB 默认绑定的 reranker。"""
    ctx = _make_ctx(
        reranker_by_name=lambda n: mock_reranker,
        kb_reranker=mock_reranker,
        embedder=mock_embedder,
        vector_store=mock_vector_store,
    )
    kb_cfg = _make_kb_cfg()

    with patch("study_rag.knowledge_bases.registry.get_registry") as mock_reg:
        mock_reg.return_value.get = MagicMock(return_value=kb_cfg)
        await search_kb(
            api_key="admin",
            kb_id="kb1",
            query="测试",
            top_k=5,
            use_rerank=True,
            reranker_name=None,
            ctx=ctx,
        )

    ctx.manager.get_reranker_for_kb.assert_called_once_with("kb1")
    ctx.manager.get_reranker.assert_not_called()


@pytest.mark.asyncio
async def test_reranker_name_not_found_raises(mock_embedder, mock_vector_store):
    """reranker_name 指定不存在的配置名 → InvalidParameterError。"""

    def raise_unavail(name):
        raise ComponentUnavailableError(
            component="reranker", name=name, hint="not found"
        )

    ctx = _make_ctx(
        reranker_by_name=raise_unavail,
        embedder=mock_embedder,
        vector_store=mock_vector_store,
    )
    kb_cfg = _make_kb_cfg()

    with patch("study_rag.knowledge_bases.registry.get_registry") as mock_reg:
        mock_reg.return_value.get = MagicMock(return_value=kb_cfg)
        with pytest.raises(InvalidParameterError):
            await search_kb(
                api_key="admin",
                kb_id="kb1",
                query="测试",
                top_k=5,
                use_rerank=True,
                reranker_name="nonexistent",
                ctx=ctx,
            )


@pytest.mark.asyncio
async def test_use_rerank_false_ignores_reranker_name(
    mock_embedder, mock_vector_store
):
    """use_rerank=False 时不重排，reranker_name 被忽略。"""
    ctx = _make_ctx(
        reranker_by_name=lambda n: pytest.fail(
            "should not call get_reranker when use_rerank=False"
        ),
        kb_reranker=None,
        embedder=mock_embedder,
        vector_store=mock_vector_store,
    )
    kb_cfg = _make_kb_cfg()

    with patch("study_rag.knowledge_bases.registry.get_registry") as mock_reg:
        mock_reg.return_value.get = MagicMock(return_value=kb_cfg)
        hits = await search_kb(
            api_key="admin",
            kb_id="kb1",
            query="测试",
            top_k=5,
            use_rerank=False,
            reranker_name="tei_bge_m3",  # 即使指定，也不重排
            ctx=ctx,
        )

    ctx.manager.get_reranker.assert_not_called()
    ctx.manager.get_reranker_for_kb.assert_not_called()
    # 直接截断，未重排（mock 返回 2 条）
    assert len(hits) == 2


@pytest.mark.asyncio
async def test_top_k_none_uses_reranker_config(
    mock_embedder, mock_vector_store, mock_reranker
):
    """top_k=None 时：向量召回用默认 5（×4=20 候选），rerank 用 reranker 配置的 top_k。

    正确语义：Top K（前端输入）= 向量召回数；reranker.top_k = 重排后保留数。
    """
    ctx = _make_ctx(
        reranker_by_name=lambda n: mock_reranker,
        kb_reranker=mock_reranker,
        embedder=mock_embedder,
        vector_store=mock_vector_store,
    )
    kb_cfg = _make_kb_cfg()

    with patch("study_rag.knowledge_bases.registry.get_registry") as mock_reg:
        mock_reg.return_value.get = MagicMock(return_value=kb_cfg)
        await search_kb(
            api_key="admin",
            kb_id="kb1",
            query="测试",
            top_k=None,  # 向量召回数用默认 5
            use_rerank=True,
            ctx=ctx,
        )

    # 向量召回：recall_k=5，启用 reranker → candidate_k = 5 * 4 = 20
    assert mock_vector_store.search.call_args.kwargs["top_k"] == 20
    # reranker.rerank 收到 top_k=None（用自身配置的 _top_k=3 过滤）
    mock_reranker.rerank.assert_called_once()
    assert mock_reranker.rerank.call_args.kwargs["top_k"] is None


@pytest.mark.asyncio
async def test_top_k_explicit_is_recall_count(
    mock_embedder, mock_vector_store, mock_reranker
):
    """显式 top_k=10 表示向量召回 10 条；reranker 仍用自身配置过滤。

    正确语义：用户填的 Top K 是向量召回数，不是最终返回数。
    """
    ctx = _make_ctx(
        reranker_by_name=lambda n: mock_reranker,
        kb_reranker=mock_reranker,
        embedder=mock_embedder,
        vector_store=mock_vector_store,
    )
    kb_cfg = _make_kb_cfg()

    with patch("study_rag.knowledge_bases.registry.get_registry") as mock_reg:
        mock_reg.return_value.get = MagicMock(return_value=kb_cfg)
        await search_kb(
            api_key="admin",
            kb_id="kb1",
            query="测试",
            top_k=10,  # 向量召回 10 条
            use_rerank=True,
            ctx=ctx,
        )

    # 向量召回：recall_k=10，启用 reranker → candidate_k = 10 * 4 = 40
    assert mock_vector_store.search.call_args.kwargs["top_k"] == 40
    # reranker 用自身配置的 top_k 过滤（收到 None）
    assert mock_reranker.rerank.call_args.kwargs["top_k"] is None
