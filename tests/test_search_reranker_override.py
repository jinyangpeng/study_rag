"""检索链路 reranker_name 覆盖逻辑测试。

验证 MCP search_kb 的 reranker 选择优先级：
  - use_rerank=False → 不重排，reranker_name 被忽略
  - use_rerank=True + reranker_name=None → 用 KB 默认绑定的 reranker
  - use_rerank=True + reranker_name="xxx" → 用指定 reranker（覆盖 KB 默认）
  - reranker_name 指定不存在的配置名 → InvalidParameterError

注意：新架构下 search_kb 委托给 ctx.manager.search_via_strategy()，
reranker 逻辑在 Manager 内部处理。测试验证 search_via_strategy 被正确调用。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from study_rag.capabilities.retrieval import RetrievalResponse, RetrievalStrategy
from study_rag.capabilities.vector_store.base import SearchResult as VSResult
from study_rag.knowledge_bases.manager import ComponentUnavailableError
from study_rag.mcp.errors import InvalidParameterError
from study_rag.mcp.tools.search import search_kb


def _make_ctx(
    reranker_by_name=None,
    kb_reranker=None,
    embedder=None,
    vector_store=None,
    search_via_strategy_result=None,
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

    # search_via_strategy 默认返回带 2 条结果的 RetrievalResponse
    if search_via_strategy_result is None:
        search_via_strategy_result = RetrievalResponse(
            kb_id="kb1",
            query="测试",
            strategy=RetrievalStrategy.DENSE,
            results=[
                VSResult(id="c1", text="文本1", score=0.5, metadata={"doc_id": "d1"}),
                VSResult(id="c2", text="文本2", score=0.4, metadata={"doc_id": "d2"}),
            ],
        )
    manager.search_via_strategy = AsyncMock(return_value=search_via_strategy_result)

    ctx.manager = manager
    return ctx


def _make_kb_cfg(reranker: str | None = "kb_default_reranker") -> MagicMock:
    cfg = MagicMock()
    cfg.enabled = True
    cfg.collection = "col_test"
    cfg.reranker = reranker
    cfg.retrieval_strategy = None
    cfg.retrieval_params = {}
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
    r._top_k = 3
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
    reranked_result = RetrievalResponse(
        kb_id="kb1",
        query="测试",
        strategy=RetrievalStrategy.DENSE,
        results=[
            VSResult(
                id="c1", text="文本1", score=0.91,
                metadata={"doc_id": "d1", "reranked_by": "mock"},
            )
        ],
    )
    ctx = _make_ctx(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        search_via_strategy_result=reranked_result,
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

    # 验证 search_via_strategy 被调用，且 reranker_name 被传递
    ctx.manager.search_via_strategy.assert_called_once()
    call_kwargs = ctx.manager.search_via_strategy.call_args.kwargs
    assert call_kwargs["reranker_name"] == "tei_bge_m3"
    assert call_kwargs["use_rerank"] is True
    assert len(hits) == 1
    assert hits[0].metadata.get("reranked_by") == "mock"


@pytest.mark.asyncio
async def test_reranker_name_none_uses_kb_default(
    mock_embedder, mock_vector_store, mock_reranker
):
    """reranker_name=None 时用 KB 默认绑定的 reranker。"""
    ctx = _make_ctx(
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

    # 验证 search_via_strategy 被调用，且 reranker_name=None
    ctx.manager.search_via_strategy.assert_called_once()
    call_kwargs = ctx.manager.search_via_strategy.call_args.kwargs
    assert call_kwargs["reranker_name"] is None
    assert call_kwargs["use_rerank"] is True


@pytest.mark.asyncio
async def test_reranker_name_not_found_raises(mock_embedder, mock_vector_store):
    """reranker_name 指定不存在的配置名 → InvalidParameterError。"""

    async def raise_unavail(**kwargs):
        raise ComponentUnavailableError(
            component="reranker", name=kwargs.get("reranker_name", ""), hint="not found"
        )

    ctx = _make_ctx(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
    )
    ctx.manager.search_via_strategy = AsyncMock(side_effect=raise_unavail)
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

    # 验证 search_via_strategy 被调用，且 use_rerank=False
    ctx.manager.search_via_strategy.assert_called_once()
    call_kwargs = ctx.manager.search_via_strategy.call_args.kwargs
    assert call_kwargs["use_rerank"] is False
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

    # 新代码路径通过 search_via_strategy 调用，验证调用了该方法
    ctx.manager.search_via_strategy.assert_called_once()
    call_kwargs = ctx.manager.search_via_strategy.call_args.kwargs
    assert call_kwargs["top_k"] == 5  # 默认值
    assert call_kwargs["use_rerank"] is True


@pytest.mark.asyncio
async def test_top_k_explicit_is_recall_count(
    mock_embedder, mock_vector_store, mock_reranker
):
    """显式 top_k=10 表示向量召回 10 条；reranker 仍用自身配置过滤。

    正确语义：用户填的 Top K 是向量召回数，不是最终返回数。
    """
    ctx = _make_ctx(
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

    # 新代码路径通过 search_via_strategy 调用
    ctx.manager.search_via_strategy.assert_called_once()
    call_kwargs = ctx.manager.search_via_strategy.call_args.kwargs
    assert call_kwargs["top_k"] == 10
    assert call_kwargs["use_rerank"] is True
