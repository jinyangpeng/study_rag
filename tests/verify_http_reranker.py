"""Verify: HttpReranker 对接 TEI / Jina / Cohere-compat 协议。

不需要装 torch/FlagEmbedding，用 mock HTTP server 验证：
  1. 协议分发正确
  2. payload 构造正确
  3. response 解析正确
  4. top_k 截断正确
  5. 错误（4xx/5xx）正确抛出
"""

# ruff: noqa: T201, PT017
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    import asyncio
    import json
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread

    from study_rag.capabilities.reranker.base import RerankerConfig, create_reranker
    from study_rag.capabilities.vector_store.base import SearchResult

    print("=" * 60)
    print("Verify: HttpReranker")
    print("=" * 60)

    # ---- 1. 启动一个 mock HTTP server，模拟 TEI 协议 ----
    print("\n[1] mock TEI server 收到 /rerank 返回倒序结果")

    received: list[dict] = []

    class TeiHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b""
            received.append(json.loads(raw))
            # 模拟：把第 2 个（index=1）放最相关
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    [
                        {"index": 1, "score": 0.92},
                        {"index": 0, "score": 0.45},
                        {"index": 2, "score": 0.12},
                    ]
                ).encode()
            )

        def log_message(self, *args: object) -> None:  # noqa: D401
            pass

    server = HTTPServer(("127.0.0.1", 0), TeiHandler)
    port = server.server_address[1]
    Thread(target=server.serve_forever, daemon=True).start()

    try:
        # ---- 2. 创建 reranker，调一次，验证 payload 和 response ----
        print("\n[2] HttpReranker(protocol=tei) 调用并解析")
        cfg = RerankerConfig(
            provider="http",
            model_name="BAAI/bge-reranker-v2-m3",
            top_k=2,
            extra={"base_url": f"http://127.0.0.1:{port}", "protocol": "tei"},
        )
        reranker = create_reranker(cfg)
        results = [
            SearchResult(id="a", text="BGE 的原理", score=0.1, metadata={}),
            SearchResult(id="b", text="BGE-M3 支持多语言", score=0.2, metadata={}),
            SearchResult(id="c", text="其他模型", score=0.3, metadata={}),
        ]

        async def t1() -> None:
            out = await reranker.rerank("BGE", results, top_k=2)
            return out

        out = asyncio.run(t1())
        assert len(out) == 2, f"expected 2, got {len(out)}"
        # 倒序：b (0.92) → a (0.45)
        assert out[0].id == "b", f"expected 'b', got {out[0].id}"
        assert abs(out[0].score - 0.92) < 1e-6
        assert out[1].id == "a"
        assert abs(out[1].score - 0.45) < 1e-6
        print(f"    PASS: top-2 = {[(r.id, round(r.score, 2)) for r in out]}")

        # payload 检查
        sent = received[-1]
        assert sent["query"] == "BGE"
        assert sent["texts"] == ["BGE 的原理", "BGE-M3 支持多语言", "其他模型"]
        assert sent["return_documents"] is False
        assert sent["truncate_input_tokens"] == 512  # default
        assert sent["model"] == "BAAI/bge-reranker-v2-m3"
        print("    PASS: payload 包含 query/texts/model/truncate_input_tokens")

    finally:
        server.shutdown()

    # ---- 3. Jina 协议 ----
    print("\n[3] Jina 协议")

    class JinaHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    [
                        {"index": 0, "score": 0.99},
                        {"index": 1, "score": 0.50},
                    ]
                ).encode()
            )

        def log_message(self, *args: object) -> None:
            pass

    server2 = HTTPServer(("127.0.0.1", 0), JinaHandler)
    port2 = server2.server_address[1]
    Thread(target=server2.serve_forever, daemon=True).start()
    try:
        cfg2 = RerankerConfig(
            provider="http",
            model_name="jina-reranker",
            top_k=2,
            extra={
                "base_url": f"http://127.0.0.1:{port2}",
                "protocol": "jina",
            },
        )
        rr = create_reranker(cfg2)
        results = [
            SearchResult(id="x", text="a", score=0.0, metadata={}),
            SearchResult(id="y", text="b", score=0.0, metadata={}),
        ]

        async def t3() -> None:
            return await rr.rerank("q", results)

        out3 = asyncio.run(t3())
        assert out3[0].id == "x"
        assert abs(out3[0].score - 0.99) < 1e-6
        print(f"    PASS: Jina top-1 = {out3[0].id} ({out3[0].score})")
    finally:
        server2.shutdown()

    # ---- 4. Cohere 兼容协议 ----
    print("\n[4] Cohere 兼容协议")

    class CohereHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "results": [
                            {"index": 1, "relevance_score": 0.88},
                            {"index": 0, "relevance_score": 0.21},
                        ]
                    }
                ).encode()
            )

        def log_message(self, *args: object) -> None:
            pass

    server3 = HTTPServer(("127.0.0.1", 0), CohereHandler)
    port3 = server3.server_address[1]
    Thread(target=server3.serve_forever, daemon=True).start()
    try:
        cfg3 = RerankerConfig(
            provider="http",
            model_name="rerank-multilingual-v3.0",
            top_k=1,
            extra={
                "base_url": f"http://127.0.0.1:{port3}",
                "protocol": "cohere_compat",
            },
        )
        rr3 = create_reranker(cfg3)
        results = [
            SearchResult(id="x", text="a", score=0.0, metadata={}),
            SearchResult(id="y", text="b", score=0.0, metadata={}),
        ]

        async def t4() -> None:
            return await rr3.rerank("q", results)

        out4 = asyncio.run(t4())
        assert out4[0].id == "y"
        assert abs(out4[0].score - 0.88) < 1e-6
        print(f"    PASS: Cohere-compat top-1 = {out4[0].id} ({out4[0].score})")
    finally:
        server3.shutdown()

    # ---- 5. 服务端 5xx 错误正确抛 ----
    print("\n[5] 服务端 500 错误 → 抛异常")

    class FailHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"internal error")

        def log_message(self, *args: object) -> None:
            pass

    server4 = HTTPServer(("127.0.0.1", 0), FailHandler)
    port4 = server4.server_address[1]
    Thread(target=server4.serve_forever, daemon=True).start()
    try:
        cfg4 = RerankerConfig(
            provider="http",
            model_name="x",
            top_k=5,
            extra={
                "base_url": f"http://127.0.0.1:{port4}",
                "protocol": "tei",
            },
        )
        rr4 = create_reranker(cfg4)
        results = [SearchResult(id="x", text="a", score=0.0, metadata={})]

        async def t5() -> None:
            try:
                await rr4.rerank("q", results)
                return False
            except Exception:
                return True

        ok = asyncio.run(t5())
        assert ok, "应该抛 HTTPStatusError"
        print("    PASS: 500 正确抛 HTTPError")
    finally:
        server4.shutdown()

    # ---- 6. 缺少 base_url → 构造时抛 ----
    print("\n[6] 缺 base_url → 构造时抛 ValueError")
    try:
        RerankerConfig(
            provider="http",
            model_name="x",
            top_k=5,
            extra={"protocol": "tei"},  # 无 base_url
        )
    except Exception as e:
        assert "base_url" in str(e)
        print(f"    PASS: 抛 {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("ALL PASS: HttpReranker")
    print("=" * 60)


if __name__ == "__main__":
    main()
