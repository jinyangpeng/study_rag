import json

# 69 bytes 的可能原因
candidates = [
    "ComponentUnavailableError('Reranker local_bge_reranker_base not found')",
    "ComponentUnavailableError: Reranker local_bge_reranker_base not found",
    "InvalidParameterError('top_k must be in (0, 50] or None')",
    "InvalidParameterError: top_k must be in (0, 50] or None",
    "InvalidParameterError('query must not be empty')",
    "InvalidParameterError: query must not be empty",
    "InvalidStrategy: hybrid not available for kb rd_test",
    "HybridStrategyNotConfigured: kb rd_test missing sparse config",
]

for msg in candidates:
    body = json.dumps({"detail": msg}, ensure_ascii=False)
    print(f"{len(body):3d} bytes: {msg}")

print()
# 也测试 str(e) 格式的异常
class MockError(Exception):
    def __init__(self, msg):
        super().__init__(msg)

errors = [
    MockError("Reranker local_bge_reranker_base not loaded"),
    MockError("Hybrid engine requires sparse embedder"),
    MockError("Sparse engine init failed: no tokenizer"),
]

for e in errors:
    body = json.dumps({"detail": str(e)}, ensure_ascii=False)
    print(f"{len(body):3d} bytes: {str(e)}")
