"""Verify: 知识库 CRUD 端到端（建 → 改 → 删）。

覆盖：
  1. POST /admin/kbs 创建 KB
  2. GET /admin/kbs 能看到新 KB
  3. PATCH /admin/kbs/{id} 修改 enabled
  4. POST /admin/kbs 重复 kb_id → 409
  5. POST /admin/kbs 不存在的 embedding → 400
  6. GET /admin/embedders 列出所有配置
  7. DELETE /admin/kbs/{id} 删除并 drop collection
  8. 删除后 GET /admin/kbs 看不到
  9. yaml 文件包含新 KB
  10. 删完 yaml 也少一项

需要先在 yaml 里备份一个测试用 embedder；测试结束后清理。
"""

# ruff: noqa: T201
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _cleanup_previous_run() -> str:
    """清掉上一次失败留下的残留 KB（用 yaml 操作，不导入 app）。"""
    import yaml
    from study_rag.settings import AppPaths

    original = AppPaths.KB_CONFIG.read_text(encoding="utf-8")
    yaml_data = yaml.safe_load(original) or {}
    yaml_data["knowledge_bases"] = [
        c
        for c in yaml_data.get("knowledge_bases", [])
        if c.get("kb_id") != "test_verify_kb"
    ]
    AppPaths.KB_CONFIG.write_text(
        yaml.safe_dump(yaml_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return original


_backup_yaml = _cleanup_previous_run()


def main() -> None:
    import yaml
    from fastapi.testclient import TestClient
    from study_rag.app import app
    from study_rag.settings import AppPaths

    print("=" * 60)
    print("Verify: KB CRUD end-to-end")
    print("=" * 60)

    test_kb_id = "test_verify_kb"
    original = _backup_yaml

    # 重置 registry/manager 缓存
    from study_rag.knowledge_bases.registry import reset_registry_cache
    from study_rag.knowledge_bases.manager import reset_manager_singleton
    reset_registry_cache()
    reset_manager_singleton()

    # 注册一个测试用 embedder（mock provider，依赖最轻）
    test_embedder = {
        "mock_verify_embed": {
            "provider": "mock",
            "model_name": "test-mock",
            "dimension": 384,
            "batch_size": 16,
            "description": "测试用 mock embedder（不引真实模型）",
        }
    }
    embed_path = AppPaths.EMBEDDING_CONFIG
    embed_data = yaml.safe_load(embed_path.read_text(encoding="utf-8"))
    embed_data["embeddings"].update(test_embedder)
    embed_path.write_text(
        yaml.safe_dump(embed_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    try:
        client = TestClient(app)
        # 1. 列出 embedders
        print("\n[1] GET /admin/embedders")
        r = client.get("/admin/embedders")
        assert r.status_code == 200, r.text
        embedders = r.json()
        names = {e["name"] for e in embedders}
        assert "mock_verify_embed" in names
        print(f"    PASS: {len(embedders)} embedders, 含 mock_verify_embed")

        # 2. 创建一个新 KB
        print("\n[2] POST /admin/kbs 创建 KB")
        create_payload = {
            "kb_id": test_kb_id,
            "name": "Verify Test KB",
            "description": "测试用 KB，验证完成后会自动删除",
            "department": "QA",
            "embedding": "mock_verify_embed",
            "enabled": True,
        }
        r = client.post("/admin/kbs", json=create_payload)
        assert r.status_code == 201, f"status={r.status_code}, body={r.text}"
        created = r.json()
        assert created["kb_id"] == test_kb_id
        assert created["collection"] == f"kb_{test_kb_id}"
        print(f"    PASS: 创建成功 collection={created['collection']}")

        # 3. 列出 KB，能看到
        print("\n[3] GET /admin/kbs 包含新 KB")
        r = client.get("/admin/kbs")
        assert r.status_code == 200
        kb_ids = {k["kb_id"] for k in r.json()}
        assert test_kb_id in kb_ids
        print(f"    PASS: {len(kb_ids)} 个 KB, 含 {test_kb_id}")

        # 4. 重复创建 → 409
        print("\n[4] POST /admin/kbs 重复 → 409")
        r = client.post("/admin/kbs", json=create_payload)
        assert r.status_code == 409, f"status={r.status_code}, body={r.text}"
        print(f"    PASS: 409 ({r.json()['detail']})")

        # 5. 不存在的 embedding → 400
        print("\n[5] POST /admin/kbs 错 embedding → 400")
        bad = dict(create_payload, kb_id="bad_kb", embedding="no_such_embedder")
        r = client.post("/admin/kbs", json=bad)
        assert r.status_code == 400
        print(f"    PASS: 400 ({r.json()['detail'][:60]}...)")

        # 6. PATCH 改 enabled
        print("\n[6] PATCH /admin/kbs/{id} 关闭")
        r = client.patch(
            f"/admin/kbs/{test_kb_id}", json={"enabled": False}
        )
        assert r.status_code == 200, r.text
        assert r.json()["enabled"] is False
        print("    PASS: enabled=false 已更新")

        # 7. yaml 写回
        print("\n[7] yaml 文件包含新 KB")
        with AppPaths.KB_CONFIG.open(encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        kb_in_yaml = {c["kb_id"] for c in yaml_data["knowledge_bases"]}
        assert test_kb_id in kb_in_yaml
        print(f"    PASS: yaml 中含 {test_kb_id}")

        # 8. DELETE
        print("\n[8] DELETE /admin/kbs/{id}")
        r = client.delete(f"/admin/kbs/{test_kb_id}")
        assert r.status_code == 200
        print("    PASS: 删除返回 200")

        # 9. 删完 GET 看不到
        print("\n[9] GET /admin/kbs 不再含被删 KB")
        r = client.get("/admin/kbs")
        kb_ids = {k["kb_id"] for k in r.json()}
        assert test_kb_id not in kb_ids
        print("    PASS: 列表里已无该 KB")

        # 10. yaml 也少了一项
        print("\n[10] yaml 文件不再含被删 KB")
        with AppPaths.KB_CONFIG.open(encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        kb_in_yaml = {c["kb_id"] for c in yaml_data["knowledge_bases"]}
        assert test_kb_id not in kb_in_yaml
        print("    PASS: yaml 中已无该 KB")

        print("\n" + "=" * 60)
        print("ALL PASS: KB CRUD")
        print("=" * 60)

    finally:
        # 恢复原 yaml 和 embedder 配置
        AppPaths.KB_CONFIG.write_text(original, encoding="utf-8")
        embed_data = yaml.safe_load(embed_path.read_text(encoding="utf-8"))
        embed_data["embeddings"].pop("mock_verify_embed", None)
        embed_path.write_text(
            yaml.safe_dump(embed_data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        # 清一下 .yaml.tmp 残留（原子写异常时）
        for tmp in AppPaths.KB_CONFIG.parent.glob(".kbs_*.yaml.tmp"):
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
