"""pytest 共享 fixtures + 配置。

verify_*.py 是脚本式验证（直接 if __name__ == "__main__"），
不是 pytest 测试函数，默认不收集。pytest 只收集 test_*.py。
"""

from __future__ import annotations

# 显式只收集 test_*.py
collect_ignore_glob = ["verify_*.py"]
