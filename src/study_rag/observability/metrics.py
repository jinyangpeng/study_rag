"""In-memory Prometheus 兼容 metrics。

不引入 prometheus_client 等新依赖，自己实现 counter / histogram。
暴露格式遵循 Prometheus text exposition format。

用法:
    metrics = get_metrics()
    metrics.inc("study_rag_search_total", {"kb_id": "rd_frontend", "status": "ok"})
    metrics.observe("study_rag_search_latency_ms", 42.0, {"kb_id": "rd_frontend"})

端点:
    GET /metrics  →  Prometheus text format
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any


class MetricsRegistry:
    """轻量级 metrics 注册表（线程安全）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # metric_name -> labels_key -> count
        # type: ignore[type-arg]  # defaultdict 推断为嵌套 dict 但实际是 defaultdict
        self._counters: dict[str, dict[frozenset, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # metric_name -> labels_key -> [values]
        self._histograms: dict[str, dict[frozenset, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # metric_name -> labels_key -> sum
        self._histograms_sum: dict[str, dict[frozenset, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # metric_name -> labels_key -> count
        self._histograms_count: dict[str, dict[frozenset, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def inc(
        self,
        name: str,
        labels: dict[str, str] | None = None,
        value: float = 1.0,
    ) -> None:
        """counter +value（默认 +1）。"""
        labels_key = frozenset((labels or {}).items())
        with self._lock:
            self._counters[name][labels_key] += value

    def observe(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """histogram 记录一个观测值。"""
        labels_key = frozenset((labels or {}).items())
        with self._lock:
            self._histograms[name][labels_key].append(value)
            self._histograms_sum[name][labels_key] += value
            self._histograms_count[name][labels_key] += 1

    def render(self) -> str:
        """渲染为 Prometheus text format。"""
        lines: list[str] = []

        # Counters
        with self._lock:
            for name, label_map in self._counters.items():
                lines.append(f"# TYPE {name} counter")
                for labels_key, value in label_map.items():
                    labels_str = self._format_labels(dict(labels_key))
                    lines.append(f"{name}{labels_str} {value}")

            # Histograms
            for h_name, h_label_map in self._histograms.items():
                lines.append(f"# TYPE {h_name} histogram")
                for h_labels_key, h_values in h_label_map.items():
                    base_labels = dict(h_labels_key)
                    total_count = self._histograms_count[h_name][h_labels_key]
                    total_sum = self._histograms_sum[h_name][h_labels_key]

                    # 桶
                    buckets: list[float] = [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
                    sorted_values: list[float] = sorted(h_values)
                    for bucket in buckets:
                        b: float = bucket
                        count = sum(1 for v in sorted_values if v <= b)
                        bucket_labels = {**base_labels, "le": str(bucket)}
                        labels_str = self._format_labels(bucket_labels)
                        lines.append(f"{h_name}_bucket{labels_str} {count}")
                    # +Inf 桶
                    inf_labels = {**base_labels, "le": "+Inf"}
                    labels_str = self._format_labels(inf_labels)
                    lines.append(f"{h_name}_bucket{labels_str} {total_count}")

                    # sum / count
                    labels_str = self._format_labels(base_labels)
                    lines.append(f"{h_name}_sum{labels_str} {total_sum}")
                    lines.append(f"{h_name}_count{labels_str} {total_count}")

        return "\n".join(lines) + "\n"

    def _format_labels(self, labels: dict[str, str]) -> str:
        if not labels:
            return ""
        parts = ",".join(f'{k}="{self._escape(v)}"' for k, v in sorted(labels.items()))
        return "{" + parts + "}"

    def _escape(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


# ---- 单例 ----
_metrics_singleton: MetricsRegistry | None = None
_metrics_lock = threading.Lock()


def get_metrics() -> MetricsRegistry:
    """获取全局 metrics 单例。"""
    global _metrics_singleton
    if _metrics_singleton is None:
        with _metrics_lock:
            if _metrics_singleton is None:
                _metrics_singleton = MetricsRegistry()
    return _metrics_singleton


# ---- 预定义 metric 名（避免散落字符串） ----


class SearchMetrics:
    """检索相关 metric 名集中定义。"""

    TOTAL = "study_rag_search_total"
    LATENCY = "study_rag_search_latency_ms"
    HITS = "study_rag_search_hits"


class AdminMetrics:
    """Admin REST metric 名集中定义。"""

    REQUESTS = "study_rag_admin_requests_total"
    LATENCY = "study_rag_admin_latency_ms"
    DOCUMENTS = "study_rag_admin_documents_total"


# 编译期 type hint（保留 Any 用于标签）
_ = Any
