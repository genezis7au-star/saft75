"""
Production monitoring: real-time quality metrics, regression detection,
cost/latency tracking, and alerting.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MetricPoint:
    """A single observation of a named metric."""

    name: str
    value: float
    timestamp: str = field(default_factory=_utcnow_iso)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """An alert triggered when a metric crosses a threshold."""

    name: str
    message: str
    severity: str  # "info" | "warning" | "critical"
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: str = field(default_factory=_utcnow_iso)


@dataclass
class AlertRule:
    """Rule that fires an alert when a condition is met."""

    name: str
    metric_name: str
    threshold: float
    condition: str  # "above" | "below"
    severity: str = "warning"
    message_template: str = "Metric {metric} = {value:.3f} crossed threshold {threshold}"

    def check(self, metric_name: str, value: float) -> Optional[Alert]:
        if metric_name != self.metric_name:
            return None
        triggered = (self.condition == "above" and value > self.threshold) or (
            self.condition == "below" and value < self.threshold
        )
        if not triggered:
            return None
        return Alert(
            name=self.name,
            message=self.message_template.format(
                metric=metric_name, value=value, threshold=self.threshold
            ),
            severity=self.severity,
            metric_name=metric_name,
            metric_value=value,
            threshold=self.threshold,
        )


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """
    Lightweight in-memory metrics collector.

    Records named metrics as time series, supports windowed statistics,
    regression detection, and alert rules.

    Designed for production use — can be extended to emit to Prometheus,
    Grafana, or a time-series DB.
    """

    def __init__(self, window_size: int = 100) -> None:
        self._window_size = window_size
        self._series: Dict[str, Deque[MetricPoint]] = {}
        self._alert_rules: List[AlertRule] = []
        self._fired_alerts: List[Alert] = []
        self._alert_handlers: List[Callable[[Alert], None]] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a metric observation."""
        if name not in self._series:
            self._series[name] = deque(maxlen=self._window_size)

        point = MetricPoint(name=name, value=value, labels=labels or {})
        self._series[name].append(point)
        self._check_alerts(name, value)

    def record_latency(self, operation: str, duration_ms: float) -> None:
        self.record(f"latency_ms.{operation}", duration_ms)

    def record_cost(self, operation: str, cost_usd: float) -> None:
        self.record(f"cost_usd.{operation}", cost_usd)

    def record_quality(self, dimension: str, score: float) -> None:
        self.record(f"quality.{dimension}", score)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_series(self, name: str) -> List[MetricPoint]:
        return list(self._series.get(name, []))

    def stats(self, name: str) -> Dict[str, Any]:
        """Return windowed statistics for a named metric."""
        points = self.get_series(name)
        if not points:
            return {"count": 0}
        values = [p.value for p in points]
        mean = sum(values) / len(values)
        sorted_v = sorted(values)
        n = len(sorted_v)
        return {
            "count": n,
            "mean": mean,
            "min": sorted_v[0],
            "max": sorted_v[-1],
            "p50": sorted_v[n // 2],
            "p95": sorted_v[int(n * 0.95)],
            "std": (sum((v - mean) ** 2 for v in values) / n) ** 0.5,
            "latest": values[-1],
        }

    def all_metrics(self) -> List[str]:
        return list(self._series.keys())

    # ------------------------------------------------------------------
    # Regression detection
    # ------------------------------------------------------------------

    def detect_regression(
        self,
        name: str,
        baseline_window: int = 20,
        current_window: int = 5,
        threshold_pct: float = 10.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Detect a regression in a metric.

        Compares the rolling mean of the most recent ``current_window`` points
        against the preceding ``baseline_window`` points. Returns a dict if
        a regression is detected, otherwise None.
        """
        points = self.get_series(name)
        total_needed = baseline_window + current_window
        if len(points) < total_needed:
            return None

        baseline_vals = [p.value for p in points[-(total_needed):-current_window]]
        current_vals = [p.value for p in points[-current_window:]]

        baseline_mean = sum(baseline_vals) / len(baseline_vals)
        current_mean = sum(current_vals) / len(current_vals)

        if baseline_mean == 0:
            return None

        change_pct = (current_mean - baseline_mean) / baseline_mean * 100.0
        if abs(change_pct) > threshold_pct:
            return {
                "metric": name,
                "baseline_mean": baseline_mean,
                "current_mean": current_mean,
                "change_pct": change_pct,
                "direction": "improvement" if change_pct > 0 else "regression",
                "detected_at": _utcnow_iso(),
            }
        return None

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def add_alert_rule(self, rule: AlertRule) -> None:
        self._alert_rules.append(rule)

    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        self._alert_handlers.append(handler)

    def _check_alerts(self, name: str, value: float) -> None:
        for rule in self._alert_rules:
            alert = rule.check(name, value)
            if alert:
                self._fired_alerts.append(alert)
                for handler in self._alert_handlers:
                    handler(alert)

    def get_alerts(self, severity: Optional[str] = None) -> List[Alert]:
        if severity is None:
            return list(self._fired_alerts)
        return [a for a in self._fired_alerts if a.severity == severity]

    def clear_alerts(self) -> None:
        self._fired_alerts.clear()

    # ------------------------------------------------------------------
    # Dashboard summary
    # ------------------------------------------------------------------

    def dashboard(self) -> Dict[str, Any]:
        """Return a dashboard-ready summary of all metrics."""
        return {
            "metrics": {name: self.stats(name) for name in self._series},
            "active_alerts": len(self._fired_alerts),
            "alert_rules": len(self._alert_rules),
            "generated_at": _utcnow_iso(),
        }


# ---------------------------------------------------------------------------
# Latency context manager
# ---------------------------------------------------------------------------


class LatencyTimer:
    """Context manager for recording operation latency."""

    def __init__(self, collector: MetricsCollector, operation: str) -> None:
        self._collector = collector
        self._operation = operation
        self._start: float = 0.0

    def __enter__(self) -> "LatencyTimer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *_: Any) -> None:
        duration_ms = (time.monotonic() - self._start) * 1000
        self._collector.record_latency(self._operation, duration_ms)
