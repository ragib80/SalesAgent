"""Schemas for Phase 2 structured analytical query planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _as_list(value: Any) -> list[str]:
    """Normalize scalar or list-like values into a list of strings."""

    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _as_dict(value: Any) -> dict[str, Any]:
    """Return a dictionary for structured plan fields."""

    return value if isinstance(value, dict) else {}


def _as_limit(value: Any) -> int | None:
    """Normalize optional result limits from LLM output."""

    if value in (None, ""):
        return None
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return None
    return limit if limit > 0 else None


def _as_confidence(value: Any) -> float | None:
    """Normalize confidence values to a 0.0-1.0 float when possible."""

    if value in (None, ""):
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, confidence))


@dataclass(slots=True)
class QueryPlan:
    """Structured interpretation of a user's SAP sales analysis request."""

    intent: str = "unknown"
    metrics: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    time_range: dict[str, Any] = field(default_factory=dict)
    aggregation: str = ""
    sorting: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None
    comparison: dict[str, Any] = field(default_factory=dict)
    visualization: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    assumptions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QueryPlan":
        """Build a normalized plan from raw LLM JSON output."""

        metrics = payload.get("metrics", payload.get("metric"))
        return cls(
            intent=str(payload.get("intent") or "unknown").strip() or "unknown",
            metrics=_as_list(metrics),
            dimensions=_as_list(payload.get("dimensions")),
            filters=_as_dict(payload.get("filters")),
            time_range=_as_dict(payload.get("time_range")),
            aggregation=str(payload.get("aggregation") or "").strip(),
            sorting=_as_dict(payload.get("sorting")),
            limit=_as_limit(payload.get("limit")),
            comparison=_as_dict(payload.get("comparison")),
            visualization=_as_dict(payload.get("visualization")),
            confidence=_as_confidence(payload.get("confidence")),
            assumptions=_as_list(payload.get("assumptions")),
        )

    @classmethod
    def empty(cls) -> "QueryPlan":
        """Return an empty plan for fallback paths."""

        return cls()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the plan."""

        return asdict(self)
