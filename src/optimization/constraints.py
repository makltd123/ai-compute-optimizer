from __future__ import annotations


def choose_min_cost_at_quality(strategies: list[dict], target_quality: float) -> dict | None:
    eligible = [s for s in strategies if s.get("quality", 0.0) >= target_quality and s.get("cost") is not None]
    return min(eligible, key=lambda s: s["cost"]) if eligible else None


def choose_max_quality_at_budget(strategies: list[dict], budget: float) -> dict | None:
    eligible = [s for s in strategies if s.get("cost") is not None and s["cost"] <= budget]
    return max(eligible, key=lambda s: (s.get("quality", 0.0), -(s.get("cost") or 0))) if eligible else None
