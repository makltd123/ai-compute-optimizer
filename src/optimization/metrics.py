from __future__ import annotations

from collections import defaultdict
from statistics import mean


def summarize(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in records:
        groups[r["model"]].append(r)
    out = []
    for model, rows in groups.items():
        quality = mean(float(r.get("quality") or 0) for r in rows)
        costs = [r["estimated_cost"] for r in rows if r.get("estimated_cost") is not None]
        latencies = [r["latency_seconds"] for r in rows if r.get("latency_seconds") is not None]
        out.append({
            "strategy": model,
            "quality": quality,
            "cost": sum(costs) if costs else None,
            "average_cost_per_task": mean(costs) if costs else None,
            "latency": mean(latencies) if latencies else None,
            "tasks": len(rows),
        })
    return out


def cost_ratio(adaptive_cost: float, strong_cost: float) -> float:
    return adaptive_cost / strong_cost if strong_cost else float("nan")


def savings(adaptive_cost: float, strong_cost: float) -> float:
    return 1.0 - cost_ratio(adaptive_cost, strong_cost)
