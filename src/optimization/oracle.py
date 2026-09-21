from __future__ import annotations


def oracle_choice(records: list[dict], preference=("cheap", "medium", "strong")) -> dict | None:
    eligible = [r for r in records if r.get("success")]
    if not eligible:
        return None
    rank = {name: i for i, name in enumerate(preference)}
    return min(eligible, key=lambda r: (r.get("estimated_cost") if r.get("estimated_cost") is not None else float("inf"), rank.get(r["model"], 999)))


def build_oracle(records: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record["task_id"]), []).append(record)
    return [choice for task_records in grouped.values() if (choice := oracle_choice(task_records))]
