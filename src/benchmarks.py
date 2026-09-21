from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    category: str = "coding"
    difficulty: str | None = None
    payload: dict[str, Any] | None = None


def load_humaneval(limit: int | None = None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the optional benchmark dependency: pip install -e '.[bench]'") from exc
    ds = load_dataset("openai/openai_humaneval", split="test")
    rows: list[dict[str, Any]] = []
    for row in ds:
        rows.append({
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "test": row["test"],
            "entry_point": row["entry_point"],
            "category": "coding",
        })
        if limit and len(rows) >= limit:
            break
    return rows
