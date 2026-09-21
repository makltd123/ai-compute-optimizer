from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt


def quality_vs_cost(rows: list[dict], output: str | Path) -> None:
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    xs = [r["cost"] for r in rows if r.get("cost") is not None]
    ys = [r["quality"] for r in rows if r.get("cost") is not None]
    labels = [r["strategy"] for r in rows if r.get("cost") is not None]
    plt.figure()
    plt.scatter(xs, ys)
    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y))
    plt.xlabel("Theoretical cost (USD)")
    plt.ylabel("Quality")
    plt.title("Quality vs theoretical cost")
    plt.tight_layout()
    plt.savefig(p, dpi=160)
    plt.close()


def saving_vs_quality_drop(rows: list[dict], strong_quality: float, strong_cost: float, output: str | Path) -> None:
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    xs, ys, labels = [], [], []
    for r in rows:
        if r.get("cost") is None:
            continue
        xs.append(1 - r["cost"] / strong_cost if strong_cost else math.nan)
        ys.append(strong_quality - r["quality"])
        labels.append(r["strategy"])
    plt.figure()
    plt.scatter(xs, ys)
    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y))
    plt.xlabel("Cost saving vs Strong")
    plt.ylabel("Quality drop vs Strong")
    plt.title("Cost saving vs quality drop")
    plt.tight_layout()
    plt.savefig(p, dpi=160)
    plt.close()
