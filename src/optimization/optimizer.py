from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class ModelProfile:
    name: str
    input_price: float
    output_price: float
    capability: float


@dataclass(frozen=True)
class Decision:
    task_id: str
    difficulty: float
    model: str
    estimated_cost: float


def difficulty(task: dict) -> float:
    """
    Complexity is calculated ONLY from the current task prompt.
    No historical/API data is used.
    """
    prompt = str(task.get("prompt", ""))
    text = prompt.lower()

    score = min(len(prompt) / 1800.0, 0.35)

    keywords = {
        "recursion": 0.15,
        "recursive": 0.15,
        "dynamic programming": 0.25,
        "backtracking": 0.20,
        "graph": 0.18,
        "tree": 0.12,
        "regex": 0.12,
        "parser": 0.12,
        "parse": 0.08,
        "nested": 0.08,
        "combin": 0.12,
        "permutation": 0.10,
        "binary search": 0.06,
    }

    for word, weight in keywords.items():
        if word in text:
            score += weight

    return min(score, 1.0)


def estimate_cost(
    task: dict,
    model: ModelProfile,
    max_output_tokens: int,
) -> float:
    prompt = str(task.get("prompt", ""))

    # Deterministic token estimate.
    input_tokens = max(1, ceil(len(prompt) / 4))

    return (
        input_tokens * model.input_price / 1_000_000
        + max_output_tokens * model.output_price / 1_000_000
    )


class WorkloadOptimizer:
    """
    Model-free optimizer.

    INPUT:
        current tasks
        current pricing

    DOES NOT USE:
        previous experiment results
        historical quality
        results/*.jsonl
        results/*.json
        API observations

    Objective:

        minimize total expected inference cost

    subject to:

        model capability >= task difficulty
    """

    def __init__(
        self,
        pricing: dict,
        target_quality: float = 0.90,
        max_output_tokens: int = 512,
    ):
        self.target_quality = target_quality
        self.max_output_tokens = max_output_tokens

        # These are MODEL CAPABILITY PRIORS.
        # They are fixed before the experiment and are NOT learned
        # from previous requests.
        #
        # They represent the maximum task-complexity level for which
        # the model is considered capable of meeting the target.
        self.models = [
            ModelProfile(
                name="cheap",
                input_price=pricing["cheap"]["input_per_million"],
                output_price=pricing["cheap"]["output_per_million"],
                capability=0.35,
            ),
            ModelProfile(
                name="medium",
                input_price=pricing["medium"]["input_per_million"],
                output_price=pricing["medium"]["output_per_million"],
                capability=0.65,
            ),
            ModelProfile(
                name="strong",
                input_price=pricing["strong"]["input_per_million"],
                output_price=pricing["strong"]["output_per_million"],
                capability=1.00,
            ),
        ]

    def optimize(self, tasks: list[dict]) -> list[Decision]:
        decisions = []

        for task in tasks:
            d = difficulty(task)

            candidates = [
                model
                for model in self.models
                if d <= model.capability
            ]

            # Strong is guaranteed to be a fallback.
            if not candidates:
                candidates = [self.models[-1]]

            chosen = min(
                candidates,
                key=lambda model: estimate_cost(
                    task,
                    model,
                    self.max_output_tokens,
                ),
            )

            decisions.append(
                Decision(
                    task_id=str(task["task_id"]),
                    difficulty=d,
                    model=chosen.name,
                    estimated_cost=estimate_cost(
                        task,
                        chosen,
                        self.max_output_tokens,
                    ),
                )
            )

        return decisions

    def summary(self, decisions: list[Decision]) -> dict:
        from collections import Counter

        routing = Counter(d.model for d in decisions)

        total_cost = sum(
            d.estimated_cost
            for d in decisions
        )

        return {
            "tasks": len(decisions),
            "target_quality": self.target_quality,
            "estimated_total_cost": total_cost,
            "routing": dict(routing),
        }