from pathlib import Path

from src.cost.model import Pricing
from src.inference.runner import InferenceRunner
from src.optimization.metrics import summarize
from src.optimization.oracle import build_oracle
from src.routing.heuristic import route
from src.providers.base import GenerationResult


class FakeProvider:
    def generate(self, prompt, *, model_id, **kwargs):
        # Deterministic stand-in: short prompts succeed on all models, hard prompts
        # require the strongest tier. This exercises routing/logging/oracle only.
        hard = "hard" in prompt.lower()
        success = (model_id != "fake-cheap" and hard) or not hard
        return GenerationResult(
            text="def f():\n    return 1",
            model=model_id,
            input_tokens=20,
            output_tokens=10,
            metadata={"latency_seconds": 0.01},
        )


def test_end_to_end(tmp_path: Path):
    runner = InferenceRunner(
        FakeProvider(),
        {
            "cheap": {"model_id": "fake-cheap"},
            "medium": {"model_id": "fake-medium"},
            "strong": {"model_id": "fake-strong"},
        },
        {
            "cheap": Pricing(0.15, 0.6),
            "medium": Pricing(0.5, 1.5),
            "strong": Pricing(1.5, 7.5),
        },
        tmp_path / "results.jsonl",
    )
    tasks = [
        {"task_id": "easy", "prompt": "easy"},
        {"task_id": "hard", "prompt": "hard graph"},
    ]
    records = []
    for task in tasks:
        chosen = "strong" if task["task_id"] == "hard" else "cheap"
        records.append(runner.run_one(
            experiment_id="dry",
            task=task,
            logical_model=chosen,
            prompt=task["prompt"],
            evaluator=lambda t, text: {"success": True, "quality": 1.0},
        ))
    summary = summarize(records)
    assert {r["strategy"] for r in summary} == {"cheap", "strong"}
    oracle = build_oracle(records)
    assert len(oracle) == 2
    assert route(tasks[0]) == "cheap"
