from pathlib import Path

from src.cost.model import Pricing
from src.inference.runner import InferenceRunner
from src.providers.base import GenerationResult


class FakeProvider:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, *, model_id, **kwargs):
        self.calls += 1
        return GenerationResult(
            text="return 42",
            model=model_id,
            input_tokens=10,
            output_tokens=5,
            metadata={"latency_seconds": 0.01},
        )


def test_runner_logs_and_evaluates(tmp_path: Path):
    log = tmp_path / "results.jsonl"
    runner = InferenceRunner(
        FakeProvider(),
        {"cheap": {"model_id": "fake"}},
        {"cheap": Pricing(1, 2)},
        log,
    )
    result = runner.run_one(
        experiment_id="x",
        task={"task_id": "1"},
        logical_model="cheap",
        prompt="x",
        evaluator=lambda task, text: {"success": True, "quality": 1.0},
    )
    assert result["estimated_cost"] == 0.00002
    assert result["success"] is True
    assert log.exists()
