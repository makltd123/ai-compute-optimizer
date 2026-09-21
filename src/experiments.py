from __future__ import annotations

from pathlib import Path
from typing import Any

from src.cost.model import Pricing
from src.evaluation.coding import evaluate_humaneval
from src.inference.runner import InferenceRunner
from src.optimization.metrics import summarize
from src.optimization.oracle import build_oracle
from src.routing.heuristic import route


def build_runner(provider, model_cfg: dict[str, Any], pricing_cfg: dict[str, Any], log_path: str | Path):
    registry = {item["name"]: item for item in model_cfg["models"] if item.get("enabled", True)}
    pricing = {}
    for name, value in pricing_cfg.get("pricing", {}).items():
        if value.get("input_per_million") is not None and value.get("output_per_million") is not None:
            pricing[name] = Pricing(value["input_per_million"], value["output_per_million"])
    return InferenceRunner(provider, registry, pricing, log_path)


def run_strategy(runner: InferenceRunner, experiment_id: str, tasks: list[dict], logical_model: str, generation: dict) -> list[dict]:
    results = []
    for task in tasks:
        results.append(runner.run_one(
            experiment_id=experiment_id,
            task=task,
            logical_model=logical_model,
            prompt=task["prompt"],
            generation=generation,
            evaluator=lambda t, text: evaluate_humaneval(t, text),
        ))
    return results


def run_heuristic(runner: InferenceRunner, experiment_id: str, tasks: list[dict], generation: dict) -> list[dict]:
    results = []
    for task in tasks:
        chosen = route(task)
        results.append(runner.run_one(
            experiment_id=experiment_id,
            task=task,
            logical_model=chosen,
            prompt=task["prompt"],
            generation=generation,
            evaluator=lambda t, text: evaluate_humaneval(t, text),
        ))
    return results


def build_oracle_summary(all_baseline_results: list[dict]) -> list[dict]:
    return build_oracle(all_baseline_results)


def summarize_results(records: list[dict]) -> list[dict]:
    return summarize(records)
