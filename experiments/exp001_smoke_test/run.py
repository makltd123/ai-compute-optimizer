from __future__ import annotations

import json

from src.benchmarks import load_humaneval
from src.experiments import build_runner, run_heuristic, summarize_results
from src.providers.mistral import MistralProvider
from src.utils.config import load_yaml
from src.utils.io import write_json


def main() -> int:
    model_cfg = load_yaml("configs/models.yaml")
    pricing_cfg = load_yaml("configs/pricing.yaml")
    exp_cfg = load_yaml("configs/experiments.yaml")["experiments"]["exp001_smoke_test"]

    tasks = load_humaneval(exp_cfg["task_limit"])

    provider = MistralProvider()

    runner = build_runner(
        provider,
        model_cfg,
        pricing_cfg,
        "results/exp001_heuristic_requests.jsonl",
    )

    results = run_heuristic(
        runner,
        "exp001_heuristic",
        tasks,
        exp_cfg["generation"],
    )

    summary = summarize_results(results)

    write_json(
        "results/exp001_heuristic_summary.json",
        summary,
    )

    print("AI Compute Optimizer")
    print("=" * 30)
    print(json.dumps(summary, indent=2))

    print("\nRouting:")
    counts = {}
    for record in results:
        model = record.get("model")
        counts[model] = counts.get(model, 0) + 1

    for model, count in counts.items():
        print(f"  {model}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())