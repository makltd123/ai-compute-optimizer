from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.benchmarks import load_humaneval
from src.experiments import build_runner
from src.optimization.optimizer import WorkloadOptimizer
from src.utils.config import load_yaml
from src.utils.io import write_json


def main() -> int:
    model_cfg = load_yaml("configs/models.yaml")
    pricing_cfg = load_yaml("configs/pricing.yaml")
    exp_cfg = load_yaml("configs/experiments.yaml")["experiments"]["exp004_router"]

    tasks = load_humaneval(
        exp_cfg.get("task_limit", 164)
    )

    generation = exp_cfg.get(
        "generation",
        {
            "temperature": 0,
            "max_tokens": 512,
        },
    )

    target_quality = float(
        exp_cfg.get("target_quality", 0.90)
    )

    print("AI Compute Optimizer — FINAL")
    print("============================")
    print()
    print(f"Tasks: {len(tasks)}")
    print()
    print("Data used for optimization:")
    print("  Current task prompts: YES")
    print("  Pricing config:       YES")
    print("  Historical results:   NO")
    print("  Previous experiment records: NO")
    print("  Previous API results: NO")
    print()

    optimizer = WorkloadOptimizer(
        pricing=pricing_cfg["pricing"],
        target_quality=target_quality,
        max_output_tokens=generation["max_tokens"],
    )

    decisions = optimizer.optimize(tasks)
    summary = optimizer.summary(decisions)

    print("Optimization:")
    print("  Objective: minimize total inference cost")
    print("  Constraint: per-task model capability")
    print()

    print(
        f"Estimated total cost: "
        f"${summary['estimated_total_cost']:.6f}"
    )
    print()

    print("Routing:")

    routing = Counter(
        d.model for d in decisions
    )

    for model in ("cheap", "medium", "strong"):
        print(
            f"  {model}: "
            f"{routing.get(model, 0)}"
        )

    plan = {
        "experiment": "exp004_router_final",
        "method": "model_free_per_task_cost_optimization",
        "historical_data_used": False,
        "training_data_used": False,
        "tasks": len(tasks),
        "target_quality": target_quality,
        "estimated_total_cost": summary["estimated_total_cost"],
        "routing": dict(routing),
        "decisions": [
            {
                "task_id": d.task_id,
                "difficulty": d.difficulty,
                "model": d.model,
                "estimated_cost": d.estimated_cost,
            }
            for d in decisions
        ],
    }

    write_json(
        "results/exp004_router_plan.json",
        plan,
    )

    print()
    print("Routing plan saved.")
    print()

    # ---------------------------------------------------------
    # REAL API EXECUTION
    # ---------------------------------------------------------

    print("Starting REAL API execution...")
    print()

    from src.providers.mistral import MistralProvider
    from src.evaluation.coding import evaluate_humaneval

    provider = MistralProvider()

    output_path = Path(
        "results/exp004_router_final_requests.jsonl"
    )

    runner = build_runner(
        provider,
        model_cfg,
        pricing_cfg,
        output_path,
    )

    task_map = {
        str(task["task_id"]): task
        for task in tasks
    }

    results = []

    for index, decision in enumerate(
        decisions,
        start=1,
    ):
        task = task_map[decision.task_id]

        print(
            f"[{index}/{len(decisions)}] "
            f"{decision.task_id} -> {decision.model}"
        )

        result = runner.run_one(
            experiment_id="exp004_router_final",
            task=task,
            logical_model=decision.model,
            prompt=task["prompt"],
            generation=generation,
            evaluator=lambda t, text: evaluate_humaneval(
                t,
                text,
            ),
        )

        results.append(result)

    actual_quality = (
        sum(
            float(r.get("quality") or 0.0)
            for r in results
        )
        / len(results)
    )

    actual_cost = sum(
        float(r.get("estimated_cost") or 0.0)
        for r in results
    )

    actual_routing = Counter(
        r["model"]
        for r in results
    )

    final = {
        "experiment": "exp004_router_final",
        "tasks": len(results),
        "target_quality": target_quality,

        # Predictions came ONLY from the current workload
        # and fixed model/pricing priors.
        "predicted_cost": summary["estimated_total_cost"],

        # These are measurements from THIS run.
        "actual_quality": actual_quality,
        "actual_cost": actual_cost,

        "routing": dict(actual_routing),

        "historical_data_used": False,
        "training_data_used": False,
    }

    write_json(
        "results/exp004_router_final_summary.json",
        final,
    )

    print()
    print("============================")
    print("FINAL RESULT")
    print("============================")
    print()
    print(f"Tasks:             {len(results)}")
    print(f"Target quality:    {target_quality:.1%}")
    print(f"Actual quality:    {actual_quality:.1%}")
    print(f"Predicted cost:    ${summary['estimated_total_cost']:.6f}")
    print(f"Actual cost:       ${actual_cost:.6f}")
    print()
    print("Actual routing:")

    for model in ("cheap", "medium", "strong"):
        print(
            f"  {model}: "
            f"{actual_routing.get(model, 0)}"
        )

    print()
    print("FINAL RUN COMPLETE.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())