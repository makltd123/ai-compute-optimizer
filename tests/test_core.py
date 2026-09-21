from pathlib import Path

from src.cost.model import Pricing, estimated_cost
from src.optimization.constraints import choose_max_quality_at_budget, choose_min_cost_at_quality
from src.optimization.oracle import oracle_choice
from src.routing.heuristic import route


def test_cost():
    assert estimated_cost(1_000_000, 1_000_000, Pricing(1.0, 2.0)) == 3.0


def test_oracle_picks_cheapest_success():
    rows = [
        {"model": "cheap", "success": False, "estimated_cost": 0.001},
        {"model": "medium", "success": True, "estimated_cost": 0.004},
        {"model": "strong", "success": True, "estimated_cost": 0.01},
    ]
    assert oracle_choice(rows)["model"] == "medium"


def test_constraints():
    rows = [
        {"strategy": "a", "quality": 0.90, "cost": 1},
        {"strategy": "b", "quality": 0.95, "cost": 2},
    ]
    assert choose_min_cost_at_quality(rows, 0.94)["strategy"] == "b"
    assert choose_max_quality_at_budget(rows, 1)["strategy"] == "a"


def test_router_is_deterministic():
    easy = {"prompt": "Return the sum of two integers."}
    hard = {"prompt": "Solve a dynamic programming graph optimization problem with recursion and tree traversal."}
    assert route(easy) == "cheap"
    assert route(hard) == "strong"
