from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pricing:
    input_per_million: float
    output_per_million: float


def estimated_cost(input_tokens: int, output_tokens: int, pricing: Pricing) -> float:
    return (
        input_tokens * pricing.input_per_million / 1_000_000
        + output_tokens * pricing.output_per_million / 1_000_000
    )
