from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class GenerationResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelProvider(Protocol):
    def generate(self, prompt: str, *, model_id: str, **kwargs: Any) -> GenerationResult:
        ...
