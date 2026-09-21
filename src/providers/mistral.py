from __future__ import annotations

import os
import time
from typing import Any

from .base import GenerationResult


class MistralProvider:
    """Thin adapter over the official Mistral Python SDK.

    Benchmark/optimizer code only depends on this class's generate() contract.
    """

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from mistralai.client import Mistral
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the `mistralai` package first.") from exc

        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
            raise ValueError("MISTRAL_API_KEY is not set")
        self.client = Mistral(api_key=key)

    def list_models(self) -> list[dict[str, Any]]:
        response = self.client.models.list()
        data = getattr(response, "data", response)
        return [self._to_dict(item) for item in (data or [])]

    def generate(self, prompt: str, *, model_id: str, **kwargs: Any) -> GenerationResult:
        start = time.perf_counter()
        response = self.client.chat.complete(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        latency = time.perf_counter() - start
        choice = response.choices[0]
        text = getattr(getattr(choice, "message", None), "content", "") or ""
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return GenerationResult(
            text=text,
            model=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw=response,
            metadata={
                "latency_seconds": latency,
                "request_model": getattr(response, "model", model_id),
            },
        )

    @staticmethod
    def _to_dict(obj: Any) -> dict[str, Any]:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if isinstance(obj, dict):
            return obj
        return {"id": getattr(obj, "id", None), "object": getattr(obj, "object", None)}
