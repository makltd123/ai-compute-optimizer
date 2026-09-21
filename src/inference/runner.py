from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from src.cost.model import Pricing, estimated_cost
from src.providers.base import GenerationResult, ModelProvider


class InferenceRunner:
    def __init__(
        self,
        provider: ModelProvider,
        model_registry: dict[str, dict[str, Any]],
        pricing: dict[str, Pricing],
        log_path: str | Path,
        *,
        max_attempts: int = 4,
        min_wait_seconds: float = 1.0,
        max_wait_seconds: float = 20.0,
    ):
        self.provider = provider
        self.model_registry = model_registry
        self.pricing = pricing
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max_attempts
        self.min_wait_seconds = min_wait_seconds
        self.max_wait_seconds = max_wait_seconds

    def run_one(
        self,
        *,
        experiment_id: str,
        task: dict[str, Any],
        logical_model: str,
        prompt: str,
        generation: dict[str, Any] | None = None,
        evaluator: Callable[[dict[str, Any], str], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        started = time.time()
        model_id = self.model_registry[logical_model]["model_id"]
        result: GenerationResult | None = None
        last_error = None
        failure_type = None
        attempts = 0

        for attempts in range(1, self.max_attempts + 1):
            try:
                result = self.provider.generate(prompt, model_id=model_id, **(generation or {}))
                break
            except Exception as exc:  # provider-specific SDK exceptions vary by version
                last_error = str(exc)
                failure_type = self._classify_api_error(exc)
                if failure_type in {"rate_limit", "timeout", "api_failure"} and attempts < self.max_attempts:
                    sleep_seconds = min(self.max_wait_seconds, self.min_wait_seconds * (2 ** (attempts - 1)))
                    time.sleep(sleep_seconds)
                    continue
                break

        if result is None:
            record = {
                "experiment_id": experiment_id,
                "task_id": str(task["task_id"]),
                "model": logical_model,
                "model_id": model_id,
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_seconds": time.time() - started,
                "estimated_cost": 0.0,
                "success": False,
                "quality": None,
                "evaluation": {},
                "prompt": prompt,
                "generation": generation or {},
                "timestamp": started,
                "attempts": attempts,
                "failure_type": failure_type or "api_failure",
                "error_message": last_error,
            }
            self._append_jsonl(record)
            return record

        eval_result = evaluator(task, result.text) if evaluator else {}
        price = self.pricing.get(logical_model)
        cost = estimated_cost(result.input_tokens, result.output_tokens, price) if price else None
        record = {
            "experiment_id": experiment_id,
            "task_id": str(task["task_id"]),
            "model": logical_model,
            "model_id": model_id,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_seconds": result.metadata.get("latency_seconds"),
            "estimated_cost": cost,
            "success": bool(eval_result.get("success")) if eval_result else None,
            "quality": float(eval_result.get("quality", 0.0)) if eval_result else None,
            "evaluation": eval_result,
            "prompt": prompt,
            "generation": generation or {},
            "timestamp": started,
            "attempts": attempts,
            "failure_type": None,
            "error_message": None,
        }
        self._append_jsonl(record)
        return record

    @staticmethod
    def _classify_api_error(exc: Exception) -> str:
        message = str(exc).lower()
        name = exc.__class__.__name__.lower()
        if "429" in message or "rate" in message or "ratelimit" in name:
            return "rate_limit"
        if "timeout" in message or "timed out" in message or "timeout" in name:
            return "timeout"
        return "api_failure"

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
