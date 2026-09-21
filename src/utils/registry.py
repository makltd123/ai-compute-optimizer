from __future__ import annotations

from src.providers.mistral import MistralProvider
from src.utils.config import load_yaml


def main() -> int:
    config = load_yaml("configs/models.yaml")
    configured = {item["model_id"]: item for item in config.get("models", []) if item.get("enabled", True)}
    provider = MistralProvider()
    available = {item.get("id") for item in provider.list_models()}
    print("Configured models:")
    for model_id, item in configured.items():
        status = "AVAILABLE" if model_id in available else "NOT_RETURNED"
        print(f"  {item['name']}: {model_id} [{status}]")
    print(f"API returned {len(available)} models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
