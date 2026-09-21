from __future__ import annotations

from src.utils.config import load_yaml


def main() -> int:
    models = load_yaml("configs/models.yaml")["models"]
    pricing = load_yaml("configs/pricing.yaml")["pricing"]
    names = [m["name"] for m in models]
    assert names == ["cheap", "medium", "strong"], names
    assert all(pricing[n]["input_per_million"] is not None for n in names)
    assert all(pricing[n]["output_per_million"] is not None for n in names)
    print("Config validation: OK")
    for m in models:
        p = pricing[m["name"]]
        print(f"{m['name']:>6} -> {m['model_id']:<24} ${p['input_per_million']}/{p['output_per_million']} per 1M")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
