# AI Compute Optimizer

Research MVP for the hypothesis: an external decision/optimization layer can route heterogeneous AI workload across models with different cost/quality profiles and reduce theoretical inference cost while maintaining a target quality level.

## Scope of this first implementation

- Mistral API adapter behind a provider abstraction.
- External model registry and pricing configuration.
- Token/cost/latency logging per request.
- HumanEval-compatible benchmark loader (with optional `datasets` dependency).
- Python unit-test evaluator.
- Fixed baselines: Cheap / Medium / Strong.
- Offline Oracle.
- First heuristic router: easy → cheap, medium → medium, hard → strong.
- Constraint utilities for quality targets and budget.
- Reproducible experiment metadata and resumable JSONL results.
- Plot/report generation.

## Research integrity

The implementation does not declare the hypothesis proven. It produces raw results, metrics, and plots. Final scientific/commercial interpretation remains with the researcher.

## Model tiers

| Tier | Model | USD per 1M tokens (input / output) |
|---|---|---|
| cheap | Mistral Small 4 | 0.15 / 0.60 |
| medium | Mistral Large 3 | 0.50 / 1.50 |
| strong | Mistral Medium 3.5 | 1.50 / 7.50 |

Tier names follow the price per token, not the model names: `strong` is Mistral Medium 3.5 because it is the most expensive of the three.

## Results

HumanEval, pass@1 from the benchmark's unit tests, `temperature=0`, `max_tokens=512`. Costs are theoretical, computed from token counts and the prices above. Summaries live in `results/*_summary.json`; raw per-request logs are not committed.

### Fixed baselines: first 60 tasks

| Strategy | pass@1 | Cost, USD | Avg latency |
|---|---|---|---|
| cheap | 95.0% | 0.0169 | 3.1 s |
| medium | 58.3%* | 0.0257 | 7.7 s |
| strong | 95.0% | 0.2268 | 5.5 s |

\* 24 of the 25 failed medium requests were HTTP 429 (rate limit) errors that were scored as failed tasks. The model returned a completion for 36 tasks and 35 of them passed, so this number measures rate limiting rather than model quality. It needs a re-run.

### Routers: all 164 tasks, quality target 90%

| Strategy | Routing (cheap / medium / strong) | pass@1 | Cost, USD |
|---|---|---|---|
| Heuristic router | 141 / 22 / 1 | 89.0% | 0.0665 |
| Cost optimizer, v1 | 164 / 0 / 0 | 88.4% | 0.0506 |
| Cost optimizer, final | 156 / 8 / 0 | 87.2% | 0.0547 |

- None of the routers reached the 90% target.
- None of them beat simply sending every task to the cheap model.
- Run-to-run noise is about the size of the differences between strategies: across the two optimizer runs, 6 of the 156 tasks sent to the cheap model both times changed outcome even at `temperature=0`.
- Mistral Small 4 already solves most of HumanEval, so this benchmark leaves little room for routing. Testing the hypothesis likely needs a harder or more mixed workload.

## Current Mistral integration

The adapter uses the official `mistralai` Python SDK and `client.chat.complete(...)`. Model availability can be checked through the account's `/v1/models` endpoint. Keep `MISTRAL_API_KEY` in the environment only.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,bench]'
cp .env.example .env
export MISTRAL_API_KEY='...'
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[test,bench]"
$env:MISTRAL_API_KEY = "..."
```

Verify configured models:

```bash
python -m src.utils.registry
```

Run local tests:

```bash
pytest -q
```

Run Experiment 001 (requires API key and benchmark dependency):

```bash
python -m experiments.exp001_smoke_test.run
```

Run Experiment 004, the cost optimizer (requires API key):

```bash
python -m experiments.exp004_router.run
```

## Important before external reporting

`configs/pricing.yaml` holds standard (non-batch) Mistral API prices verified on 2026-08-12. Prices change, and the `*-latest` aliases can point to newer models over time, so re-check both before quoting cost numbers and record the model IDs the API actually returned.

## License

[MIT](LICENSE)
