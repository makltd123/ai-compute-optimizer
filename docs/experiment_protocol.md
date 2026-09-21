# Experiment 001 protocol

## Objective

Test whether adaptive model selection can reduce theoretical inference cost while maintaining quality on a reproducible public coding benchmark.

## Locked comparison set

- Cheap: Mistral Small 4
- Medium: Mistral Large 3
- Strong: Mistral Medium 3.5

The registry uses current Mistral aliases and records the model returned by the API. For a publishable run, pin versioned IDs when the account exposes them and store the exact `/v1/models` response.

## Conditions

- temperature = 0
- common `max_tokens` budget unless the model's context constraints require a documented change
- same prompt text and evaluator for every model
- retry/API failures are separated from model failures
- successful requests are never repeated on resume

## Primary outputs

1. raw JSONL request log
2. CSV strategy summary
3. Quality vs theoretical cost plot
4. Cost saving vs quality drop plot
5. raw benchmark/evaluator version metadata

## Decision rule

The first research signal is an adaptive policy that moves toward the quality/cost Pareto frontier and improves on fixed routing. The predefined practical threshold is approximately 20–30%+ theoretical cost reduction with no more than 1 percentage point quality loss. This is a decision threshold, not a statistical-significance claim.
