from __future__ import annotations


def difficulty_score(task: dict) -> float:
    prompt = task.get("prompt", "")
    score = min(len(prompt) / 2200.0, 1.0)
    keywords = {
        "recursion": 0.18,
        "dynamic programming": 0.25,
        "graph": 0.18,
        "tree": 0.12,
        "regex": 0.10,
        "sorting": 0.08,
        "parse": 0.08,
    }
    lower = prompt.lower()
    score += sum(weight for word, weight in keywords.items() if word in lower)
    return min(score, 1.0)


def route(task: dict) -> str:
    score = difficulty_score(task)
    if score < 0.34:
        return "cheap"
    if score < 0.68:
        return "medium"
    return "strong"
