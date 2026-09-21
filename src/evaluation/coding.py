from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def extract_code(text: str) -> str:
    fenced = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    code = fenced[0] if fenced else text
    return code.strip()


def evaluate_humaneval(task: dict, generated: str, timeout_seconds: int = 5) -> dict:
    prompt = task["prompt"]
    completion = extract_code(generated)
    candidate = prompt + "\n" + completion
    try:
        ast.parse(candidate)
    except SyntaxError as exc:
        return {"success": False, "quality": 0.0, "failure_type": "syntax_error", "error": str(exc)}

    tests = task.get("test") or task.get("tests") or ""
    entry_point = task.get("entry_point")
    harness = candidate + "\n" + tests + f"\ncheck({entry_point})\n" if entry_point and tests else candidate + "\n" + tests
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(harness, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {"success": False, "quality": 0.0, "failure_type": "timeout"}
    success = proc.returncode == 0
    return {
        "success": success,
        "quality": 1.0 if success else 0.0,
        "failure_type": None if success else "test_failure",
        "stderr": proc.stderr[-2000:],
    }
