"""Offline checker: import each package in its own subprocess and classify the result.

Run with the venv interpreter:
    .\\ingestion\\.venv\\Scripts\\python.exe ingestion\\check_native_imports.py

Exit code 0 only if no package is BLOCKED or MISSING.
"""

from __future__ import annotations

import subprocess
import sys

PACKAGES = [
    "uuid_utils",
    "langchain_core",
    "langchain_text_splitters",
    "langchain_groq",
    "fastembed",
    "onnxruntime",
    "tokenizers",
    "numpy",
    "pydantic_core",
    "orjson",
    "jiter",
    "tiktoken",
    "httpx",
    "requests",
    "supabase",
    "groq",
    "fastapi",
    "starlette",
    "uvicorn",
    "dotenv",
    "yaml",
]

BLOCKED_MARKERS = ("application control", "blocked", "dll load failed")
MISSING_MARKERS = ("modulenotfounderror", "no module named")


def classify(returncode: int, output: str) -> str:
    if returncode == 0:
        return "OK"
    lowered = output.lower()
    if any(marker in lowered for marker in BLOCKED_MARKERS):
        return "BLOCKED"
    if any(marker in lowered for marker in MISSING_MARKERS):
        return "MISSING"
    return "OTHER"


def last_line(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def check_package(name: str, code: str | None = None, timeout: int = 120) -> tuple[str, str]:
    """Import ``name`` in a fresh interpreter; return (status, last error line)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code or f"import {name}"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "OTHER", f"timed out after {timeout}s"
    output = (proc.stderr or "") + (proc.stdout or "")
    status = classify(proc.returncode, output)
    return status, "" if status == "OK" else last_line(output)


def main() -> int:
    print(f"interpreter: {sys.executable}")
    rows = [(name, *check_package(name)) for name in PACKAGES]
    width = max(len(name) for name in PACKAGES)
    print(f"{'package'.ljust(width)} | status  | last error line")
    print(f"{'-' * width}-+---------+----------------")
    for name, status, error in rows:
        print(f"{name.ljust(width)} | {status.ljust(7)} | {error}")
    bad = [name for name, status, _ in rows if status in {"BLOCKED", "MISSING"}]
    if bad:
        print(f"\nBLOCKED or MISSING: {', '.join(bad)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
