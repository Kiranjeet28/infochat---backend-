"""requirements-api.txt: exact pins, runtime only, consistent with the tested environment."""

from __future__ import annotations

import re
import unittest
from importlib import metadata
from pathlib import Path

LINES = [l.strip() for l in (Path(__file__).resolve().parent.parent / "requirements-api.txt").read_text(encoding="utf-8").splitlines()
         if l.strip() and not l.startswith("#")]


class RequirementsApiTests(unittest.TestCase):
    def test_every_line_is_an_exact_pin(self):
        for line in LINES:
            self.assertRegex(line, r"^[A-Za-z0-9_.\-]+==[A-Za-z0-9_.\-+!]+$", line)

    def test_runtime_packages_present_and_dev_tools_absent(self):
        names = {l.split("==")[0].lower().replace("_", "-") for l in LINES}
        for required in ("fastapi", "uvicorn", "pydantic", "requests", "langchain-core", "langchain-groq", "groq", "fastembed", "python-dotenv"):
            self.assertIn(required, names)
        for forbidden in ("pytest", "playwright", "selenium", "ruff", "black", "mypy", "pdfplumber", "beautifulsoup4", "sentence-transformers", "torch"):
            self.assertNotIn(forbidden, names)

    def test_pins_match_the_tested_environment(self):
        for line in LINES:
            name, version = line.split("==")
            try:
                installed = metadata.version(name)
            except metadata.PackageNotFoundError:
                continue  # platform-specific package not present in this environment
            self.assertEqual(installed, version, name)


if __name__ == "__main__":
    unittest.main()
