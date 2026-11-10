"""Static checks of the deployment files (Docker is not required to run these)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


class DockerfileTests(unittest.TestCase):
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    def test_base_user_install_and_start(self):
        self.assertRegex(self.text, r"(?m)^FROM python:3\.12-slim")
        self.assertRegex(self.text, r"(?m)^USER app")
        self.assertNotRegex(self.text, r"(?m)^USER root")
        self.assertIn("pip install -r requirements-api.txt", self.text)
        self.assertIn("--host 0.0.0.0 --port ${PORT:-8000}", self.text)
        self.assertIn("uvicorn api:app", self.text)

    def test_model_is_downloaded_at_build_and_never_at_runtime(self):
        self.assertIn("create_embedding_model(allow_download=True)", self.text)
        code = " ".join(l for l in self.text.splitlines() if not l.lstrip().startswith("#"))
        self.assertNotRegex(code, r"EMBEDDING_ALLOW_DOWNLOAD")

    def test_no_secret_is_copied_or_baked_in(self):
        self.assertNotRegex(self.text, r"(?im)^(COPY|ADD)\s+.*\.env")
        self.assertNotRegex(self.text, r"(?im)^ENV\s+.*(KEY|TOKEN|SECRET)")
        self.assertNotRegex(self.text, r"gsk_|eyJ[A-Za-z0-9_-]{20,}|sb_secret")

    def test_requirements_python_version_is_compatible(self):
        # numpy in requirements-api.txt needs Python >= 3.12
        pins = (ROOT / "requirements-api.txt").read_text(encoding="utf-8")
        self.assertRegex(pins, r"(?m)^numpy==")
        self.assertIn("python:3.12", self.text)


class DockerignoreTests(unittest.TestCase):
    lines = {l.strip() for l in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")}

    def test_secrets_and_local_state_are_excluded(self):
        for entry in (".env", ".env.*", "qa", "qa_blackbox", "results", "logs", ".venv", "**/.venv", "models_cache", "qa/backup", ".git"):
            self.assertIn(entry, self.lines)
        self.assertIn("!.env.example", self.lines)


class RenderBlueprintTests(unittest.TestCase):
    text = (ROOT / "render.yaml").read_text(encoding="utf-8")
    service = yaml.safe_load(text)["services"][0]

    def test_service_settings(self):
        self.assertEqual(self.service["runtime"], "docker")
        self.assertEqual(self.service["healthCheckPath"], "/health")
        self.assertIs(self.service["autoDeploy"], False)

    def test_secrets_are_not_in_the_file(self):
        env = {e["key"]: e for e in self.service["envVars"]}
        for name in ("GROQ_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "CORS_ALLOWED_ORIGINS"):
            self.assertIs(env[name].get("sync"), False)
            self.assertNotIn("value", env[name])
        self.assertNotRegex(self.text, r"gsk_|eyJ[A-Za-z0-9_-]{20,}|sb_secret")
        self.assertEqual(env["TRUST_PROXY_HEADERS"]["value"], "1")
        self.assertEqual(env["GROQ_ENABLED"]["value"], "true")

    def test_every_blueprint_variable_is_documented_in_env_example(self):
        documented = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", (ROOT / ".env.example").read_text(encoding="utf-8"), re.M))
        for entry in self.service["envVars"]:
            self.assertIn(entry["key"], documented)


if __name__ == "__main__":
    unittest.main()
