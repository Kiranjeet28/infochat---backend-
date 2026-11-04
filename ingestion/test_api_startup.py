"""Offline API behavior and startup tests; no provider or database calls."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
INGESTION = ROOT / "ingestion"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MISSING = [
    package for package in ("fastapi", "httpx")
    if importlib.util.find_spec(package) is None
]

if not MISSING:
    from fastapi.testclient import TestClient
    import api
    from ingestion import embedding_config, retrieve as retrieve_module


@unittest.skipIf(MISSING, "install fastapi, uvicorn, httpx")
class ApiStartupTests(unittest.TestCase):
    def setUp(self):
        self.original_cwd = Path.cwd()

    def tearDown(self):
        os.chdir(self.original_cwd)

    def _working_directories(self):
        return (ROOT, INGESTION)

    def _client(self, *, chain=None, groq_enabled="true", gemini_enabled="false"):
        environment = {"GROQ_ENABLED": groq_enabled, "GEMINI_ENABLED": gemini_enabled}
        return (
            patch.dict(os.environ, environment),
            patch.object(api, "initialize_embedding_model", return_value=object()),
            patch.object(api, "_answer_chain", chain),
        )

    def test_embedding_model_is_requested_from_local_cache_only(self):
        with patch.object(embedding_config, "TextEmbedding") as text_embedding:
            embedding_config.create_embedding_model()
        self.assertEqual(text_embedding.call_args.kwargs["model_name"], embedding_config.EMBEDDING_MODEL)
        self.assertEqual(text_embedding.call_args.kwargs["cache_dir"], str(embedding_config.EMBEDDING_CACHE_DIR))
        self.assertIs(text_embedding.call_args.kwargs["local_files_only"], True)

    def test_missing_embedding_cache_fails_during_startup_clearly(self):
        with patch.dict(os.environ, {"GROQ_ENABLED": "true", "GEMINI_ENABLED": "false"}), \
             patch.object(retrieve_module, "_embedding_model", None), \
             patch.object(retrieve_module, "create_embedding_model", side_effect=OSError("offline")):
            with self.assertRaisesRegex(RuntimeError, "FastEmbed model cache is missing or unusable"):
                with TestClient(api.app):
                    self.fail("startup unexpectedly succeeded without the local model")

    def test_startup_fails_when_no_provider_is_enabled(self):
        for directory in self._working_directories():
            with self.subTest(cwd=directory), patch.dict(
                os.environ, {"GROQ_ENABLED": "false", "GEMINI_ENABLED": "false"}
            ):
                os.chdir(directory)
                with self.assertRaisesRegex(RuntimeError, "^no model provider configured$"):
                    with TestClient(api.app):
                        self.fail("startup unexpectedly succeeded")

    def test_enabled_provider_allows_health_from_both_working_directories(self):
        for directory in self._working_directories():
            with self.subTest(cwd=directory):
                os.chdir(directory)
                env_patch, model_patch, chain_patch = self._client()
                with env_patch, model_patch, chain_patch, TestClient(api.app) as client:
                    response = client.get("/health")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"status": "ok"})

    def test_normal_answer_has_ask_response_shape_from_both_directories(self):
        class FakeChain:
            async def ainvoke(self, _values):
                return {
                    "answer": "The exam starts on 2 September [1].",
                    "sources": [{"title": "Exam notice", "source_url": "https://example.test"}],
                    "status": "answered",
                }

        for directory in self._working_directories():
            with self.subTest(cwd=directory):
                os.chdir(directory)
                env_patch, model_patch, chain_patch = self._client(chain=FakeChain())
                with env_patch, model_patch, chain_patch, TestClient(api.app) as client:
                    response = client.post("/api/ask", json={"question": "When is the exam?"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(set(response.json()), {"answer", "sources", "status", "sources_kind", "cached"})
                self.assertFalse(response.json()["cached"])
                self.assertEqual(response.json()["status"], "answered")
                self.assertTrue(response.headers.get("X-Request-ID"))

    def test_retrieval_not_found_status_is_normalized_to_api_contract(self):
        class FakeChain:
            async def ainvoke(self, _values):
                return {"answer": "The available notices do not establish this.", "sources": [],
                        "status": "not_found_retrieval"}

        for directory in self._working_directories():
            with self.subTest(cwd=directory):
                os.chdir(directory)
                env_patch, model_patch, chain_patch = self._client(chain=FakeChain())
                with env_patch, model_patch, chain_patch, TestClient(api.app) as client:
                    response = client.post("/api/ask", json={"question": "string"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "not_found")

    def test_unexpected_chain_error_returns_generic_json_without_traceback(self):
        class FakeChain:
            async def ainvoke(self, _values):
                raise RuntimeError("secret prompt and notice content")

        expected = {
            "status": "error",
            "message": "Something went wrong. Please try again.",
        }
        for directory in self._working_directories():
            with self.subTest(cwd=directory):
                os.chdir(directory)
                env_patch, model_patch, chain_patch = self._client(chain=FakeChain())
                with env_patch, model_patch, chain_patch, TestClient(
                    api.app, raise_server_exceptions=False
                ) as client:
                    response = client.post("/api/ask", json={"question": "What is announced?"})
                self.assertEqual(response.status_code, 500)
                self.assertEqual({key: response.json()[key] for key in expected}, expected)
                self.assertEqual(response.json()["request_id"], response.headers["X-Request-ID"])
                self.assertNotIn("Traceback", response.text)
                self.assertNotIn("secret prompt", response.text)
                self.assertTrue(response.headers.get("X-Request-ID"))

    def test_empty_and_too_long_questions_return_clear_json_errors(self):
        class FakeChain:
            async def ainvoke(self, _values):
                self.fail("invalid question must not invoke the answer chain")

        for directory in self._working_directories():
            with self.subTest(cwd=directory):
                os.chdir(directory)
                env_patch, model_patch, chain_patch = self._client(chain=FakeChain())
                with env_patch, model_patch, chain_patch, TestClient(api.app) as client:
                    empty = client.post("/api/ask", json={"question": "   "})
                    too_long = client.post("/api/ask", json={"question": "x" * 1001})
                self.assertIn(empty.status_code, {400, 422})
                self.assertIn("question", empty.json()["message"])
                self.assertIn(too_long.status_code, {400, 422})
                self.assertIn("1000", too_long.json()["message"])


if __name__ == "__main__":
    unittest.main()
