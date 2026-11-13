"""Offline API behavior and startup tests; no provider or database calls."""

from __future__ import annotations

import os as _os

for _name in ("RATE_LIMIT_PER_MINUTE", "RATE_LIMIT_PER_DAY", "GLOBAL_DAILY_MODEL_CALLS"):
    _os.environ.setdefault(_name, "0")  # off for these tests; limits are tested in test_limits.py

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
                self.assertEqual(
                    set(response.json()) - {"data_as_of"},
                    {"answer", "sources", "status", "sources_kind", "cached", "links"},
                )
                self.assertFalse(response.json()["cached"])
                self.assertEqual(response.json()["status"], "answered")
                self.assertTrue(response.headers.get("X-Request-ID"))

    def test_cors_allows_configured_frontend_origin_and_exposes_request_id(self):
        for origin in ("http://localhost:5173", "http://127.0.0.1:5173"):
            with self.subTest(origin=origin):
                env_patch, model_patch, chain_patch = self._client()
                with env_patch, model_patch, chain_patch, TestClient(api.app) as client:
                    response = client.get("/health", headers={"Origin": origin})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["access-control-allow-origin"], origin)
                self.assertEqual(response.headers["access-control-expose-headers"], "X-Request-ID")
                self.assertTrue(response.headers["X-Request-ID"])

    def test_cors_preflight_accepts_both_vite_origins(self):
        for origin in ("http://127.0.0.1:5173", "http://localhost:5173"):
            with self.subTest(origin=origin):
                env_patch, model_patch, chain_patch = self._client()
                headers = {
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                }
                with env_patch, model_patch, chain_patch, TestClient(api.app) as client:
                    response = client.options("/api/ask", headers=headers)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers.get("access-control-allow-origin"), origin)
                self.assertIn("POST", response.headers.get("access-control-allow-methods", ""))
                self.assertIn("content-type", response.headers.get("access-control-allow-headers", "").lower())
                self.assertEqual(response.headers.get("access-control-max-age"), "600")

    def test_cors_preflight_accepts_all_explicitly_allowed_headers(self):
        env_patch, model_patch, chain_patch = self._client()
        headers = {
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,accept,x-request-id",
        }
        with env_patch, model_patch, chain_patch, TestClient(api.app) as client:
            response = client.options("/api/ask", headers=headers)
        self.assertEqual(response.status_code, 200)
        allowed_headers = response.headers.get("access-control-allow-headers", "").lower()
        for header in ("content-type", "accept", "x-request-id"):
            self.assertIn(header, allowed_headers)

    def test_cors_preflight_rejects_untrusted_origin(self):
        env_patch, model_patch, chain_patch = self._client()
        headers = {
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        }
        with env_patch, model_patch, chain_patch, TestClient(api.app) as client:
            response = client.options("/api/ask", headers=headers)
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_post_from_allowed_origin_exposes_request_id(self):
        class FakeChain:
            async def ainvoke(self, _values):
                return {
                    "answer": "A verified answer [1]", "sources": [{"title": "Notice"}],
                    "status": "answered", "citations_valid": True,
                }

        env_patch, model_patch, chain_patch = self._client(chain=FakeChain())
        with env_patch, model_patch, chain_patch, TestClient(api.app) as client:
            response = client.post(
                "/api/ask", json={"question": "What does the notice say?"},
                headers={"Origin": "http://127.0.0.1:5173"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://127.0.0.1:5173")
        self.assertIn("x-request-id", response.headers.get("access-control-expose-headers", "").lower())

    def test_cors_headers_are_added_to_catch_all_500(self):
        async def raise_unhandled():
            raise RuntimeError("test-only failure")

        path = "/_test/cors-unhandled-error"
        api.fastapi_app.add_api_route(path, raise_unhandled, methods=["GET"])
        env_patch, model_patch, chain_patch = self._client()
        try:
            with env_patch, model_patch, chain_patch, TestClient(api.app, raise_server_exceptions=False) as client:
                response = client.get(path, headers={"Origin": "http://localhost:5173"})
        finally:
            api.fastapi_app.router.routes = [route for route in api.fastapi_app.router.routes if getattr(route, "path", None) != path]
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:5173")

    def test_cors_headers_are_added_to_empty_question_422(self):
        env_patch, model_patch, chain_patch = self._client()
        with env_patch, model_patch, chain_patch, TestClient(api.app) as client:
            response = client.post(
                "/api/ask", json={"question": ""},
                headers={"Origin": "http://127.0.0.1:5173"},
            )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://127.0.0.1:5173")

    def test_cors_origins_trim_spaces_and_trailing_slashes(self):
        with patch.dict(os.environ, {"CORS_ALLOWED_ORIGINS": " *, http://example.test/ , , http://localhost:5173/ "}):
            origins = api._cors_allowed_origins()
            self.assertEqual(origins, ["http://example.test", "http://localhost:5173"])
            test_app = api._with_cors(api.fastapi_app, origins)
            env_patch, model_patch, chain_patch = self._client()
            headers = {
                "Origin": "http://example.test",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            }
            with env_patch, model_patch, chain_patch, TestClient(test_app) as client:
                response = client.options("/api/ask", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://example.test")

    def test_cors_does_not_allow_unconfigured_origin(self):
        env_patch, model_patch, chain_patch = self._client()
        with env_patch, model_patch, chain_patch, TestClient(api.app) as client:
            response = client.get("/health", headers={"Origin": "http://evil.example"})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("access-control-allow-origin", response.headers)
        # Starlette adds Expose-Headers to any response with an Origin header; without
        # allow-origin the browser ignores it, so only allow-origin/credentials matter.
        self.assertNotIn("access-control-allow-credentials", response.headers)

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


class CorsConfigurationTests(unittest.TestCase):
    def test_defaults_are_exact_and_cors_is_outermost_with_explicit_policy(self):
        self.assertEqual(api._cors_allowed_origins(), [
            "http://127.0.0.1:5173", "http://localhost:5173",
            "http://127.0.0.1:3000", "http://localhost:3000",
        ])
        self.assertIsInstance(api.app, api.CORSMiddleware)
        self.assertEqual(api.app.allow_methods, ["GET", "POST", "OPTIONS"])
        self.assertTrue({"accept", "content-type", "x-request-id"}.issubset(set(api.app.allow_headers)))
        self.assertNotIn("*", api.app.allow_headers)
        self.assertFalse(api.app.allow_credentials)
        self.assertEqual(api.app.simple_headers.get("Access-Control-Expose-Headers"), "X-Request-ID")
        self.assertEqual(api.app.preflight_headers.get("Access-Control-Max-Age"), "600")

    def test_configured_origin_parser_ignores_empty_and_wildcard_entries(self):
        with patch.dict(os.environ, {"CORS_ALLOWED_ORIGINS": " , *, http://127.0.0.1:5173/ ,,"}):
            self.assertEqual(api._cors_allowed_origins(), ["http://127.0.0.1:5173"])


if __name__ == "__main__":
    unittest.main()
