"""Offline tests for the uuid_utils fallback and the native-import checker."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

from ingestion import check_native_imports
from ingestion import compat_uuid_utils as compat

BACKEND_ROOT = Path(__file__).resolve().parent.parent
WARNING = "uuid_utils native module blocked; using pure-Python fallback"


def run_python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(BACKEND_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )


class Uuid7FallbackTests(unittest.TestCase):
    def test_valid_v7_rfc_variant(self) -> None:
        value = compat.uuid7()
        self.assertIsInstance(value, uuid.UUID)
        self.assertEqual(value.version, 7)
        self.assertEqual(value.variant, uuid.RFC_4122)

    def test_timestamp_and_nanos_arguments(self) -> None:
        value = compat.uuid7(timestamp=1_700_000_000, nanos=123_000_000)
        self.assertEqual(value.int >> 80, 1_700_000_000_123)
        self.assertEqual(value.version, 7)

    def test_sorts_by_time(self) -> None:
        early = compat.uuid7()
        time.sleep(0.01)
        late = compat.uuid7()
        self.assertLess(early, late)
        self.assertLess(compat.uuid7(timestamp=1000), compat.uuid7(timestamp=2000))

    def test_thousand_unique_and_monotonic(self) -> None:
        values = [compat.uuid7() for _ in range(1000)]
        self.assertEqual(len(set(values)), 1000)
        self.assertEqual(values, sorted(values))


class EnsureUuidUtilsTests(unittest.TestCase):
    def test_noop_when_real_module_imports(self) -> None:
        proc = run_python(
            "import sys, types\n"
            "pkg = types.ModuleType('uuid_utils'); pkg.__path__ = []\n"
            "cm = types.ModuleType('uuid_utils.compat'); cm.uuid7 = lambda **k: None\n"
            "sys.modules['uuid_utils'] = pkg; sys.modules['uuid_utils.compat'] = cm\n"
            "from ingestion.compat_uuid_utils import ensure_uuid_utils\n"
            "print(ensure_uuid_utils(), sys.modules['uuid_utils.compat'] is cm)\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "False True")
        self.assertNotIn(WARNING, proc.stderr)

    def test_blocked_module_gets_fallback_for_langchain(self) -> None:
        proc = run_python(
            "import sys\n"
            "sys.modules['uuid_utils'] = None\n"
            "from ingestion.compat_uuid_utils import ensure_uuid_utils\n"
            "ensure_uuid_utils(); ensure_uuid_utils()\n"
            "import langchain_core.runnables\n"
            "from langchain_core.utils.uuid import uuid7\n"
            "assert uuid7().version == 7 and uuid7(nanoseconds=1_700_000_000_000_000_000).version == 7\n"
            "print('ok')\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "ok")
        self.assertEqual(proc.stderr.count(WARNING), 1, proc.stderr)

    def test_api_imports_when_native_module_blocked(self) -> None:
        proc = run_python(
            "import sys\n"
            "sys.modules['uuid_utils'] = None\n"
            "import api\n"
            "print('ok')\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip().splitlines()[-1], "ok")
        self.assertEqual(proc.stderr.count(WARNING), 1, proc.stderr)


class CheckerTests(unittest.TestCase):
    def test_classify(self) -> None:
        classify = check_native_imports.classify
        self.assertEqual(classify(0, ""), "OK")
        self.assertEqual(classify(1, "ImportError: DLL load failed while importing x"), "BLOCKED")
        self.assertEqual(classify(1, "An Application Control policy has blocked this file."), "BLOCKED")
        self.assertEqual(classify(1, "ModuleNotFoundError: No module named 'zzz'"), "MISSING")
        self.assertEqual(classify(1, "ValueError: boom"), "OTHER")

    def test_simulated_blocked_module_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = (
                f"import sys; sys.path.insert(0, {tmp!r}); "
                "open(sys.path[0] + '/fake_native.py', 'w').write("
                "'raise ImportError(\"DLL load failed while importing x: "
                "An Application Control policy has blocked this file.\")'); "
                "import fake_native"
            )
            status, error = check_native_imports.check_package("fake_native", code=code)
        self.assertEqual(status, "BLOCKED")
        self.assertIn("Application Control", error)

    def test_missing_module(self) -> None:
        status, _ = check_native_imports.check_package("definitely_not_a_real_module_xyz")
        self.assertEqual(status, "MISSING")


if __name__ == "__main__":
    unittest.main()
