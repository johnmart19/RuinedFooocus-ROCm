"""Installer probes use mocked children and never load a GPU runtime."""

import json
import os
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules import runtime_support


class InstalledRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.version = patch.object(runtime_support.importlib.metadata, "version", return_value="2.9.1")
        self.version.start()
        self.addCleanup(self.version.stop)

    def test_valid_bundle_survives_native_library_output(self):
        bundle = {"cuda": "12.8", "versions": {"torch": "2.9.1+cu128"}}
        child = SimpleNamespace(returncode=0, stdout="native library notice\n" + json.dumps(bundle), stderr="")
        with patch.object(runtime_support.subprocess, "run", return_value=child):
            self.assertEqual(runtime_support.inspect_installed_cuda("Windows"), bundle)

    def test_unusable_cuda_probe_does_not_abort_startup(self):
        for output in ("", "native library notice", "[]", "{}", '{"cuda": 128, "versions": {}}'):
            with self.subTest(output=output), patch.object(runtime_support.subprocess, "run",
                    return_value=SimpleNamespace(returncode=0, stdout=output, stderr="")):
                self.assertIsNone(runtime_support.inspect_installed_cuda("Windows"))
        for error in (OSError("cannot launch child"), subprocess.TimeoutExpired("probe", 90)):
            with self.subTest(error=error), patch.object(runtime_support.subprocess, "run", side_effect=error):
                self.assertIsNone(runtime_support.inspect_installed_cuda("Windows"))

    def test_unusable_rocm_probe_requires_explicit_repair(self):
        child = SimpleNamespace(returncode=0, stdout="not JSON", stderr="")
        with patch.object(runtime_support.subprocess, "run", return_value=child):
            with self.assertRaisesRegex(RuntimeError, "has not replaced them"):
                runtime_support.inspect_installed_rocm("Windows")
            self.assertIsNone(runtime_support.inspect_installed_rocm("Windows", repair=True))

    def test_explicit_backend_override_prevents_other_gpu_probes(self):
        with patch.dict(os.environ, {"TORCH_PLATFORM": "cpu"}), patch.object(
                runtime_support.subprocess, "run") as run:
            self.assertIsNone(runtime_support.inspect_installed_cuda("Windows"))
            self.assertIsNone(runtime_support.inspect_installed_rocm("Windows"))
            run.assert_not_called()

    def test_missing_runtime_is_not_a_probe_error(self):
        with patch.object(runtime_support.subprocess, "run", return_value=SimpleNamespace(
                returncode=0, stdout="null\n", stderr="")):
            self.assertIsNone(runtime_support.inspect_installed_cpu())
            self.assertIsNone(runtime_support.inspect_installed_cuda("Windows"))
            self.assertIsNone(runtime_support.inspect_installed_rocm("Windows"))


if __name__ == "__main__":
    unittest.main()
