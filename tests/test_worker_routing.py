"""Exercise the worker dispatch boundary without loading a model or starting it."""

import ast
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace


class WorkerRoutingTests(unittest.TestCase):
    def test_startup_loads_weights_only_when_explicitly_enabled(self):
        source = Path(__file__).resolve().parents[1] / "ruinedfooocus/app/async_worker.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "preload_startup_model"
        )
        registry = SimpleNamespace(update=Mock())
        settings = SimpleNamespace(default_settings={"base_model": "selected.safetensors"})
        scope = {"settings": settings, "pipelines": registry}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
        scope["preload_startup_model"]()
        registry.update.assert_not_called()
        settings.default_settings["preload_image_model"] = True
        scope["preload_startup_model"]()
        registry.update.assert_called_once_with({"base_model_name": "selected.safetensors"})
        registry.update.return_value.load_base_model.assert_called_once_with("selected.safetensors")

    def test_image_dispatch_uses_explicit_controls_and_preserves_queued_request(self):
        source = Path(__file__).resolve().parents[1] / "ruinedfooocus/app/async_worker.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        handler = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "handler"
        )
        for kind, expected in (("img2img", None), ("rembg", "rembg")):
            with self.subTest(kind=kind):
                process, release = Mock(), Mock()
                controls = SimpleNamespace(get_settings=Mock(return_value={"type": kind}))
                scope = {"control_tools": controls, "process": process}
                exec(
                    compile(ast.Module(body=[handler], type_ignores=[]), str(source), "exec"), scope
                )
                request = {
                    "task_type": "process",
                    "prompt": "Keep two red apples.",
                    "image_number": 2,
                }
                original = dict(request)
                with patch(
                    "ruinedfooocus.features.image_processing.auxiliary_resources.release_except",
                    release,
                ):
                    scope["handler"](request)
                release.assert_called_once_with(expected)
                process.assert_called_once_with(request)
                controls.get_settings.assert_called_once_with(request)
                self.assertEqual(request, original)


if __name__ == "__main__":
    unittest.main()
