"""Exercise the worker dispatch boundary without loading a model or starting it."""

import ast
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace


class WorkerRoutingTests(unittest.TestCase):
    def test_image_dispatch_uses_explicit_controls_and_preserves_queued_request(self):
        source = Path(__file__).resolve().parents[1] / 'ruinedfooocus/app/async_worker.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        handler = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'handler')
        for kind, expected in (('img2img', None), ('rembg', 'rembg')):
            with self.subTest(kind=kind):
                process, release = Mock(), Mock()
                controls = SimpleNamespace(get_settings=Mock(return_value={'type': kind}))
                scope = {'control_tools': controls, 'process': process}
                exec(compile(ast.Module(body=[handler], type_ignores=[]), str(source), 'exec'), scope)
                request = {'task_type': 'process', 'prompt': 'Keep two red apples.', 'image_number': 2}
                original = dict(request)
                with patch('ruinedfooocus.features.image_processing.auxiliary_resources.release_except', release):
                    scope['handler'](request)
                release.assert_called_once_with(expected)
                process.assert_called_once_with(request)
                controls.get_settings.assert_called_once_with(request)
                self.assertEqual(request, original)


if __name__ == '__main__':
    unittest.main()
