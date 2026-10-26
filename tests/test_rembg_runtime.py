"""Lazy session reuse must preserve model outputs and transparent-input behavior."""
import ast
import base64
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock

import PIL.Image


class BackgroundSessionTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(Path('modules/rembg_pipeline.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'pipeline')
        self.rembg = Mock()
        self.worker = Mock()
        self.args = SimpleNamespace(cpu=False)
        namespace = {'rembg': self.rembg, 'worker': self.worker, 'PIL': PIL, 'Any': Any, 'args': self.args, 'new_session': self.rembg.new_session}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'rembg_pipeline.py', 'exec'), namespace)
        self.pipeline = namespace['pipeline']()

    def test_repeated_requests_reuse_session_but_run_inference_for_each_image(self):
        first = PIL.Image.new('RGB', (4, 4), 'red')
        second = PIL.Image.new('RGB', (4, 4), 'blue')
        self.pipeline.remove_background(first)
        self.pipeline.remove_background(second)
        self.rembg.new_session.assert_called_once_with(cpu=False)
        self.assertEqual(self.rembg.remove.call_count, 2)
        self.assertIs(self.rembg.remove.call_args_list[0].args[0], first)
        self.assertIs(self.rembg.remove.call_args_list[1].args[0], second)
        self.assertIs(self.rembg.remove.call_args.kwargs['session'], self.rembg.new_session.return_value)

    def test_transparent_input_is_unchanged_without_loading_model_unless_forced(self):
        image = PIL.Image.new('RGBA', (4, 4), (255, 0, 0, 0))
        self.assertIs(self.pipeline.remove_background(image), image)
        self.rembg.new_session.assert_not_called()
        self.rembg.remove.assert_not_called()
        self.pipeline.remove_background(image, force=True)
        self.rembg.new_session.assert_called_once()

    def test_supplied_session_and_options_are_preserved(self):
        session = object()
        image = PIL.Image.new('RGB', (4, 4))
        self.pipeline.remove_background(image, rembg_session=session, alpha_matting=True)
        self.rembg.new_session.assert_not_called()
        self.rembg.remove.assert_called_once_with(image, session=session, alpha_matting=True)

    def test_explicit_cpu_mode_does_not_select_an_installed_gpu_provider(self):
        self.args.cpu = True
        self.pipeline.remove_background(PIL.Image.new('RGB', (4, 4)))
        self.rembg.new_session.assert_called_once_with(cpu=True)

    def test_removal_reports_source_dimensions_without_resizing(self):
        image = PIL.Image.new('RGBA', (320, 240))
        data = self.pipeline.parse_gen_data({'input_image': image, 'image_number': 3,
                                            'custom_width': 1024, 'custom_height': 1024})
        self.assertEqual((data['width'], data['height']), image.size)
        self.assertIs(data['input_image'], image)
        self.assertEqual(data['image_number'], 1)

    def test_cancelled_request_never_loads_session(self):
        self.worker.check_interrupt.side_effect = RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            self.pipeline.process({'task_id': 1, 'input_image': PIL.Image.new('RGB', (4, 4))})
        self.rembg.new_session.assert_not_called()

    def test_api_can_preserve_png_alpha_without_changing_normal_rgb_inputs(self):
        from modules.api_runtime import decode_image
        from modules.openai_api import ImageRequest
        self.assertEqual(ImageRequest(input_image='base64', controlnet={'type': 'rembg'}).controlnet.type, 'rembg')
        image = PIL.Image.new('RGBA', (4, 4), (200, 100, 50, 42))
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        encoded = base64.b64encode(buffer.getvalue()).decode()
        self.assertEqual(decode_image(encoded).mode, 'RGB')
        preserved = decode_image(encoded, preserve_alpha=True)
        self.assertEqual(preserved.mode, 'RGBA')
        self.assertEqual(preserved.tobytes(), image.tobytes())


if __name__ == '__main__':
    unittest.main()
