"""Exercise real Gradio preprocessing configured by both app upload controls."""
import ast
from pathlib import Path
import tempfile
import unittest

import gradio as gr
from gradio.data_classes import ImageData
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


class TransparentUploadTests(unittest.TestCase):
    def test_both_uploaders_preserve_rgba_palette_and_grayscale_transparency(self):
        rgba = Image.fromarray(np.array([
            [[255, 0, 0, 0], [0, 255, 0, 64]],
            [[0, 0, 255, 128], [255, 255, 255, 255]],
        ], dtype=np.uint8))
        images = [rgba, rgba.quantize(method=Image.Quantize.FASTOCTREE), rgba.convert('LA')]
        for source, name in [('webui.py', 'upscale_upload'),
                             ('modules/ui/ui_controlnet.py', 'input_image')]:
            tree = ast.parse((ROOT / source).read_text(encoding='utf-8'))
            assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
                and isinstance(node.value, ast.Call))
            options = {keyword.arg: ast.literal_eval(keyword.value)
                for keyword in assignment.value.keywords if keyword.arg in ('type', 'image_mode')}
            # Use the actual app options, including Gradio's default if removed.
            uploader = gr.Image(**options)
            with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
                for image in images:
                    with self.subTest(uploader=name, mode=image.mode):
                        path = Path(directory) / 'transparent.png'
                        image.save(path)
                        uploaded = uploader.preprocess(ImageData(path=str(path), orig_name=path.name))
                        self.assertEqual(uploaded.mode, 'RGBA')
                        np.testing.assert_array_equal(np.array(uploaded), np.array(image.convert('RGBA')))


if __name__ == '__main__':
    unittest.main()
