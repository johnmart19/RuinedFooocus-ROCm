"""Source geometry and input bounds, including tall images and native edit canvases."""
import base64
import io
import unittest

import numpy as np
from PIL import Image, ImageDraw
import torch

from modules.image_geometry import reference_plan, prepare_reference, restore_reference_frame
from modules.image_guidance import validate_guidance


class ImageGeometryTests(unittest.TestCase):
    def test_all_six_reference_shapes_scale_uniformly_without_forced_landscape(self):
        for source in ((3364, 7480), (928, 1664), (2758, 3270), (768, 1344), (1344, 896), (928, 1232)):
            with self.subTest(source=source):
                plan = reference_plan(source, (1152, 896))
                width, height = plan['width'], plan['height']
                self.assertLessEqual(abs(width * source[1] - height * source[0]), sum(source) / 2)
                self.assertLessEqual(width, source[0])
                self.assertLessEqual(height, source[1])
                self.assertEqual(plan['canvas_width'] % 16, 0)
                self.assertEqual(plan['canvas_height'] % 16, 0)
                self.assertLess(plan['canvas_width'] - width, 16)
                self.assertLess(plan['canvas_height'] - height, 16)

    def test_a_circle_stays_round_after_portrait_downscaling(self):
        image = Image.new('RGB', (501, 1001))
        ImageDraw.Draw(image).ellipse((150, 400, 350, 600), fill='white')
        pixels, frame = prepare_reference(image, (512, 512))
        restored = restore_reference_frame(pixels, frame)[0]
        y, x = torch.where(restored[..., 0] > .5)
        self.assertLessEqual(abs(int(x.max() - x.min()) - int(y.max() - y.min())), 1)
        self.assertEqual(tuple(restored.shape[:2]), (frame['height'], frame['width']))

    def test_small_sources_are_kept_at_their_original_dimensions(self):
        image = Image.new('RGB', (117, 239), (17, 67, 193))
        pixels, frame = prepare_reference(image, (1152, 896))
        restored = restore_reference_frame(pixels, frame)
        self.assertEqual(tuple(restored.shape), (1, 239, 117, 3))
        np.testing.assert_array_equal((restored[0].numpy() * 255).round().astype(np.uint8), np.asarray(image))

    def test_explicit_crop_fills_the_requested_size_without_stretching(self):
        image = Image.new('RGB', (64, 112))
        ImageDraw.Draw(image).ellipse((20, 44, 44, 68), fill='white')
        pixels, frame = prepare_reference(image, (512, 768), 'crop')
        restored = restore_reference_frame(pixels, frame)[0]
        self.assertEqual(tuple(restored.shape), (768, 512, 3))
        y, x = torch.where(restored[..., 0] > .5)
        self.assertLessEqual(abs(int(x.max() - x.min()) - int(y.max() - y.min())), 1)

    def test_a_wrong_decoder_canvas_fails_instead_of_rescaling_the_output(self):
        plan = reference_plan((928, 1664), (1152, 896))
        with self.assertRaisesRegex(ValueError, 'refusing to stretch'):
            restore_reference_frame(torch.zeros(1, 896, 1152, 3), plan)
        with self.assertRaises(ValueError):
            validate_guidance({'image_resize_mode': 'stretch'})

    def test_large_generation_inputs_can_be_downscaled_without_weakening_other_image_limits(self):
        from modules.api_runtime import decode_image
        with Image.new('L', (4097, 4096)) as image:
            data = io.BytesIO()
            image.save(data, format='PNG')
        encoded = base64.b64encode(data.getvalue()).decode()
        with self.assertRaisesRegex(ValueError, '16 megapixels'):
            decode_image(encoded)
        decoded = decode_image(encoded, max_pixels=67108864)
        self.assertEqual(decoded.size, (4097, 4096))
        decoded.close()


if __name__ == '__main__':
    unittest.main()
