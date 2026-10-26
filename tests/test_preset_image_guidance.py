"""PNG preset guidance restoration without UI startup or generation."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

from modules.image_guidance import FIELD_SCHEMA, restore_guidance_metadata


class PresetImageGuidanceTests(unittest.TestCase):
    def test_grouped_fields_restore_and_flat_fields_take_precedence(self):
        group = dict(moss_image_edit_format='ai-toolkit', moss_image_edit_strength=2,
                     moss_image_slg=True, sampling_shift=None, cfg_norm_strength=1,
                     effective_sampling_shift=5, reference_images=['private/path.png'])
        result = restore_guidance_metadata({'image_guidance': group,
                                             'moss_image_edit_strength': 0, 'cfg_norm_strength': None})
        self.assertEqual(result, dict(moss_image_edit_format='ai-toolkit', moss_image_edit_strength=0,
                                     moss_image_slg=True, sampling_shift=5, cfg_norm_strength=None))
        self.assertTrue(set(result).issubset(FIELD_SCHEMA))

    def test_valid_explicit_shift_wins_and_auto_promotes_only_valid_actual(self):
        for auto in (None, ''):
            self.assertEqual(restore_guidance_metadata({'sampling_shift': auto, 'effective_sampling_shift': 3.1}),
                             {'sampling_shift': 3.1})
        self.assertEqual(restore_guidance_metadata({'effective_sampling_shift': 5}), {'sampling_shift': 5})
        self.assertEqual(restore_guidance_metadata({'sampling_shift': 4, 'effective_sampling_shift': 5}),
                         {'sampling_shift': 4})
        for invalid in (True, 0, 101, float('inf'), float('nan'), 'not a number'):
            self.assertEqual(restore_guidance_metadata({'effective_sampling_shift': invalid}), {})
        self.assertEqual(restore_guidance_metadata({'sampling_shift': 0, 'effective_sampling_shift': 5}), {})
        self.assertEqual(restore_guidance_metadata({'effective_sampling_shift': 1.15}, family='Flux'), {})
        self.assertEqual(restore_guidance_metadata({'effective_sampling_shift': 1.15,
                                                   'sampling_shift_supported': False}), {})

    def test_old_or_malformed_metadata_does_not_add_untrusted_values(self):
        self.assertEqual(restore_guidance_metadata({'steps': 20}), {})
        self.assertEqual(restore_guidance_metadata({'image_guidance': 'not a dictionary'}), {})
        self.assertEqual(restore_guidance_metadata({'reference_images': ['image.png'],
            'moss_image_edit_format': 'unknown', 'moss_image_slg': 'false'}), {})

    def test_generate_clicked_restores_guidance_only_when_sampling_preset_bit_enabled(self):
        source = (Path(__file__).resolve().parents[1] / 'webui.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        generate = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'generate_clicked')
        branch = next(node for node in ast.walk(generate) if isinstance(node, ast.If)
                      and isinstance(node.test, ast.BinOp) and isinstance(node.test.op, ast.BitAnd)
                      and isinstance(node.test.left, ast.Constant) and node.test.left.value == 2)
        recipe = dict(steps=30, cfg=4, sampler_name='euler', scheduler='simple', clip_skip=1,
                      image_guidance={'moss_image_edit_format': 'Moss-Image-Edit',
                                      'sampling_shift': None, 'effective_sampling_shift': 5})
        for mode in (0, 2):
            namespace = dict(settings={'preset_mode': mode}, preset_data=recipe,
                             gen_data={'base_model_name': 'moss', 'moss_image_edit_format': 'Disabled'})
            with patch('modules.ui.ui_image_guidance.family_for_checkpoint', return_value='MOSSImage'):
                exec(compile(ast.Module(body=[branch], type_ignores=[]), 'preset_sampling', 'exec'), namespace)
            if mode:
                self.assertEqual(namespace['gen_data']['moss_image_edit_format'], 'Moss-Image-Edit')
                self.assertEqual(namespace['gen_data']['sampling_shift'], 5)
            else:
                self.assertEqual(namespace['gen_data'], {'base_model_name': 'moss', 'moss_image_edit_format': 'Disabled'})


if __name__ == '__main__':
    unittest.main()
