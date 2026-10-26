"""CPU-only replay of the production metadata importer and guidance policy."""
import ast
import json
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.image_guidance import resolve_guidance

ROOT = Path(__file__).resolve().parents[1]


def replay(parameters, family='QwenImage'):
    tree = ast.parse((ROOT / 'modules/prompt_processing.py').read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'process_metadata')
    scope = {'json': json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'metadata_replay', 'exec'), scope)
    models = SimpleNamespace(get_models_by_path=lambda *args, **kwargs: {},
        get_model_base=lambda info: family, get_file=lambda *args: Path('model.safetensors'),
        detect_checkpoint_base=lambda path: family)
    with patch.dict('sys.modules', {'shared': SimpleNamespace(models=models)}):
        return scope['process_metadata']({'base_model_name': 'model.safetensors',
            'prompt': json.dumps({'prompt': 'edit', **parameters})})


class EffectiveShiftReplayTests(unittest.TestCase):
    def test_recorded_auto_shift_replays_when_current_model_default_changes(self):
        imported = replay({'sampling_shift': None, 'effective_sampling_shift': 3.1})
        self.assertEqual(resolve_guidance(imported, {}, 'QwenImage')['sampling_shift'], 3.1)
        self.assertEqual(imported['effective_sampling_shift'], 3.1)

    def test_explicit_requested_shift_has_precedence(self):
        imported = replay({'sampling_shift': 2.2, 'effective_sampling_shift': 3.1})
        self.assertEqual(imported['sampling_shift'], 2.2)

    def test_grouped_metadata_can_restore_recorded_effective_shift(self):
        imported = replay({'image_guidance': {'sampling_shift': None, 'effective_sampling_shift': 4.0}})
        self.assertEqual(imported['sampling_shift'], 4.0)

    def test_grouped_moss_guidance_replays_all_fields_and_keeps_unrelated_metadata(self):
        imported = replay({'cfg': 4.5, 'seed': 71, 'image_guidance': {
            'moss_image_edit_format': 'Moss-Image-Edit', 'moss_image_edit_strength': 2.5,
            'moss_image_slg': True, 'sampling_shift': None,
            'effective_sampling_shift': 5.0, 'cfg_norm_strength': 1.25}}, family='MOSSImage')
        values = resolve_guidance(imported, {}, 'MOSSImage')
        self.assertEqual(values, {'moss_image_edit_format': 'Moss-Image-Edit',
            'moss_image_edit_strength': 2.5, 'moss_image_slg': True,
            'sampling_shift': 5.0, 'cfg_norm_strength': 1.25})
        self.assertEqual(imported['cfg'], 4.5)
        self.assertEqual(imported['seed'], 71)
        self.assertEqual(imported['prompt'], 'edit')
        self.assertEqual(imported['performance_selection'], 'Own settings')

    def test_invalid_recorded_effective_shift_is_not_promoted(self):
        for value in (True, float('nan'), float('inf'), -1, 0, 101, 'not a number'):
            with self.subTest(value=value):
                imported = replay({'sampling_shift': None, 'effective_sampling_shift': value})
                self.assertIsNone(imported['sampling_shift'])

    def test_native_flux_shift_is_not_promoted_to_unsupported_override(self):
        imported = replay({'sampling_shift': None, 'effective_sampling_shift': 1.15}, family='Flux.1 D')
        self.assertIsNone(imported['sampling_shift'])
        self.assertIsNone(resolve_guidance(imported, {}, 'Flux.1 D')['sampling_shift'])

    def test_older_auto_metadata_without_effective_shift_stays_auto(self):
        self.assertIsNone(replay({'sampling_shift': None})['sampling_shift'])


if __name__ == '__main__':
    unittest.main()
