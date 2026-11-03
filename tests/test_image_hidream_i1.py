"""HiDream I1 variant recipes and request shift precedence, without weights."""
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from modules.image_guidance import request_model, resolve_guidance
from modules.model_recommendations import ModelRecommendations
from modules.performance import PerformanceSettings
from test_image_shift_defaults import Patcher, native_nodes

ROOT = Path(__file__).resolve().parents[1]
SAMPLERS = ['euler', 'uni_pc', 'lcm', 'dpmpp_2m_sde_gpu']
SCHEDULERS = ['simple', 'normal', 'karras']
REGISTRY = SimpleNamespace(KSampler=SimpleNamespace(SAMPLERS=SAMPLERS, SCHEDULERS=SCHEDULERS))


class HiDreamI1Tests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.performance = PerformanceSettings.__new__(PerformanceSettings)
        self.performance.performance_options = json.loads(PerformanceSettings.DEFAULT_PERFORMANCE_FILE.read_text(encoding='utf-8'))
        self.performance.legacy_options = {}
        self.performance.model_recommendations = ModelRecommendations(self.root, self.root / 'settings.json', offline=True)

    def recipe(self, checkpoint):
        with patch.dict(sys.modules, {'comfy.samplers': REGISTRY}):
            return self.performance.get_perf_options('Model Defaults', checkpoint, 'HiDream')

    def prepared(self, checkpoint='hidream_i1_dev_fp8.safetensors', **values):
        with patch.dict(sys.modules, {'comfy.samplers': REGISTRY}):
            return self.performance.apply(dict(performance_selection='Model Defaults', base_model_name=checkpoint, **values), 'HiDream')

    def route(self, data, startup=None):
        tree = ast.parse((ROOT / 'modules/image_pipeline.py').read_text(encoding='utf-8'))
        process = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'process')
        fallback = next(n for n in process.body if isinstance(n, ast.If) and ast.unparse(n.test).startswith("gen_data.get('sampling_shift'"))
        base = Patcher(1)
        scope = dict(gen_data=data, settings=SimpleNamespace(default_settings={}),
            self=SimpleNamespace(model_info={'sampling_shift': startup}))
        exec(compile(ast.Module(body=[fallback], type_ignores=[]), 'hidream_i1_shift', 'exec'), scope)
        options = resolve_guidance(data, {}, 'HiDream')
        return base, request_model(base, options, 'HiDream', nodes=native_nodes()), options

    def test_full_dev_fast_use_distinct_native_workflows_and_basename_only(self):
        cases = [('hidream_i1_full_fp8.safetensors', (50, 5, 'uni_pc', 'simple', 3)),
            ('folder/HiDream-I1-DEV-Q4.gguf', (28, 1, 'lcm', 'normal', 6)),
            ('folder\\hidream_i1_fast_fp8.safetensors', (16, 1, 'lcm', 'normal', 3)),
            ('dev/fastlane_hidream.safetensors', (50, 5, 'uni_pc', 'simple', 3))]
        for checkpoint, expected in cases:
            with self.subTest(checkpoint=checkpoint):
                values = self.recipe(checkpoint)
                self.assertEqual(tuple(values.get(key) for key in ('custom_steps', 'cfg', 'sampler_name', 'scheduler', 'sampling_shift')), expected)
        self.assertNotIn(self.performance.family_preset('HiDreamO1', 'hidream_o1_dev.safetensors'),
                         ('HiDream I1 Full', 'HiDream I1 Dev', 'HiDream I1 Fast'))

    def test_automatic_recipe_shift_uses_request_clone_and_respects_explicit_values(self):
        for startup, requested, expected in ((None, None, 6), (4.25, None, 4.25), (4.25, 7, 7)):
            with self.subTest(startup=startup, requested=requested):
                data = self.prepared(sampling_shift=requested)
                base, model, options = self.route(data, startup)
                self.assertEqual(options['sampling_shift'], expected)
                self.assertEqual(model.sampling.shift, expected)
                self.assertEqual(base.sampling.shift, 1)
                self.assertIsNot(model.sampling, base.sampling)

    def test_own_settings_and_imported_metadata_do_not_inherit_a_stale_recipe_shift(self):
        for selection in ('Own settings', 'Custom...', None):
            data = self.prepared(sampling_shift=None)
            data['performance_selection'] = selection  # Metadata import happens after named defaults.
            base, model, options = self.route(data)
            self.assertIsNone(options['sampling_shift'])
            self.assertIs(model.sampling, base.sampling)
            with patch.dict(sys.modules, {'comfy.samplers': REGISTRY}):
                clean = self.performance.apply(data, 'HiDream')
            self.assertNotIn('_model_default_sampling_shift', clean)

    def test_public_and_saved_sampling_values_override_family_recipe_and_shift_is_visible(self):
        checkpoint = 'hidream_i1_dev.safetensors'
        (self.root / 'hidream_i1_dev.json').write_text(json.dumps(dict(baseModel='HiDream',
            generationConfig=dict(steps=35, cfgScale=2.5, sampler_name='euler', scheduler='simple'))))
        values = self.recipe(checkpoint)
        self.assertEqual((values['custom_steps'], values['cfg'], values.get('sampling_shift')), (35, 2.5, 6))
        description = self.performance.model_recommendations.describe(checkpoint, values, SAMPLERS, SCHEDULERS)
        self.assertIn('| Sampling shift | 6 | Family preset |', description)
        self.performance.model_recommendations.save_override(checkpoint,
            dict(custom_steps=12, cfg=3, sampler_name='lcm', scheduler='normal', clip_skip=1), SAMPLERS, SCHEDULERS)
        values = self.recipe(checkpoint)
        self.assertEqual((values['custom_steps'], values['cfg'], values['sampler_name'], values.get('sampling_shift')), (12, 3, 'lcm', 6))

    def test_recipe_shift_is_validated_before_a_native_patch(self):
        self.performance.performance_options['HiDream I1 Dev'] = dict(
            custom_steps=28, cfg=1, sampler_name='lcm', scheduler='normal', clip_skip=1, sampling_shift=101)
        with self.assertRaisesRegex(ValueError, 'sampling_shift'):
            self.route(self.prepared(sampling_shift=None))


if __name__ == '__main__':
    unittest.main()
