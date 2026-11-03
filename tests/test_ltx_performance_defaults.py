"""Resolve real family presets through Model Defaults without importing a GPU."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from modules.performance import PerformanceSettings


class LTXPerformanceDefaultsTests(unittest.TestCase):
    def test_model_defaults_select_each_video_generation_and_preserve_legacy(self):
        defaults = Path(__file__).resolve().parents[1] / 'settings/performance.default'
        registry = SimpleNamespace(KSampler=SimpleNamespace(SAMPLERS=[], SCHEDULERS=[]))
        with tempfile.TemporaryDirectory(dir=defaults.parents[1]) as directory:
            local = Path(directory) / 'performance.json'
            local.write_text('{}', encoding='utf-8')
            with patch.object(PerformanceSettings, 'DEFAULT_PERFORMANCE_FILE', defaults), \
                 patch.object(PerformanceSettings, 'PERFORMANCE_FILE', local), \
                 patch.dict(sys.modules, {'comfy.samplers': registry}):
                settings = PerformanceSettings(cache_dir=directory, offline=True)
                for family, model, expected in (
                    ('LTXV', 'ltxv-2b-0.9.6-dev.safetensors', (30, 3, 'euler', 'simple')),
                    ('LTXV2', 'ltx-2-19b-dev-Q4_K_M.gguf', (20, 4, 'euler_ancestral', 'simple')),
                    ('LTXV 2.3', 'ltx-2.3-22b-dev-Q4_K_M.gguf', (30, 3, 'euler', 'simple')),
                    ('LTXV 2.5', 'ltx-2.5-distilled.safetensors', (8, 1, 'euler_ancestral', 'simple')),
                ):
                    with self.subTest(family=family):
                        values = settings.get_perf_options('Model Defaults', model, family)
                        self.assertEqual(tuple(values[key] for key in
                            ('custom_steps', 'cfg', 'sampler_name', 'scheduler')), expected)
                        if family in ('LTXV2', 'LTXV 2.3'):
                            self.assertIn('768x512', values['recommended_resolutions'])
                            self.assertIn('STG off', values['recommended_resolutions'])
                saved = json.loads(local.read_text(encoding='utf-8'))
                self.assertEqual(saved['LTX 2 Dev']['video_models'], ['LTXV2'])
                self.assertEqual(saved['LTX 2.3 Dev']['video_models'], ['LTXV 2.3'])


if __name__ == '__main__':
    unittest.main()
