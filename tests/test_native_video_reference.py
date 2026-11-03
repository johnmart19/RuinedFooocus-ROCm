"""CPU-only contracts for the native reference's sampler choice."""
import ast
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from tools.comfy_video_reference import _wan_sample


class NativeWanSamplerTests(unittest.TestCase):
    def test_wan_executes_stock_node_and_observes_its_actual_sigmas(self):
        effective_sigmas = np.array([1., .8, .4, 0.])
        output = {'samples': object()}

        class Guider:
            def sample(self, noise, latent, sampler, sigmas, **kwargs):
                return output

        node = Mock()
        def execute(*args, **kwargs):
            self.assertEqual(args[4:6], ('uni_pc', 'simple'))
            self.assertEqual(kwargs, {'denoise': 1.0})
            return (Guider().sample(None, None, None, effective_sigmas),)
        node.sample.side_effect = execute
        nodes = types.ModuleType('nodes')
        nodes.KSampler = Mock(return_value=node)
        samplers = types.ModuleType('comfy.samplers')
        samplers.CFGGuider = Guider
        original = Guider.sample
        with patch.dict(sys.modules, {'nodes': nodes, 'comfy.samplers': samplers}):
            result, sigmas = _wan_sample(None, 7, 3, 4., 'uni_pc', 'simple', [], [], {})
        self.assertIs(result, output)
        self.assertIs(sigmas, effective_sigmas)
        self.assertIs(Guider.sample, original)
        nodes.KSampler.assert_called_once_with()

    def test_backend_uni_pc_adds_step_then_discards_penultimate_sigma(self):
        # Execute the installed backend's actual schedule method with NumPy
        # standing in for torch.cat, avoiding GPU/backend imports entirely.
        path = Path(__file__).resolve().parents[1] / 'repositories/ComfyUI/comfy/samplers.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'KSampler')
        method = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == 'calculate_sigmas')
        discard = next(node for node in klass.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == 'DISCARD_PENULTIMATE_SIGMA_SAMPLERS' for target in node.targets))
        scope = {'torch': types.SimpleNamespace(cat=np.concatenate),
                 'calculate_sigmas': Mock(return_value=np.array([1., .9, .7, .4, 0.]))}
        isolated = ast.fix_missing_locations(ast.Module(body=[discard, method], type_ignores=[]))
        exec(compile(isolated, str(path), 'exec'), scope)
        owner = types.SimpleNamespace(sampler='uni_pc', scheduler='simple', model=Mock(),
            DISCARD_PENULTIMATE_SIGMA_SAMPLERS=scope['DISCARD_PENULTIMATE_SIGMA_SAMPLERS'])
        result = scope['calculate_sigmas']  # The definition replaced the mock name.
        schedule = Mock(return_value=np.array([1., .9, .7, .4, 0.]))
        scope['calculate_sigmas'] = schedule
        actual = result(owner, 3)
        self.assertEqual(schedule.call_args.args[-1], 4)
        np.testing.assert_array_equal(actual, [1., .9, .7, 0.])


if __name__ == '__main__':
    unittest.main()
