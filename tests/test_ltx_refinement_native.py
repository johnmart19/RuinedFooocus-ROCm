"""Verify the actual refinement call matches the native LTX-2.5 graph."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch


class NativeRefinementTests(unittest.TestCase):
    def test_stage_two_sampler_and_sigmas_match_official_workflow(self):
        source = Path(__file__).resolve().parents[1] / "modules/ltx_refinement.py"
        tree = ast.parse(source.read_text())
        refine = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "refine")
        sample = next(node for node in ast.walk(refine)
                      if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                      and ast.unparse(node.value.func) == "guider.sample")
        selector = Mock()
        selector.get_sampler.return_value = ("native ancestral sampler",)
        guider = Mock(model_patcher="model")
        noise = Mock(seed=42)
        namespace = dict(guider=guider, noise=noise,
                         latent={"samples": "av latent", "noise_mask": "mask"},
                         KSamplerSelect=lambda: selector, torch=torch,
                         video_callback=Mock(return_value="callback"), gen_data={},
                         comfy=SimpleNamespace(utils=SimpleNamespace(PROGRESS_BAR_ENABLED=True)))
        exec(compile(ast.Module(body=[sample], type_ignores=[]), str(source), "exec"), namespace)
        selector.get_sampler.assert_called_once_with("euler_ancestral")
        args = guider.sample.call_args.args
        self.assertEqual(args[2], "native ancestral sampler")
        torch.testing.assert_close(args[3], torch.tensor([0.85, 0.7250, 0.4219, 0.0]))
        self.assertEqual(guider.sample.call_args.kwargs["denoise_mask"], "mask")


if __name__ == "__main__":
    unittest.main()
