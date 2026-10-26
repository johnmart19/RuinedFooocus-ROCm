"""Isolate GPU/UI boundaries while exercising the production function bodies."""

import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from modules.settings_validation import validate_settings


ROOT = Path(__file__).resolve().parents[1]


class ImageOutputTests(unittest.TestCase):
    def test_decode_oom_releases_diffusion_before_one_retry(self):
        tree = ast.parse((ROOT / "modules/image_pipeline.py").read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "pipeline")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "decode_latent")
        memory = SimpleNamespace(unload_model_and_clones=Mock(), soft_empty_cache=Mock())
        def raise_non_oom(error):
            if not isinstance(error, MemoryError):
                raise error
        memory.raise_non_oom = raise_non_oom
        namespace = {"comfy": SimpleNamespace(model_management=memory)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "decode_retry", "exec"), namespace)
        owner = SimpleNamespace(xl_base_patched=SimpleNamespace(unet=object()),
                                _decode_latent=Mock(side_effect=[MemoryError(), "pixels"]))
        self.assertEqual(namespace['decode_latent'](owner, "latent"), "pixels")
        self.assertEqual(owner._decode_latent.call_count, 2)
        memory.unload_model_and_clones.assert_called_once_with(owner.xl_base_patched.unet)
        for failure, calls in ((ValueError("invalid VAE"), 1), (MemoryError(), 2)):
            owner._decode_latent = Mock(side_effect=failure)
            with self.assertRaises(type(failure)):
                namespace['decode_latent'](owner, "latent")
            self.assertEqual(owner._decode_latent.call_count, calls)

    def test_file_checkpoint_loader_retains_embedded_components_and_fills_missing_ones(self):
        tree = ast.parse((ROOT / "modules/image_pipeline.py").read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "pipeline")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "load_base_model")
        branch = next(node for node in method.body if isinstance(node, ast.If) and 'unet_only' in ast.unparse(node.test))
        function = ast.parse("def load(self, name, filename, hash=None): return unet, clip, vae, clip_vision").body[0]
        function.body = branch.orelse + function.body
        class Clip:
            pass
        class Vae:
            def __init__(self, valid=True):
                self.first_stage_model = object() if valid else None
                self.throw_exception_if_invalid = Mock()
        loader = Mock()
        namespace = dict(load_checkpoint_guess_config=loader, CLIP=Clip, VAE=Vae)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                     "file_checkpoint", "exec"), namespace)
        unet, clip, vae = object(), Clip(), Vae()
        owner = SimpleNamespace(load_base_model=Mock())
        loader.return_value = unet, clip, vae, None
        self.assertEqual(namespace['load'](owner, 'model', 'model.safetensors'), (unet, clip, vae, None))
        loader.assert_called_once_with('model.safetensors')
        owner.load_base_model.assert_not_called()
        vae.throw_exception_if_invalid.assert_called_once()
        for embedded_clip, embedded_vae, expected in ((clip, Vae(False), {'clip': clip}), (None, vae, {'vae': vae})):
            owner.load_base_model.reset_mock()
            loader.return_value = unet, embedded_clip, embedded_vae, None
            self.assertIsNone(namespace['load'](owner, 'model', 'model.safetensors'))
            owner.load_base_model.assert_called_once_with('model', hash=None, unet_only=True,
                                                        input_unet=unet, input_components=expected)

    def test_diffusion_loader_preserves_native_and_quantized_paths(self):
        tree = ast.parse((ROOT / "modules/pipeline_utils.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "load_diffusion_model")
        class NativePatcher:
            pass
        state_loader = Mock(return_value="native weights")
        file_loader = Mock(return_value="file weights")
        gguf = Mock(return_value="quantized weights")
        namespace = dict(Path=Path, ModelPatcher=NativePatcher, load_gguf_model=gguf,
                         comfy=SimpleNamespace(sd=SimpleNamespace(
                             load_diffusion_model_state_dict=state_loader,
                             load_diffusion_model=file_loader)))
        exec(compile(ast.Module(body=[function], type_ignores=[]), "diffusion_loader", "exec"), namespace)
        load = namespace['load_diffusion_model']
        patcher = NativePatcher()
        self.assertIs(load('test.safetensors', patcher), patcher)
        state_loader.assert_not_called()
        file_loader.assert_not_called()
        weights = {'tensor': object()}
        self.assertEqual(load('test.safetensors', weights), 'native weights')
        state_loader.assert_called_once_with(weights)
        self.assertEqual(load('test.safetensors'), 'file weights')
        file_loader.assert_called_once_with('test.safetensors', model_options={})
        self.assertEqual(load('test.GGUF', patcher), 'quantized weights')
        gguf.assert_called_once_with('test.GGUF')

    def test_upscaled_batch_loopback_remains_at_native_size(self):
        tree = ast.parse((ROOT / "modules/image_pipeline.py").read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "pipeline")
        process = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "process")
        start = next(index for index, node in enumerate(process.body) if isinstance(node, ast.Assign)
                     and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                     and node.value.func.attr == "decode_latent")
        # Isolate the real decode/output stage from model loading and sampling.
        function = ast.parse("def finish(self, sampled_latent, gen_data, architecture, callback=None): pass").body[0]
        function.body = process.body[start:]
        namespace = {"np": np, "shared": SimpleNamespace(shared_cache={})}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                     "image_output", "exec"), namespace)
        cache = namespace["shared"].shared_cache
        upscaler = SimpleNamespace(upscale_decoded=lambda owner, image, data:
            image.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2))
        owner = SimpleNamespace(decode_latent=lambda samples: samples)
        native = torch.full((1, 16, 24, 3), 0.5)
        with patch.dict(sys.modules, {"modules.automatic_upscale": upscaler}):
            for _ in range(3):
                result = namespace["finish"](owner, native, {"steps": 1}, "SDXL")
                self.assertEqual(result[0].shape, (32, 48, 3))
                self.assertEqual(cache["prev_image"].shape, (16, 24, 3))
                native = torch.from_numpy(cache["prev_image"].copy()).unsqueeze(0).float() / 255


class MetadataModeTests(unittest.TestCase):
    def save(self, local_only, launch_offline=False, environment_offline=False):
        tree = ast.parse((ROOT / "modules/ui/ui_settings.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "save_clicked")
        settings = SimpleNamespace(default_settings={}, set_settings_path=Mock(), save_settings=Mock(),
                                   settings_path="synthetic-settings.json", name=None)
        store = SimpleNamespace(offline=False)
        models = SimpleNamespace(offline=False)
        namespace = dict(settings=settings, models=models, performance_settings=SimpleNamespace(model_recommendations=store),
            state={"setting_name": ["local_model_metadata"]}, validate_settings=validate_settings,
            path_manager=SimpleNamespace(set_settings_path=Mock(), save_paths=Mock()),
            gr=SimpleNamespace(Info=Mock(), Error=ValueError), t=lambda text, **kwargs: text)
        exec(compile(ast.Module(body=[function], type_ignores=[]), "settings_save", "exec"), namespace)
        with patch.dict(sys.modules, {"argparser": SimpleNamespace(args=SimpleNamespace(offline=launch_offline))}), \
                patch.dict(os.environ, {"RF_OFFLINE": "1" if environment_offline else "0"}):
            namespace["save_clicked"](local_only)
        self.assertEqual(store.offline, models.offline)
        return store.offline

    def test_saving_local_metadata_mode_updates_both_network_clients(self):
        self.assertTrue(self.save(True))
        self.assertFalse(self.save(False))

    def test_saved_settings_cannot_override_launch_offline_mode(self):
        self.assertTrue(self.save(False, launch_offline=True))
        self.assertTrue(self.save(False, environment_offline=True))


if __name__ == "__main__":
    unittest.main()
