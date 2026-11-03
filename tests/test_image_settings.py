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
    def test_only_unchanged_stock_img2img_preset_is_migrated(self):
        import json
        import shutil
        import tempfile
        tree = ast.parse((ROOT / 'modules/controlnet.py').read_text(encoding='utf-8'))
        loader = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_cnsettings')
        defaults = json.loads((ROOT / 'settings/powerup.default').read_text())
        from modules.openai_api import ControlNetRequest
        self.assertEqual(ControlNetRequest(type='img2img').denoise, defaults['Img2Img']['denoise'])
        legacy = {'type': 'img2img', 'start': 0.06, 'denoise': 0.64}
        for preset in (legacy, dict(legacy, denoise=0.5), dict(legacy, loopback=True)):
            with self.subTest(preset=preset), tempfile.TemporaryDirectory() as folder:
                source, saved = Path(folder) / 'defaults.json', Path(folder) / 'saved.json'
                source.write_text(json.dumps(defaults))
                saved.write_text(json.dumps(defaults | {'Img2Img': preset}))
                save = Mock(side_effect=lambda path, data: Path(path).write_text(json.dumps(data)))
                scope = dict(os=os, shutil=shutil, json=json, CNSETTINGS_FILE=str(saved),
                    DEFAULT_CNSETTINGS_FILE=str(source), save_json=save,
                    controlnet_models={value['type']: None for value in defaults.values()})
                exec(compile(ast.Module(body=[loader], type_ignores=[]), 'load_cnsettings', 'exec'), scope)
                expected = defaults['Img2Img'] if preset == legacy else preset
                self.assertEqual(scope['load_cnsettings']()['Img2Img'], expected)
                self.assertEqual(json.loads(saved.read_text())['Img2Img'], expected)
                self.assertEqual(save.call_count, int(preset == legacy))

    def test_decode_oom_releases_diffusion_before_one_retry(self):
        from modules.vae_decode import decode_image_vae
        memory = SimpleNamespace(unload_model_and_clones=Mock(), soft_empty_cache=Mock())
        def raise_non_oom(error):
            if not isinstance(error, MemoryError):
                raise error
        memory.raise_non_oom = raise_non_oom
        decode = Mock(side_effect=[MemoryError(), ['pixels']])
        nodes = SimpleNamespace(VAEDecode=lambda:SimpleNamespace(decode=decode), VAEDecodeTiled=Mock())
        models = SimpleNamespace(unet=object(), vae=SimpleNamespace(device=SimpleNamespace(type='cpu')))
        samples = {'samples':torch.zeros(1, 4, 1, 1)}
        with patch.dict(sys.modules, {'comfy':SimpleNamespace(model_management=memory), 'nodes':nodes}):
            self.assertEqual(decode_image_vae(models, samples), 'pixels')
            self.assertEqual(decode.call_count, 2)
            memory.unload_model_and_clones.assert_called_once_with(models.unet)
            for failure, calls in ((ValueError("invalid VAE"), 1), (MemoryError(), 2)):
                decode.reset_mock(side_effect=True)
                decode.side_effect = failure
                with self.assertRaises(type(failure)):
                    decode_image_vae(models, samples)
                self.assertEqual(decode.call_count, calls)

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

    def test_img2img_batches_encode_the_reference_unless_feedback_is_explicit(self):
        from PIL import Image
        tree = ast.parse((ROOT / 'modules/image_pipeline.py').read_text(encoding='utf-8'))
        process = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'process')
        branch = next(n for n in ast.walk(process) if isinstance(n, ast.If)
                      and ast.unparse(n.test) == 'img2img_mode')
        function = ast.parse('def encode(self, controlnet, gen_data, input_image): pass').body[0]
        function.body = branch.body
        encoded = Mock(return_value=[{'samples': torch.zeros(1, 4, 2, 3)}])
        scope = dict(Image=Image, np=np, torch=torch, VAEEncode=lambda: SimpleNamespace(encode=encoded),
                     shared=SimpleNamespace(state={'preview_count': 9},
                         shared_cache={'prev_image': np.full((16, 24, 3), 255, dtype=np.uint8)}))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                     'img2img_reference', 'exec'), scope)
        owner = SimpleNamespace(xl_base_patched=SimpleNamespace(
            vae=SimpleNamespace(spacial_compression_decode=lambda: 8)))
        reference = torch.zeros(1, 16, 24, 3)
        feedback = np.full((16, 24, 3), 128, dtype=np.uint8)
        for control, data in [({'type': 'img2img'}, {}),
                              ({'type': 'img2img'}, {'_loopback_image': feedback}),
                              ({'type': 'img2img', 'loopback': True}, {})]:
            scope['encode'](owner, control, data, reference)
            self.assertIs(encoded.call_args.kwargs['pixels'], reference)
        scope['encode'](owner, {'type': 'img2img', 'loopback': True},
                        {'_loopback_image': feedback}, reference)
        self.assertTrue(torch.allclose(encoded.call_args.kwargs['pixels'],
                                       torch.full_like(reference, 128 / 255)))

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
        from modules.image_geometry import restore_reference_frame
        namespace = {"np": np, "img2img_mode": True, "controlnet": {"loopback": True},
                     "reference_frame": None, "restore_reference_frame": restore_reference_frame,
                     "guidance": {'vae_tiled':False, 'vae_tile_size':None, 'vae_overlap':None}}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                     "image_output", "exec"), namespace)
        request = {"steps": 1}
        upscaler = SimpleNamespace(upscale_decoded=lambda owner, image, data:
            image.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2))
        owner = SimpleNamespace(decode_latent=lambda samples, **options: samples)
        native = torch.full((1, 16, 24, 3), 0.5)
        with patch.dict(sys.modules, {"modules.automatic_upscale": upscaler}):
            for _ in range(3):
                result = namespace["finish"](owner, native, request, "SDXL")
                self.assertEqual(result[0].shape, (32, 48, 3))
                self.assertEqual(request["_loopback_image"].shape, (16, 24, 3))
                native = torch.from_numpy(request["_loopback_image"].copy()).unsqueeze(0).float() / 255
            namespace['controlnet'] = {}
            independent = {'steps': 1}
            namespace['finish'](owner, native, independent, 'SDXL')
            self.assertNotIn('_loopback_image', independent)


class MetadataModeTests(unittest.TestCase):
    def save(self, local_only, launch_offline=False, environment_offline=False, extra=None):
        tree = ast.parse((ROOT / "modules/ui/ui_settings.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "save_clicked")
        settings = SimpleNamespace(default_settings={}, set_settings_path=Mock(), save_settings=Mock(),
                                   settings_path="synthetic-settings.json", name=None)
        store = SimpleNamespace(offline=False)
        models = SimpleNamespace(offline=False)
        namespace = dict(settings=settings, models=models, performance_settings=SimpleNamespace(model_recommendations=store),
            state={"setting_name": ["local_model_metadata", *(extra or {})]}, validate_settings=validate_settings,
            path_manager=SimpleNamespace(set_settings_path=Mock(), save_paths=Mock()),
            gr=SimpleNamespace(Info=Mock(), Error=ValueError), t=lambda text, **kwargs: text)
        exec(compile(ast.Module(body=[function], type_ignores=[]), "settings_save", "exec"), namespace)
        with patch.dict(sys.modules, {"argparser": SimpleNamespace(args=SimpleNamespace(offline=launch_offline))}), \
                patch.dict(os.environ, {"RF_OFFLINE": "1" if environment_offline else "0"}):
            namespace["save_clicked"](local_only, *(extra or {}).values())
        self.assertEqual(store.offline, models.offline)
        for key, value in (extra or {}).items():
            self.assertEqual(settings.default_settings[key], value)
        return store.offline

    def test_empty_layer_lists_remain_explicit_startup_values(self):
        self.save(False, extra={'moss_image_slg_double_layers':'', 'moss_image_slg_single_layers':''})

    def test_saving_local_metadata_mode_updates_both_network_clients(self):
        self.assertTrue(self.save(True))
        self.assertFalse(self.save(False))

    def test_saved_settings_cannot_override_launch_offline_mode(self):
        self.assertTrue(self.save(False, launch_offline=True))
        self.assertTrue(self.save(False, environment_offline=True))


if __name__ == "__main__":
    unittest.main()
