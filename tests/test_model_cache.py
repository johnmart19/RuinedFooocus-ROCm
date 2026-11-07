from ruinedfooocus.core.paths import asset_path, resource_path

"""Exercise production loader caches with local files and mocked native loaders."""
import ast
import builtins
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import ruinedfooocus.features.models.model_identity as model_identity
from ruinedfooocus.core.config_io import save_json

ROOT = Path(__file__).resolve().parents[1]


def method(source, name, scope):
    tree = ast.parse((ROOT / source).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(
        vars(model_identity), asset_path=asset_path, resource_path=resource_path, **scope
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), source, "exec"), namespace)
    return namespace[name]


class Clip:
    pass


class Vae:
    def __init__(self):
        self.first_stage_model = object()
        self.patcher = SimpleNamespace()
        self.throw_exception_if_invalid = Mock()


class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def file(self, name):
        path = self.root / name
        path.write_bytes(b"local test weights")
        return path

    def replace(self, path):
        before = path.stat()
        path.write_bytes(b"replacement weights")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 10000000))

    def base_loader(self, checkpoint, bare=False, defaults=None):
        clip, vae = Clip(), Vae()
        unet = SimpleNamespace(model=type("SDXL", (), {})())
        backend = SimpleNamespace(
            model_management=Mock(),
            sd=SimpleNamespace(
                load_clip=Mock(return_value=clip),
                VAE=Mock(return_value=vae),
                load_vae_patcher=Mock(),
            ),
            utils=SimpleNamespace(load_torch_file=Mock(return_value=({}, {}))),
        )
        backend.model_management.get_torch_device.return_value = SimpleNamespace(type="cpu")
        files = Mock(get_model_path=Mock(return_value=checkpoint))
        loader = Mock(return_value=(unet, clip, vae, None))
        diffusion = Mock(return_value=unet)
        owner = SimpleNamespace(
            xl_base_hash=None,
            xl_base=None,
            StableDiffusionModel=lambda **values: SimpleNamespace(**values),
            known_models={"SDXL"},
            get_clip_and_vae=Mock(
                return_value=dict(
                    options={},
                    clip_names=["encoder.safetensors"],
                    vae_name="vae.safetensors",
                    clip_type="native",
                )
            ),
        )
        import torch

        run = method(
            "ruinedfooocus/features/generation/image_pipeline.py",
            "load_base_model",
            dict(
                settings=SimpleNamespace(default_settings={} if defaults is None else defaults),
                shared=SimpleNamespace(models=files),
                Path=Path,
                os=os,
                gc=Mock(),
                comfy=backend,
                torch=torch,
                load_checkpoint_guess_config=loader,
                load_diffusion_model=diffusion,
                CLIP=Clip,
                VAE=Vae,
                traceback=Mock(),
                path_manager=SimpleNamespace(
                    model_paths={"clip_path": str(self.root), "vae_path": str(self.root)},
                    get_folder_file_path=lambda folder, name, **kwargs: self.root / name,
                ),
            ),
        )
        owner.load_base_model = MethodType(run, owner)
        attention = SimpleNamespace(configure_qwen_attention=Mock())
        context = patch.dict(
            "sys.modules", {"ruinedfooocus.backends.comfy.model_attention": attention}
        )
        context.start()
        self.addCleanup(context.stop)

        def load():
            owner.load_base_model("checkpoint.safetensors", unet_only=bare)

        return load, owner, diffusion if bare else loader, backend, files

    def test_checkpoint_cache_reloads_replaced_weights_but_reuses_unchanged_weights(self):
        checkpoint = self.file("checkpoint.safetensors")
        load, owner, native, _, _ = self.base_loader(checkpoint)
        load()
        load()
        self.assertEqual(native.call_count, 1)
        self.replace(checkpoint)
        load()
        self.assertEqual(native.call_count, 2)

    def test_decode_preferences_do_not_reload_weights_but_component_overrides_do(self):
        checkpoint = self.file("checkpoint.safetensors")
        defaults = {"clip_l": "encoder.safetensors", "vae_tile_size": None}
        load, _, native, _, _ = self.base_loader(checkpoint, defaults=defaults)
        load()
        defaults.update(vae_tiled=True, vae_tile_size=256, vae_overlap=0, clip_skip=2)
        load()
        self.assertEqual(native.call_count, 1)
        defaults["clip_l"] = "other-encoder.safetensors"
        load()
        self.assertEqual(native.call_count, 2)

    def test_removed_checkpoint_cannot_reuse_the_old_model(self):
        checkpoint = self.file("checkpoint.safetensors")
        load, owner, native, _, files = self.base_loader(checkpoint)
        load()
        checkpoint.unlink()
        files.get_model_path.return_value = None
        load()
        self.assertIsNone(owner.xl_base)
        self.assertIsNone(owner.xl_base_hash)

    def test_a_different_resolved_checkpoint_path_invalidates_a_matching_name(self):
        checkpoint = self.file("checkpoint.safetensors")
        load, _, native, _, files = self.base_loader(checkpoint)
        load()
        other = self.file("other.safetensors")
        files.get_model_path.return_value = other
        load()
        self.assertEqual(native.call_count, 2)
        self.assertEqual(native.call_args.args[0], str(other))

    def hash_reader(self):
        owner = SimpleNamespace(cache_paths={"checkpoints": self.root / "cache/checkpoints"})
        run = method(
            "ruinedfooocus/features/models/model_handler.py",
            "model_sha256",
            dict(
                hashlib=hashlib,
                json=json,
                save_json=save_json,
                t=lambda text, mapping: text.format(**mapping),
            ),
        )
        return lambda path: run(owner, path)

    def test_model_hash_cache_reuses_stable_files_and_rehashes_replacements(self):
        path = self.file("checkpoint.safetensors")
        digest = self.hash_reader()
        expected = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        self.assertEqual(digest(path), expected)
        with patch(
            "builtins.open", side_effect=AssertionError("cached weights must not be reread")
        ):
            self.assertEqual(digest(path), expected)
        self.replace(path)
        self.assertEqual(digest(path), hashlib.sha256(path.read_bytes()).hexdigest().upper())

    def test_a_file_changing_during_hashing_is_not_published_as_a_valid_hash(self):
        path = self.file("checkpoint.safetensors")
        original_open = builtins.open
        changed = False

        @contextmanager
        def changing_reader(name, mode):
            nonlocal changed
            with original_open(name, mode) as source:

                def read(size):
                    nonlocal changed
                    block = source.read(size)
                    if block and not changed:
                        self.replace(path)
                        changed = True
                    return block

                yield SimpleNamespace(read=read)

        with patch("builtins.open", changing_reader):
            self.assertIsNone(self.hash_reader()(path))
        self.assertTrue(changed)
        self.assertEqual(list((self.root / "cache/hashes").glob("*.json")), [])

    def test_replacing_an_external_encoder_or_vae_invalidates_the_base_cache(self):
        checkpoint = self.file("checkpoint.safetensors")
        encoder, vae = self.file("encoder.safetensors"), self.file("vae.safetensors")
        load, _, native, backend, _ = self.base_loader(checkpoint, bare=True)
        load()
        load()
        self.assertEqual(native.call_count, 1)
        for file in (encoder, vae):
            self.replace(file)
            load()
        self.assertEqual(native.call_count, 3)
        self.assertEqual(backend.sd.load_clip.call_count, 3)

    def test_failed_component_configuration_does_not_leave_a_successful_cache_entry(self):
        checkpoint = self.file("checkpoint.safetensors")
        load, owner, native, backend, _ = self.base_loader(checkpoint)
        owner.get_clip_and_vae.side_effect = [ValueError("configuration failed"), {"options": {}}]
        with self.assertRaisesRegex(ValueError, "configuration failed"):
            load()
        self.assertIsNone(owner.xl_base_hash)
        load()
        self.assertEqual(native.call_count, 2)
        self.assertEqual(owner.xl_base_hash, "checkpoint.safetensors")

    def test_lora_cache_reloads_replaced_weights_and_starts_from_the_base(self):
        path = self.file("style.safetensors")
        # A compatible LoRA adds native patches; incompatible ones are rejected by load_loras.
        native = Mock(
            return_value=(
                SimpleNamespace(patches={"diffusion_model.w": [object()]}),
                SimpleNamespace(patcher=SimpleNamespace(patches={})),
            )
        )
        reader = Mock(return_value="lora weights")
        resolve = Mock(return_value=path)
        owner = SimpleNamespace(
            xl_base_patched_hash=None,
            xl_base=SimpleNamespace(
                unet="base unet", clip="base clip", vae="vae", clip_vision=None
            ),
            StableDiffusionModel=lambda **values: SimpleNamespace(**values),
        )
        run = method(
            "ruinedfooocus/features/generation/image_pipeline.py",
            "load_loras",
            dict(
                shared=SimpleNamespace(models=SimpleNamespace(get_model_path=resolve)),
                comfy=SimpleNamespace(
                    utils=SimpleNamespace(load_torch_file=reader),
                    sd=SimpleNamespace(load_lora_for_models=native),
                ),
            ),
        )
        request = [{"name": path.name, "weight": 0.7}]
        run(owner, request)
        run(owner, request)
        self.assertEqual(native.call_count, 1)
        self.replace(path)
        run(owner, request)
        self.assertEqual(native.call_count, 2)
        for call in native.call_args_list:
            self.assertEqual(call.args[:2], ("base unet", "base clip"))
        run(owner, [{"name": "missing.safetensors", "weight": 0}])
        self.assertIs(owner.xl_base_patched, owner.xl_base)
        self.assertEqual(native.call_count, 2)

    def test_automatic_upscale_checks_the_shared_loader_on_each_image(self):
        from ruinedfooocus.features.generation.automatic_upscale import upscale_decoded

        model = SimpleNamespace(scale=2)
        loader = SimpleNamespace(load_upscaler_model=Mock(return_value=model))
        execute = Mock(return_value=["upscaled"])
        native = SimpleNamespace(ImageUpscaleWithModel=SimpleNamespace(execute=execute))
        with patch.dict(
            "sys.modules",
            {
                "ruinedfooocus.features.image_processing.upscale_pipeline": SimpleNamespace(
                    pipeline=lambda: loader, InterruptibleUpscaler=lambda model, data: model
                ),
                "ruinedfooocus.app.async_worker": Mock(),
                "comfy_extras.nodes_upscale_model": native,
            },
        ):
            owner = SimpleNamespace()
            data = dict(automatic_upscale="2× General (RealESRGAN)", task_id=1)
            self.assertEqual(upscale_decoded(owner, "pixels", data), "upscaled")
            self.assertEqual(upscale_decoded(owner, "pixels", data), "upscaled")
        self.assertEqual(loader.load_upscaler_model.call_count, 2)

    def test_controlnet_cache_reloads_replaced_weights(self):
        path = self.file("depth.safetensors")
        native = Mock(return_value=object())
        run = method(
            "ruinedfooocus/features/generation/image_pipeline.py",
            "refresh_controlnet",
            dict(
                control_tools=SimpleNamespace(get_model=lambda name: path),
                comfy=SimpleNamespace(controlnet=SimpleNamespace(load_controlnet=native)),
            ),
        )
        owner = SimpleNamespace(xl_controlnet=None, xl_controlnet_hash=None)
        run(owner, "depth")
        run(owner, "depth")
        self.assertEqual(native.call_count, 1)
        self.replace(path)
        run(owner, "depth")
        self.assertEqual(native.call_count, 2)

    def test_upscaler_cache_reloads_replaced_weights(self):
        path = self.file("upscaler.pth")
        native = Mock(return_value=[object()])
        run = method(
            "ruinedfooocus/features/image_processing/upscale_pipeline.py",
            "load_upscaler_model",
            dict(
                path_manager=SimpleNamespace(
                    get_file_path=lambda *args, **kwargs: path,
                    model_paths={"upscaler_path": str(self.root)},
                ),
                folder_paths=Mock(),
                Path=Path,
                os=os,
            ),
        )
        owner = SimpleNamespace()
        with patch.dict(
            "sys.modules",
            {
                "comfy_extras.nodes_upscale_model": SimpleNamespace(
                    UpscaleModelLoader=SimpleNamespace(execute=native)
                )
            },
        ):
            self.assertIs(run(owner, path.name), run(owner, path.name))
            self.assertEqual(native.call_count, 1)
            self.replace(path)
            run(owner, path.name)
        self.assertEqual(native.call_count, 2)


if __name__ == "__main__":
    unittest.main()
