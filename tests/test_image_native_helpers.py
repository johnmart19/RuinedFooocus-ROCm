"""Native preview selection/fallback and image-upscale error boundaries."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]


def function(path, name, scope):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), scope)
    return scope[name]


class NativeImageHelperTests(unittest.TestCase):
    def test_component_variants_use_native_config_without_reading_state_dicts(self):
        from modules.image_components import IMAGE_COMPONENTS
        from modules.resource_defaults import resource_name
        tree = ast.parse((ROOT / 'modules/image_pipeline.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'pipeline')
        nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_clip_and_vae'
                 or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'known_model_info' for t in n.targets)]
        scope = dict(comfy=Mock(), settings=SimpleNamespace(default_settings={}))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'image_components', 'exec'), scope)
        owner = SimpleNamespace(known_model_info=scope['known_model_info'])
        for name, config, family in [('Lumina2', {'dim': 2304}, 'Lumina2'),
            ('Lumina2', {'dim': 3840}, 'ZImage'), ('Lumina2', {'clip_text_dim': 1024}, 'NewBieImage'),
            ('Flux2', {'depth_single_blocks': 48}, 'Flux2'),
            ('Flux2', {'depth_single_blocks': 24}, 'Flux2Klein9B'),
            ('Flux2', {'depth_single_blocks': 20}, 'Flux2Klein4B'), ('MOSSImage', {}, 'MOSSImage')]:
            with self.subTest(family=family):
                model = type(name, (), {})()
                model.model_config = SimpleNamespace(unet_config=config)
                patcher = SimpleNamespace(model=model, model_state_dict=Mock(side_effect=AssertionError('Must not read tensors')))
                result = scope['get_clip_and_vae'](owner, patcher)
                expected = IMAGE_COMPONENTS[family]
                self.assertEqual(result['clip_names'], [resource_name(key, {}) for key in expected['clip_names']])
                self.assertEqual(result['vae_name'], resource_name(expected['vae_name'], {}))
                patcher.model_state_dict.assert_not_called()

    def preview(self, method='Auto', files=()):
        methods = SimpleNamespace(**{n:n for n in ('Auto', 'NoPreviews', 'TAESD', 'Latent2RGB')})
        args = SimpleNamespace(preview_method=method)
        paths = SimpleNamespace(add_model_folder_path=Mock(), get_filename_list=Mock(return_value=files))
        native = SimpleNamespace(get_previewer=Mock(), TAEHVPreviewerImpl=type('Temporal', (), {}))
        scope = dict(args=args, LatentPreviewMethod=methods, folder_paths=paths,
                     latent_preview=native, logging=Mock())
        run = function('modules/pipeline_utils.py', 'get_previewer', scope)
        return run, args, paths, native

    def test_disabled_preview_never_loads_or_registers_a_decoder(self):
        run, _, paths, native = self.preview('NoPreviews')
        self.assertIsNone(run('cpu', SimpleNamespace()))
        native.get_previewer.assert_not_called()
        paths.add_model_folder_path.assert_not_called()

    def test_auto_uses_native_decoder_when_installed_and_restores_selection(self):
        for files, expected in [((), 'Auto'), (['taesdxl_decoder.pth'], 'TAESD')]:
            run, args, paths, native = self.preview(files=files)
            selected = SimpleNamespace(decode_latent_to_preview=Mock(return_value='native image'))
            def load(device, latent_format):
                self.assertEqual(args.preview_method, expected)
                return selected
            native.get_previewer.side_effect = load
            shared = SimpleNamespace(path_manager=SimpleNamespace(model_paths={'vae_approx_path':'custom-previews'}))
            with patch.dict('sys.modules', {'shared': shared}):
                previewer = run('cpu', SimpleNamespace(taesd_decoder_name='taesdxl'))
            self.assertIs(previewer, selected)
            self.assertEqual(args.preview_method, 'Auto')
            self.assertEqual(previewer.preview('latent', 1, 2), 'native image')
            selected.decode_latent_to_preview.assert_called_once_with('latent')
            paths.add_model_folder_path.assert_called_once_with('vae_approx', 'custom-previews', is_default=True)

    def test_corrupt_decoder_falls_back_natively_and_restores_user_mode_on_failure(self):
        for fallback_fails in (False, True):
            run, args, _, native = self.preview('TAESD')
            def load(device, latent_format):
                if args.preview_method == 'TAESD' or fallback_fails:
                    raise ValueError('bad decoder')
                self.assertEqual(args.preview_method, 'Latent2RGB')
                return None
            native.get_previewer.side_effect = load
            shared = SimpleNamespace(path_manager=SimpleNamespace(model_paths={'vae_approx_path':'previews'}))
            with patch.dict('sys.modules', {'shared': shared}):
                if fallback_fails:
                    with self.assertRaisesRegex(ValueError, 'bad decoder'):
                        run('cpu', SimpleNamespace(taesd_decoder_name='taesdxl'))
                else:
                    self.assertIsNone(run('cpu', SimpleNamespace(taesd_decoder_name='taesdxl')))
            self.assertEqual(args.preview_method, 'TAESD')
            self.assertEqual(native.get_previewer.call_count, 2)

    def test_upscale_errors_reach_worker_and_success_preserves_rgba(self):
        scope = dict(torch=torch, np=np, worker=Mock(),
                     modules=SimpleNamespace(controlnet=SimpleNamespace(get_settings=lambda data:
                         {'upscaler': 'installed-model'})),
                     InterruptibleUpscaler=lambda model, data: model)
        run = function('modules/upscale_pipeline.py', 'process', scope)
        owner = SimpleNamespace(load_upscaler_model=Mock(return_value='native model'))
        source = Image.new('RGBA', (3, 2), (255, 128, 0, 64))
        data = dict(input_image=source, task_id=1)
        execute = Mock()
        native = SimpleNamespace(ImageUpscaleWithModel=SimpleNamespace(execute=execute))
        with patch.dict('sys.modules', {'comfy_extras.nodes_upscale_model': native}):
            for error in (ValueError('invalid weights'), torch.OutOfMemoryError('GPU full')):
                execute.side_effect = error
                with self.assertRaises(type(error)):
                    run(owner, data)
            def upscale(model, image):
                self.assertEqual(model, 'native model')
                self.assertEqual(tuple(image.shape), (1, 2, 3, 4))
                return (image.repeat_interleave(2, 1).repeat_interleave(2, 2),)
            execute.side_effect = upscale
            result = run(owner, data)[0]
        self.assertEqual(result.shape, (4, 6, 4))
        np.testing.assert_array_equal(result[0, 0], [255, 128, 0, 64])


if __name__ == '__main__':
    unittest.main()
