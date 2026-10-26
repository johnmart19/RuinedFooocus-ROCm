"""Image component status/download policy without model loading or network access."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from unittest.mock import patch

from modules.image_components import IMAGE_COMPONENTS, inventory, download_missing

ROOT = Path(__file__).resolve().parents[1]


def checkpoint(path, entries=None):
    raw = json.dumps(entries or {'weight': {'shape': [1]}}).encode()
    path.write_bytes(len(raw).to_bytes(8, 'little') + raw)
    return path


class ImageComponentTests(unittest.TestCase):
    def test_friendly_cached_moss_name_resolves_with_real_header_shapes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            paths = self.paths(root)
            model = checkpoint(root / 'moss.safetensors', {
                'shared_block_modulation.weight': {'shape': [18432, 3072]},
                'layers.0.proj.ffn_up_gate.weight': {'shape': [16384, 3072]}})
            cache = root / 'cache'
            cache.mkdir()
            (cache / 'moss.json').write_text(json.dumps({'baseModel': 'MOSS Image', 'id': 123}))
            models = SimpleNamespace(get_names=lambda category: ['moss.safetensors'],
                get_file_from_name=lambda category, name: model, cache_paths={'checkpoints': cache})
            # Execute the real cache-selection function without importing Gradio/shared runtime.
            tree = ast.parse((ROOT / 'modules/ui/ui_model_components.py').read_text(encoding='utf-8'))
            selected = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'selected')
            namespace = dict(Path=Path, json=json)
            with patch.dict('sys.modules', {'shared': SimpleNamespace(models=models)}):
                exec(compile(ast.Module(body=[selected], type_ignores=[]), 'component_selection', 'exec'), namespace)
                path, family = namespace['selected']('moss.safetensors')
            self.assertEqual(family, 'MOSS Image')
            state = inventory(path, family, {}, paths)
            self.assertEqual(state['architecture'], 'MOSSImage')
            self.assertEqual([row['key'] for row in state['rows']], ['clip_qwen35_9b', 'vae_flux2'])
            # An already-initialized backend's shape detection wins over stale/friendly metadata.
            with patch.dict('sys.modules', {'comfy.model_detection': SimpleNamespace()}), \
                 patch('modules.model_architecture.detect_tensor_architecture', return_value='MOSSImage') as detect:
                self.assertEqual(inventory(path, 'Qwen Image', {}, paths)['architecture'], 'MOSSImage')
            detect.assert_called_once()
            paths.download_file.assert_not_called()

    def test_friendly_image_family_aliases(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            paths = self.paths(root)
            model = checkpoint(root / 'image.safetensors')
            for family, canonical in [('Qwen Image', 'QwenImage'), ('Mage-Flow', 'MageFlow'),
                                      ('MOSS Image', 'MOSSImage'), ('Z-Image Turbo', 'ZImage'),
                                      ('Krea 2', 'Krea2')]:
                with self.subTest(family=family):
                    self.assertEqual(inventory(model, family, {}, paths)['architecture'], canonical)

    def test_registry_matches_every_production_image_architecture(self):
        tree = ast.parse((ROOT / 'modules/image_pipeline.py').read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'pipeline')
        table = next(node.value for node in cls.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == 'known_model_info' for target in node.targets))
        architectures = set()
        for name, values in zip(table.keys, table.values):
            architectures.add(ast.literal_eval(name))
            keys = {ast.literal_eval(key) for key in values.keys}
            self.assertFalse({'clip_names', 'vae_name'} & keys, 'Component declarations must have one source')
        self.assertEqual(set(IMAGE_COMPONENTS), architectures)
        resolver = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'get_clip_and_vae')
        updates = [node for node in ast.walk(resolver) if isinstance(node, ast.Call)
                   and ast.unparse(node.func) == 'info.update']
        self.assertTrue(any(ast.unparse(node.args[0]) == 'IMAGE_COMPONENTS[unet_type]' for node in updates))

    def paths(self, root):
        for folder in ('clip', 'vae'):
            (root / folder).mkdir()
        catalog = {
            'clip/qwen3vl_4b_bf16.safetensors': {'filename': 'qwen3vl_4b_bf16.safetensors', 'path': 'path_clip', 'url': 'https://example.invalid/encoder'},
            'vae/mage_flow_vae_bf16.safetensors': {'filename': 'mage_flow_vae_bf16.safetensors', 'path': 'path_vae', 'url': 'https://example.invalid/vae'},
            'checkpoints/video': {'filename': 'video.safetensors', 'path': 'path_checkpoints', 'url': 'https://example.invalid/video'},
        }
        return SimpleNamespace(paths={'path_clip': str(root / 'clip'), 'path_vae': str(root / 'vae')},
                               get_abspath=Path, DOWNLOADABLE_FILES=catalog, download_file=Mock())

    def test_status_and_explicit_download_only_selected_missing_components(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            paths = self.paths(root)
            model = checkpoint(root / 'image.safetensors')
            (root / 'vae/mage_flow_vae_bf16.safetensors').write_bytes(b'installed')
            state = inventory(model, 'MageFlow', {}, paths)
            self.assertEqual([row['status'] for row in state['rows']], ['Missing', 'Installed'])
            paths.download_file.assert_not_called()
            self.assertEqual(download_missing(model, 'MageFlow', {}, paths), ['qwen3vl_4b_bf16.safetensors'])
            paths.download_file.assert_called_once()
            self.assertEqual(paths.download_file.call_args.args[0], 'clip/qwen3vl_4b_bf16.safetensors')
            paths.download_file.reset_mock()
            with self.assertRaisesRegex(ValueError, 'Offline'):
                download_missing(model, 'MageFlow', {}, paths, offline=True)
            paths.download_file.assert_not_called()

    def test_embedded_components_and_custom_overrides_do_not_trigger_downloads(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            paths = self.paths(root)
            model = checkpoint(root / 'image.safetensors', {'cond_stage_model.weight': {'shape': [1]},
                                                          'first_stage_model.weight': {'shape': [1]}})
            rows = inventory(model, 'MageFlow', {}, paths)['rows']
            self.assertTrue(all(row['status'] == 'Embedded in checkpoint' for row in rows))
            self.assertEqual(download_missing(model, 'MageFlow', {}, paths), [])
            checkpoint(model)
            rows = inventory(model, 'MageFlow', {'clip_qwen3vl_4b': '../outside.safetensors'}, paths)['rows']
            self.assertTrue(rows[0]['override'])
            self.assertFalse(rows[0]['download_available'])

    def test_video_headers_and_video_families_are_refused_even_with_image_label(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            paths = self.paths(root)
            model = checkpoint(root / 'video.safetensors', {'head.modulation': {'shape': [1]},
                                                          'patch_embedding.weight': {'shape': [1]}})
            with self.assertRaisesRegex(ValueError, 'image checkpoints only'):
                download_missing(model, 'MageFlow', {}, paths)
            checkpoint(model)
            with self.assertRaisesRegex(ValueError, 'image checkpoints only'):
                download_missing(model, 'Wan Video', {}, paths)
            paths.download_file.assert_not_called()

    def test_gguf_video_metadata_is_refused_without_tensor_runtime(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            paths = self.paths(root)
            model = root / 'video.gguf'
            model.write_bytes(b'GGUF')
            mmap = Mock()
            reader = SimpleNamespace(get_field=lambda key: SimpleNamespace(contents=lambda: 'wan'),
                                     tensors=[], data=SimpleNamespace(_mmap=mmap))
            with patch.dict('sys.modules', {'gguf': SimpleNamespace(GGUFReader=lambda path: reader)}):
                with self.assertRaisesRegex(ValueError, 'image checkpoints only'):
                    download_missing(model, 'MageFlow', {}, paths)
            mmap.close.assert_called_once()
            paths.download_file.assert_not_called()

    def test_gguf_header_identity_overrides_stale_image_family(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            paths = self.paths(root)
            model = root / 'image.gguf'
            model.write_bytes(b'GGUF')
            mmap = Mock()
            tensor = SimpleNamespace(name='shared_block_modulation.weight',
                                     shape=SimpleNamespace(tolist=lambda: [3072, 18432]))
            reader = SimpleNamespace(get_field=lambda key: SimpleNamespace(contents=lambda: 'moss_image'),
                                     tensors=[tensor], data=SimpleNamespace(_mmap=mmap))
            with patch.dict('sys.modules', {'gguf': SimpleNamespace(GGUFReader=lambda path: reader)}):
                state = inventory(model, 'MageFlow', {}, paths)
            self.assertEqual(state['architecture'], 'MOSSImage')
            self.assertEqual(state['rows'][0]['key'], 'clip_qwen35_9b')
            # Initialized backend shapes are authoritative even over an inaccurate GGUF tag.
            with patch.dict('sys.modules', {'gguf': SimpleNamespace(GGUFReader=lambda path: reader),
                                            'comfy.model_detection': SimpleNamespace()}), \
                 patch('modules.model_architecture.detect_tensor_architecture', return_value='QwenImage') as detect:
                self.assertEqual(inventory(model, 'MageFlow', {}, paths)['architecture'], 'QwenImage')
            self.assertEqual(detect.call_args.args[0]['shared_block_modulation.weight']['shape'], [18432, 3072])
            reader.get_field = lambda key: SimpleNamespace(contents=lambda: 'lumina2')
            with patch.dict('sys.modules', {'gguf': SimpleNamespace(GGUFReader=lambda path: reader)}):
                with self.assertRaisesRegex(ValueError, 'cannot be identified safely'):
                    download_missing(model, 'MageFlow', {}, paths)
            paths.download_file.assert_not_called()


if __name__ == '__main__':
    unittest.main()
