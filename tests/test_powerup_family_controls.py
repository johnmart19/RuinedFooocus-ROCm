"""Exercise production PowerUp family policy without starting Gradio or Torch."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'modules/ui/ui_controlnet.py'


def functions():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    names = {'checkpoint_features', 'control_types', 'input_label', 'powerup_choices'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
             or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'SDXL_CONTROLS' for target in node.targets)]
    models = SimpleNamespace(get_models_by_path=Mock(return_value={}), get_model_base=Mock(return_value='Pony'),
        get_file=Mock(return_value=Path('model.safetensors')), detect_checkpoint_base=Mock(return_value='SDXL 1.0'))
    controls = SimpleNamespace(controlnet_models=dict.fromkeys(('canny', 'depth', 'sketch', 'recolour', 'img2img', 'upscale', 'rembg', 'faceswap')),
        cn_options={'Img2Img': {'type': 'img2img'}, 'Canny': {'type': 'canny'}, 'Depth': {'type': 'depth'},
        'Sketch': {'type': 'sketch'}, 'Recolour': {'type': 'recolour'}, 'Remove background': {'type': 'rembg'}})
    namespace = {'shared': SimpleNamespace(models=models, settings=SimpleNamespace(default_settings={})), 'controlnet': controls,
                 'VIDEO_FPS': {}, 'NEWCN': 'New...', 't': lambda value: value}
    guidance = ast.parse((ROOT / 'modules/ui/ui_image_guidance.py').read_text(encoding='utf-8'))
    family = next(n for n in guidance.body if isinstance(n, ast.FunctionDef) and n.name == 'family_for_checkpoint')
    family_scope = {}
    exec(compile(ast.Module(body=[family], type_ignores=[]), 'family_policy', 'exec'), family_scope)
    def resolve(name):
        with patch.dict('sys.modules', {'shared': namespace['shared']}):
            return family_scope['family_for_checkpoint'](name)
    namespace['family_for_checkpoint'] = resolve
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace, models


class PowerUpFamilyTests(unittest.TestCase):
    def test_local_family_aliases_keep_native_edits_and_sdxl_controls_without_fetch(self):
        scope, models = functions()
        models.detect_checkpoint_base.return_value = None
        for family, expected_sdxl, expected_edit in [('Qwen Image', False, True),
            ('Mage-Flow', False, True), ('MOSS Image', False, True),
            ('Pony XL', True, False), ('Animagine XL 3.1', True, False)]:
            with self.subTest(family=family):
                models.get_model_base.return_value = family
                features = scope['checkpoint_features']('selected', 'ai-toolkit')
                self.assertEqual(features['sdxl'], expected_sdxl)
                self.assertEqual(features['native_edit'], expected_edit)
        models.get_models_by_path.assert_called_with('checkpoints', 'selected', fetch=False)

    def test_saved_upscaler_is_editable_and_rembg_has_no_ignored_sliders(self):
        scope, _ = functions()
        scope['controlnet'].cn_options['Custom upscale'] = {'type': 'upscale', 'upscaler': 'old.pth'}
        scope.update(gr=SimpleNamespace(update=lambda **kwargs: kwargs), cn_sliders=[None] * 6, cn_outputs=[None] * 3)
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        for name in ('cn_changed', 'cn_type_changed'):
            callback = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
            callback.decorator_list = []
            exec(compile(ast.Module(body=[callback], type_ignores=[]), str(SOURCE), 'exec'), scope)
        self.assertEqual(scope['cn_changed']('Custom upscale', 'Upscale', 'model')[-1],
                         {'visible': True, 'value': 'old.pth'})
        self.assertTrue(all(update['visible'] in (False, 'hidden') for update in
                            scope['cn_type_changed']('Rembg', 'New...', 'model')))
        function = next(node for node in ast.parse((ROOT / 'modules/controlnet.py').read_text(encoding='utf-8')).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'get_settings')
        preset = {'type': 'upscale', 'upscaler': 'old.pth'}
        namespace = {'controlnet_settings': {'Custom upscale': preset}, 'NEWCN': 'New...'}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'control_settings', 'exec'), namespace)
        result = namespace['get_settings']({'cn_selection': 'Custom upscale', 'cn_upscale': 'new.pth'})
        self.assertEqual(result['upscaler'], 'new.pth')
        self.assertEqual(preset['upscaler'], 'old.pth')

    def test_sdxl_aliases_use_backend_architecture_and_keep_controls(self):
        scope, models = functions()
        for alias in ('Pony', 'Illustrious', 'NoobAI', 'SDXL 1.0'):
            models.get_model_base.return_value = alias
            features = scope['checkpoint_features']('selected')
            self.assertTrue(features['sdxl'])
            choices = dict(scope['powerup_choices'](**features))
            self.assertIn('Canny', choices)
            self.assertIn('Depth', choices)
        models.detect_checkpoint_base.assert_called_with(Path('model.safetensors'))
        models.get_models_by_path.assert_called_with('checkpoints', 'selected', fetch=False)
        # Real tensor architecture takes precedence over misleading cached metadata.
        models.detect_checkpoint_base.return_value = 'QwenImage'
        self.assertFalse(scope['checkpoint_features']('selected')['sdxl'])

    def test_native_edits_keep_upload_route_and_hide_sdxl_controls(self):
        scope, models = functions()
        for family in ('MageFlow', 'Fibo2', 'QwenImage', 'MOSSImage'):
            models.get_model_base.return_value = family
            models.detect_checkpoint_base.return_value = family
            features = scope['checkpoint_features']('selected', 'ai-toolkit')
            self.assertFalse(features['sdxl'])
            self.assertEqual(scope['input_label'](features), 'Image edit reference')
            choices = scope['powerup_choices'](**features)
            self.assertIn(('Image edit reference', 'Img2Img'), choices)
            self.assertFalse({'Canny', 'Depth', 'Sketch', 'Recolour'} & {value for _, value in choices})
            self.assertIn('Img2img', scope['control_types'](features))
            self.assertNotIn('Canny', scope['control_types'](features))

    def test_model_change_resets_inpainting_and_unsupported_selection(self):
        scope, models = functions()
        models.get_model_base.return_value = 'QwenImage'
        models.detect_checkpoint_base.return_value = 'QwenImage'
        scope['gr'] = SimpleNamespace(update=lambda **kwargs: kwargs)
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        callback = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'model_changed')
        exec(compile(ast.Module(body=[callback], type_ignores=[]), str(SOURCE), 'exec'), scope)
        result = scope['model_changed']('selected', 'Canny')
        self.assertEqual(result[0]['value'], 'None')
        self.assertEqual(result[1]['label'], 'Image edit reference')
        self.assertEqual(result[2], {'value': False, 'visible': False})
        self.assertNotIn('Canny', result[3]['choices'])

    def test_moss_format_toggle_updates_presentation_without_clearing_inputs(self):
        scope, models = functions()
        models.get_model_base.return_value = 'MOSSImage'
        models.detect_checkpoint_base.return_value = 'MOSSImage'
        scope.update(gr=SimpleNamespace(update=lambda **kwargs: kwargs), cn_sliders=[None] * 6, cn_outputs=[None] * 3)
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        for name in ('cn_changed', 'edit_format_changed'):
            callback = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
            callback.decorator_list = []
            exec(compile(ast.Module(body=[callback], type_ignores=[]), str(SOURCE), 'exec'), scope)
        disabled = scope['checkpoint_features']('selected', 'Disabled')
        self.assertFalse(disabled['native_edit'])
        self.assertEqual(scope['input_label'](disabled), 'Image to image')
        for edit_format in ('ai-toolkit', 'Moss-Image-Edit', 'Disabled'):
            with self.subTest(edit_format=edit_format):
                updates = scope['edit_format_changed'](edit_format, 'selected', 'Img2Img', 'Img2img')
                self.assertNotIn('value', updates[0])  # selected saved preset remains
                self.assertNotIn('value', updates[1])  # uploaded pixels remain
                self.assertNotIn('value', updates[2])  # current denoise remains
                self.assertEqual(updates[2]['visible'], edit_format == 'Disabled')
                self.assertEqual(updates[1]['label'], 'Image to image' if edit_format == 'Disabled' else 'Image edit reference')
                self.assertIn(('Image to image' if edit_format == 'Disabled' else 'Image edit reference', 'Img2Img'), updates[0]['choices'])
                changed = scope['cn_changed']('Img2Img', 'Img2img', 'selected', edit_format)
                self.assertEqual(changed[5]['visible'], True if edit_format == 'Disabled' else 'hidden')

    def test_checkpoint_tools_are_not_an_image_operation(self):
        scope, _ = functions()
        choices = scope['powerup_choices'](False)
        self.assertNotIn('Checkpoint tools', {value for _, value in choices})
        tree = ast.parse((ROOT / 'modules/ui/ui_checkpoint_tools.py').read_text(encoding='utf-8'))
        create = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'create_ui')
        gr = MagicMock()
        namespace = {'gr': gr, 'shared': SimpleNamespace(models=SimpleNamespace(get_names=lambda category: ['model']))}
        exec(compile(ast.Module(body=[create], type_ignores=[]), 'checkpoint_panel', 'exec'), namespace)
        namespace['create_ui'](SimpleNamespace(value='model', change=Mock()))
        gr.Column.assert_called_once_with(visible=True, elem_id='checkpoint-tools')

    def test_saved_canny_exposes_thresholds_and_generation_does_not_mutate_preset(self):
        scope, _ = functions()
        scope['controlnet'].cn_options['Canny'].update(edge_low=.15, edge_high=.75)
        scope.update(gr=SimpleNamespace(update=lambda **kwargs: kwargs), cn_sliders=[None] * 6, cn_outputs=[None] * 3)
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        callback = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'cn_changed')
        callback.decorator_list = []
        exec(compile(ast.Module(body=[callback], type_ignores=[]), str(SOURCE), 'exec'), scope)
        values = scope['cn_changed']('Canny', 'Canny', 'selected')
        self.assertEqual(values[6], {'visible': True, 'value': .15})
        self.assertEqual(values[7], {'visible': True, 'value': .75})
        tree = ast.parse((ROOT / 'modules/controlnet.py').read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'get_settings')
        preset = {'type': 'canny', 'edge_low': .15, 'edge_high': .75, 'strength': 1.}
        namespace = {'controlnet_settings': {'Canny': preset}, 'NEWCN': 'New...'}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'control_settings', 'exec'), namespace)
        result = namespace['get_settings']({'cn_selection': 'Canny', 'cn_edge_low': 0., 'cn_edge_high': .9})
        self.assertEqual(result['edge_low'], 0.)
        self.assertEqual(result['edge_high'], .9)
        self.assertEqual(preset['edge_low'], .15)
        self.assertEqual(preset['edge_high'], .75)

    def test_native_reference_route_does_not_show_ignored_denoise(self):
        scope, models = functions()
        models.get_model_base.return_value = 'MageFlow'
        models.detect_checkpoint_base.return_value = 'MageFlow'
        scope.update(gr=SimpleNamespace(update=lambda **kwargs: kwargs), cn_sliders=[None] * 6, cn_outputs=[None] * 3)
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        callback = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'cn_changed')
        callback.decorator_list = []
        exec(compile(ast.Module(body=[callback], type_ignores=[]), str(SOURCE), 'exec'), scope)
        result = scope['cn_changed']('Img2Img', 'Img2img', 'selected')
        self.assertEqual(result[5]['visible'], 'hidden')


if __name__ == '__main__':
    unittest.main()
