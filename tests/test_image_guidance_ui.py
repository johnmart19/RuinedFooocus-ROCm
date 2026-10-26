import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from modules.ui import ui_image_guidance as ui


class Widget:
    def __init__(self, *args, **kwargs):
        self.value = kwargs.get('value', args[0] if args else None)
        self.handlers = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def change(self, fn, **kwargs):
        self.handlers.append((fn, kwargs))


class GuidanceUI(unittest.TestCase):
    def test_preset_state_hides_controls_restores_backup_and_drops_private_refs(self):
        current = {key: spec['default'] for key, spec in ui.FIELD_SCHEMA.items()}
        current.update(sampling_shift=3, cfg_norm_strength=2, reference_images=['private'])
        metadata = {'software': 'RuinedFooocus', 'moss_image_edit_format': 'ai-toolkit',
                    'sampling_shift': None, 'effective_sampling_shift': 5}
        entered = ui.preset_control_state(current, {}, metadata, 7, 'MOSSImage', 'preset-moss', 'main-qwen', None)
        self.assertTrue(entered['locked'])
        self.assertTrue(entered['clear_references'])
        self.assertEqual(entered['values']['sampling_shift'], 5)
        self.assertNotIn('reference_images', entered['state']['backup']['values'])
        restored = ui.preset_control_state(entered['values'], {}, {}, 7, 'QwenImage', 'main-qwen', 'main-qwen', entered['state'])
        self.assertFalse(restored['locked'])
        self.assertEqual(restored['values']['sampling_shift'], 3)
        self.assertEqual(restored['values']['cfg_norm_strength'], 2)
        changed = ui.preset_control_state(entered['values'], {}, {}, 7, 'SDXL', 'new-main', 'new-main', entered['state'])
        self.assertIsNone(changed['values']['sampling_shift'])
        self.assertIsNone(changed['values']['cfg_norm_strength'])
        self.assertIsNone(ui.preset_control_state(current, {}, None, 7, 'QwenImage', 'bad', 'main-qwen', entered['state']))

    def test_reference_visibility_respects_model_and_edit_mode(self):
        self.assertFalse(ui.references_visible('MOSSImage', 'Disabled'))
        self.assertTrue(ui.references_visible('MOSSImage', 'ai-toolkit'))
        self.assertTrue(ui.references_visible('MageFlow', 'Disabled'))
        self.assertTrue(ui.references_visible('Fibo2', 'Disabled'))
        self.assertFalse(ui.references_visible('QwenImage', 'Disabled'))
        self.assertEqual(ui.control_visibility('SDXL'), (False, False, False, True))

    def test_controls_register_and_clear_model_specific_values(self):
        registered = {}
        shared = SimpleNamespace(settings=SimpleNamespace(default_settings={}),
                                 add_ctrl=lambda key, obj: registered.update({key: obj}))
        fake = SimpleNamespace(**{name: Widget for name in
            ('Accordion', 'Group', 'Markdown', 'Dropdown', 'Slider', 'Checkbox', 'Number', 'Gallery')},
            update=lambda **values: values)
        base = Widget(value='moss')
        with patch.object(ui, 'gr', fake), patch.object(ui, 'family_for_checkpoint',
                side_effect=lambda name: {'moss': 'MOSSImage', 'sdxl': 'SDXL'}[name]), \
                patch.dict(sys.modules, {'shared': shared}):
            controls = ui.create_ui(base)
            self.assertEqual(set(registered), set(controls))
            self.assertIn('reference_images', registered)
            changes = base.handlers[0][0]('sdxl', 'ai-toolkit')
            self.assertEqual(changes[2], {'visible': False, 'value': None})
            self.assertEqual(changes[3]['value'], None)
            self.assertEqual(changes[4], {'visible': False, 'value': []})
            format_change = controls['moss_image_edit_format'].handlers[0][0]
            self.assertEqual(format_change('moss', 'Disabled'), {'visible': False, 'value': []})
            self.assertEqual(format_change('moss', 'ai-toolkit'), {'visible': True})

    def test_api_snapshot_includes_scalar_guidance_but_never_uploaded_references(self):
        source = Path('webui.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        values = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'image_keys' for t in node.targets))
        keys = ast.literal_eval(values)
        self.assertTrue(set(ui.FIELD_SCHEMA).issubset(keys))
        self.assertNotIn('reference_images', keys)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertTrue(names.isdisjoint({'perf_name', 'perf_save', 'performance_add', 'performance_delete'}))


if __name__ == '__main__':
    unittest.main()
