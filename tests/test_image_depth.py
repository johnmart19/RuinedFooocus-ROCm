"""Depth preparation, cache and UI contracts; no accelerator execution."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from modules.depth_control import DEPTH_DEFAULTS, depth_settings, prepare_depth, depth_model_name

ROOT = Path(__file__).resolve().parents[1]


def function(path, name, **scope):
    tree = ast.parse((ROOT/path).read_text(encoding='utf-8'))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), scope)
    return scope[name]


class DepthTests(unittest.TestCase):
    def test_old_maps_and_new_api_defaults_are_distinct_and_validated(self):
        from modules.openai_api import ControlNetRequest
        from pydantic import ValidationError
        self.assertEqual(depth_settings({})['depth_preprocess'], 'none')
        self.assertEqual(ControlNetRequest(type='depth').depth_preprocess, 'auto')
        self.assertEqual(depth_settings(DEPTH_DEFAULTS), DEPTH_DEFAULTS)
        for values in ({'depth_resolution':True}, {'depth_resolution':float('inf')},
                {'depth_resolution':141}, {'depth_resolution':0}, {'depth_resolution':2534},
                {'depth_dtype':'mixed'}, {'depth_resize_method':'crop'}, {'depth_normalization':'raw'}):
            with self.subTest(values=values), self.assertRaises(ValueError): depth_settings(values)
        for resolution in (0, 141, 2534):
            with self.assertRaises(ValidationError): ControlNetRequest(type='depth', depth_resolution=resolution)

    def test_supplied_map_is_returned_unchanged_without_loading_any_component(self):
        image = torch.rand(1, 23, 17, 3, device='cpu')
        self.assertIs(prepare_depth(object(), image, {}), image)
        self.assertIs(prepare_depth(object(), image, {'depth_preprocess':'none'}), image)

    def test_auto_model_uses_vram_capacity_and_keeps_cpu_and_manual_choices(self):
        memory=Mock()
        memory.get_torch_device.return_value=SimpleNamespace(type='cuda')
        with patch.dict('sys.modules',{'comfy':SimpleNamespace(model_management=memory)}):
            for size in (24,16,8,4,3):
                memory.get_total_memory.return_value=size * 1024**3
                expected='base' if size >= 8 else 'small'
                self.assertEqual(depth_model_name('auto'),f'depth_anything_3_{expected}.safetensors')
            memory.get_torch_device.return_value=SimpleNamespace(type='cpu')
            memory.get_total_memory.reset_mock()
            self.assertEqual(depth_model_name('auto'),'depth_anything_3_small.safetensors')
            memory.get_total_memory.assert_not_called()
            self.assertEqual(depth_model_name('base'),'depth_anything_3_base.safetensors')

    def test_native_nodes_receive_options_and_cache_follows_weights_and_precision(self):
        image = torch.rand(1, 23, 17, 3, device='cpu')
        load, infer = Mock(return_value=[object()]), Mock(return_value=['geometry'])
        render = Mock(return_value=[torch.zeros_like(image)])
        native = SimpleNamespace(LoadDA3Model=SimpleNamespace(execute=load), DA3Inference=SimpleNamespace(execute=infer),
                                 DA3Render=SimpleNamespace(execute=render))
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            path = Path(folder)/'depth_anything_3_small.safetensors'
            path.write_bytes(b'tiny mock weights')
            paths = SimpleNamespace(get_folder_file_path=lambda *args:path)
            owner = SimpleNamespace()
            with patch.dict('sys.modules', {'shared':SimpleNamespace(path_manager=paths),
                    'folder_paths':Mock(), 'comfy_extras.nodes_depth_anything_3':native}):
                options = dict(DEPTH_DEFAULTS, depth_model='small', depth_normalization='min_max', depth_resolution=280)
                prepare_depth(owner, image, options)
                prepare_depth(owner, image, options)
                self.assertEqual(load.call_count, 1)
                self.assertEqual(infer.call_args.kwargs['mode'], {'mode':'mono'})
                self.assertEqual(infer.call_args.kwargs['resolution'], 280)
                self.assertEqual(render.call_args.kwargs['output']['normalization'], 'min_max')
                before = path.stat()
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 10000000))
                prepare_depth(owner, image, options)
                self.assertEqual(load.call_count, 2)
                prepare_depth(owner, image, dict(options, depth_dtype='fp32'))
                self.assertEqual(load.call_count, 3)
                self.assertEqual(load.call_args.kwargs['weight_dtype'], 'fp32')
                render.return_value = [torch.zeros(1, 22, 17, 3, device='cpu')]
                with self.assertRaisesRegex(ValueError, 'dimensions'):
                    prepare_depth(owner, image, options)

    def test_ui_and_custom_presets_forward_depth_values_without_mutation(self):
        preset = dict(type='depth', strength=0.5, depth_preprocess='auto')
        get = function('modules/controlnet.py','get_settings', controlnet_settings={'Depth':preset}, NEWCN='Custom...')
        data = dict(cn_selection='Depth', cn_depth_preprocess='none', cn_depth_resolution=280, cn_depth_dtype='fp32')
        result = get(data)
        self.assertEqual((result['depth_preprocess'],result['depth_resolution'],result['depth_dtype']), ('none',280,'fp32'))
        self.assertEqual(preset, dict(type='depth', strength=0.5, depth_preprocess='auto'))
        custom = get(dict(data, cn_selection='Custom...', cn_type='Depth', cn_edge_low=.2, cn_edge_high=.8,
                          cn_start=0, cn_stop=1, cn_strength=.5, cn_upscale='None'))
        self.assertEqual(custom['depth_preprocess'], 'none')

    def test_only_unchanged_stock_depth_presets_opt_into_estimation(self):
        defaults = json.loads((ROOT/'settings/powerup.default').read_text())
        stock = {key:value for key,value in defaults['Depth (low)'].items() if key != 'depth_preprocess'}
        customized = dict(stock,strength=.73)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source, saved = folder/'default.json', folder/'saved.json'
            source.write_text(json.dumps(defaults))
            saved.write_text(json.dumps(defaults | {'Depth (low)':stock, 'Depth (high)':customized}))
            save = Mock()
            load = function('modules/controlnet.py', 'load_cnsettings', os=os, json=json,
                CNSETTINGS_FILE=str(saved), DEFAULT_CNSETTINGS_FILE=str(source), save_json=save,
                controlnet_models=dict.fromkeys(value['type'] for value in defaults.values()))
            result = load()
            self.assertEqual(result['Depth (low)']['depth_preprocess'], 'auto')
            self.assertEqual(result['Depth (high)'], customized)
            self.assertEqual(depth_settings(result['Depth (high)'])['depth_preprocess'], 'none')
            save.assert_called_once()

    def test_depth_ui_follows_model_preset_and_mode(self):
        class Widget:
            def __init__(self,*args,**kwargs): self.handlers=[]
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def change(self,fn,**kwargs): self.handlers.append(fn)
        gr = SimpleNamespace(**{name:Widget for name in ('Group','Accordion','Dropdown','Slider')},
            on=Mock(), update=lambda **kwargs:kwargs)
        cn = SimpleNamespace(NEWCN='Custom...',cn_options={'Depth':dict(type='depth',depth_preprocess='auto'),
                                                        'Old map':dict(type='depth')})
        registered={}
        build = function('modules/ui/ui_depth_control.py','create_ui', gr=gr,controlnet=cn,
            shared=SimpleNamespace(add_ctrl=lambda key,value:registered.update({key:value})),
            DEPTH_DEFAULTS=DEPTH_DEFAULTS, DEPTH_CHOICES=__import__('modules.depth_control',fromlist=['DEPTH_CHOICES']).DEPTH_CHOICES,
            depth_settings=depth_settings)
        fields=build(Widget(),Widget(),Widget(),lambda name:{'sdxl':name=='sdxl'})
        callback=gr.on.call_args.kwargs['fn']
        self.assertEqual(len(callback('Depth','Depth','sdxl')),len(gr.on.call_args.kwargs['outputs']))
        self.assertTrue(callback('Depth','Depth','sdxl')[0]['visible'])
        self.assertFalse(callback('Depth','Depth','mage')[0]['visible'])
        self.assertEqual(callback('Old map','Depth','sdxl')[1]['value'],'none')
        self.assertEqual(callback('Custom...','Depth','sdxl')[1]['value'],'auto')
        self.assertEqual(set(registered),{'cn_'+key for key in DEPTH_DEFAULTS})
        self.assertEqual(list(fields),list(DEPTH_DEFAULTS))
        self.assertEqual(fields['depth_preprocess'].handlers[0]('none'),{'visible':False})

    def test_custom_preset_save_keeps_all_depth_options(self):
        saved=Mock()
        save=function('modules/ui/ui_controlnet.py','cn_save',
            control_types=lambda features:['Depth'],checkpoint_features=lambda *args:{'sdxl':True},
            load_cnsettings=lambda:{},save_cnsettings=saved,powerup_choices=lambda **kwargs:[],
            gr=SimpleNamespace(update=lambda **kwargs:kwargs))
        values=dict(DEPTH_DEFAULTS,depth_resolution=280,depth_normalization='min_max',depth_dtype='fp32')
        save('Custom depth',None,'Depth',0,1,.5,.2,.8,'None','sdxl','Disabled',False,*values.values())
        recipe=saved.call_args.args[0]['Custom depth']
        self.assertEqual({key:recipe[key] for key in DEPTH_DEFAULTS},values)


if __name__ == '__main__': unittest.main()
