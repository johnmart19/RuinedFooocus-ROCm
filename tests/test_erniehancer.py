"""Ernie enhancement uses native generation/device policy and a bounded model cache."""
import ast
import json
from math import prod
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

ROOT=Path(__file__).resolve().parents[1]


class ErniehancerTests(unittest.TestCase):
    def setUp(self):
        self.flags=SimpleNamespace(fp16_text_enc=False,bf16_text_enc=False,
                                   fp8_e4m3fn_text_enc=False,fp8_e5m2_text_enc=False)
        self.management=SimpleNamespace(args=self.flags,
            text_encoder_device=Mock(return_value=torch.device('cpu')),
            text_encoder_offload_device=Mock(return_value=torch.device('cpu')),
            text_encoder_dtype=Mock(return_value=torch.float16),
            get_free_memory=Mock(return_value=128*1024**3))
        self.header={'model.norm.weight':{'dtype':'BF16','shape':[2]},'tekken_model':{'dtype':'U8','shape':[2]}}
        self.weight=torch.tensor([1.,2.],dtype=torch.bfloat16,device='cpu')
        self.tokenizer=torch.tensor([3,4],dtype=torch.uint8,device='cpu')
        self.state={'model.norm.weight':self.weight,'tekken_model':self.tokenizer}
        self.native=Mock(return_value=object())
        self.load=Mock(return_value=object())
        self.reader=Mock(return_value=self.state)
        self.node=SimpleNamespace(execute=Mock(return_value=('enhanced','reason')))
        self.settings={}
        self.signature=Mock(side_effect=lambda path:[path,10,100,200])
        self.scope=dict(torch=torch,prod=prod,os=os,json=json,CLIPType=SimpleNamespace(FLUX2='flux2'),
            model_management=self.management,resource_name=lambda key,values:values.get(key) or 'encoder.safetensors',
            settings=SimpleNamespace(default_settings=self.settings),
            path_manager=SimpleNamespace(model_paths={'clip_path':'models/clip'},get_folder_file_path=lambda *a,**k:k['default']),
            file_signature=self.signature,read_safetensors_header=Mock(side_effect=lambda path:self.header),
            load_clip=self.load,TextGenerate=self.node,random=SimpleNamespace(randint=lambda *a:123))
        tree=ast.parse((ROOT/'modules/prompt_expansion.py').read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Erniehancer')
        exec(compile(ast.Module(body=[node],type_ignores=[]),'erniehancer','exec'),self.scope)
        self.cls=self.scope['Erniehancer']
        utils=ModuleType('comfy.utils');utils.load_torch_file=self.reader
        sd=ModuleType('comfy.sd');sd.load_text_encoder_state_dicts=self.native
        comfy=ModuleType('comfy');comfy.utils=utils;comfy.sd=sd
        self.modules=patch.dict('sys.modules',{'comfy':comfy,'comfy.utils':utils,'comfy.sd':sd})
        self.modules.start();self.addCleanup(self.modules.stop)

    def test_unquantized_cpu_weights_promote_before_native_loader_without_mutating_input(self):
        owner=self.cls()
        self.assertIs(owner.clip,self.native.return_value)
        states=self.native.call_args.args[0]
        self.assertEqual(states[0]['model.norm.weight'].dtype,torch.float32)
        self.assertTrue(torch.equal(states[0]['model.norm.weight'],self.weight.float()))
        self.assertEqual(self.weight.dtype,torch.bfloat16)
        self.assertIs(states[0]['tekken_model'],self.tokenizer)
        self.assertTrue(self.native.call_args.kwargs['disable_dynamic'])
        self.assertEqual(self.native.call_args.kwargs['model_options']['offload_device'],torch.device('cpu'))

    def test_repeated_constructor_reuses_only_same_file_and_loader_policy(self):
        first=self.cls()
        self.assertIs(self.cls().clip,first.clip)
        self.native.assert_called_once()
        self.signature.side_effect=lambda path:[path,10,101,200]
        self.cls()
        self.assertEqual(self.native.call_count,2)
        self.flags.fp16_text_enc=True  # Same dtype getter result as Auto, different storage policy.
        self.cls()
        self.load.assert_called_once()

    def test_cuda_and_directml_keep_native_load_and_offload_devices(self):
        for label in ('cuda:0','privateuseone:0'):
            with self.subTest(device=label):
                self.management.text_encoder_device.return_value=torch.device(label)
                owner=self.cls()
                self.assertIs(owner.clip,self.load.return_value)
                options=self.load.call_args.kwargs['model_options']
                self.assertEqual(options,{'load_device':torch.device(label),'offload_device':torch.device('cpu')})
        self.reader.assert_not_called()
        self.scope['read_safetensors_header'].assert_not_called()

    def test_quantized_metadata_integer_weights_low_memory_and_explicit_precision_fall_back(self):
        for change in ('metadata','integer','memory','precision'):
            with self.subTest(change=change):
                self.cls._cache=None
                original=dict(self.header)
                self.management.get_free_memory.return_value=128*1024**3
                self.flags.bf16_text_enc=False
                if change=='metadata':self.header['__metadata__']={'format':'quantized'}
                if change=='integer':self.header['qweight']={'dtype':'I32','shape':[2,2]}
                if change=='memory':self.management.get_free_memory.return_value=1
                if change=='precision':self.flags.bf16_text_enc=True
                self.cls()
                self.header.clear();self.header.update(original)
        self.assertEqual(self.load.call_count,4)
        self.native.assert_not_called()

    def test_failed_load_is_not_cached(self):
        self.native.side_effect=RuntimeError('load failed')
        with self.assertRaisesRegex(RuntimeError,'load failed'):self.cls()
        self.assertIsNone(self.cls._cache)

    def test_native_generation_and_empty_answer_fallback_keep_original(self):
        owner=self.cls()
        prompt=' , Keep Київ and two red cups. '
        self.assertEqual(owner.execute(prompt),'enhanced')
        kwargs=self.node.execute.call_args.kwargs
        self.assertIs(kwargs['clip'],owner.clip)
        request=kwargs['prompt']
        self.assertTrue(request.startswith('[SYSTEM_PROMPT]'))
        self.assertTrue(request.endswith('[/INST]'))
        self.assertEqual(json.loads(request.split('[INST]',1)[1][:-len('[/INST]')]),{'prompt':prompt})
        self.assertEqual(kwargs['max_length'],512)
        self.assertEqual(kwargs['sampling_mode']['seed'],123)
        self.assertEqual(kwargs['sampling_mode']['repetition_penalty'],1.05)
        self.node.execute.return_value=('','incomplete reasoning')
        self.assertEqual(owner.execute(prompt),prompt)

    def test_request_encodes_quotes_newlines_backslashes_and_unicode_losslessly(self):
        prompt='Two signs: "Київ" and "東京".\nPath C:\\photos. Keep {braces}.'
        self.cls().execute(prompt)
        request=self.node.execute.call_args.kwargs['prompt']
        self.assertEqual(json.loads(request.split('[INST]',1)[1][:-len('[/INST]')])['prompt'],prompt)

    def test_pinned_native_node_separates_reasoning_before_returning_prompt(self):
        tree=ast.parse((ROOT/'repositories/ComfyUI/comfy_extras/nodes_textgen.py').read_text(encoding='utf-8'))
        owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TextGenerate')
        method=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
        method.decorator_list=[]
        scope={'io':SimpleNamespace(NodeOutput=lambda *values:values)}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'native_textgen','exec'),scope)
        for output,expected in (('<think>reason</think>two red cups','two red cups'),
                                ('<think>unfinished','original'),('two red cups','two red cups')):
            clip=SimpleNamespace(tokenize=lambda *a,**k:{'text':[[(1,1.)]]},
                generate=lambda *a,**k:[999],decode=lambda ids:output if ids==[999] else '')
            self.node.execute=lambda **kwargs:scope['execute'](None,**kwargs)
            enhancer=self.cls.__new__(self.cls);enhancer.clip=clip
            self.assertEqual(enhancer.execute('original'),expected)


if __name__=='__main__':unittest.main()
