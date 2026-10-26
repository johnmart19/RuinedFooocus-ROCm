"""Execute the production I2V branch with native-node mocks, without a GPU."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class HunyuanImageConditioningTests(unittest.TestCase):
    def test_hunyuan_vision_override_lists_vision_files_in_settings(self):
        tree = ast.parse((ROOT / 'modules/ui/ui_settings.py').read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'resource_choices')
        manager = SimpleNamespace(get_folder_list=Mock(return_value=['llava_llama3_vision.safetensors']))
        namespace = dict(RESOURCE_DEFAULTS={'clip_hunyuan_vision': 'llava_llama3_vision.safetensors'}, path_manager=manager)
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'settings_choices', 'exec'), namespace)
        choices = namespace['resource_choices']('clip_hunyuan_vision', all_files=True)
        manager.get_folder_list.assert_called_once_with('clip_vision')
        self.assertIn('llava_llama3_vision.safetensors', choices)

    def test_vision_artifacts_preserve_tensors_without_mutating_native_conditioning(self):
        from tools.comfy_video_reference import _conditioning_artifact
        class VisionOutput:
            pass
        vision = VisionOutput()
        vision.image_embeds = torch.ones((1, 2))
        source = {'positive': [[torch.zeros((1, 2)), {'clip_vision_output': vision}]]}
        with patch.dict('sys.modules', {'comfy.clip_vision': SimpleNamespace(Output=VisionOutput)}):
            artifact = _conditioning_artifact(source)
        self.assertIs(source['positive'][0][1]['clip_vision_output'], vision)
        self.assertIs(artifact['positive'][0][1]['clip_vision_output']['image_embeds'], vision.image_embeds)

    def branch(self):
        tree = ast.parse((ROOT / 'modules/hunyuan_video_pipeline.py').read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'pipeline')
        process = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'process')
        branch = next(node for node in process.body if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "gen_data['input_image']")
        return compile(ast.Module(body=[branch], type_ignores=[]), 'production_i2v', 'exec')

    def test_image_prompt_reaches_latent_conditioning_and_refreshes_for_each_image(self):
        owner = SimpleNamespace(model_base_patched=SimpleNamespace(clip='text encoder', vae='vae', clip_vision=None),
                                conditions={'+': {'cache': 'stale text-only conditioning'}})
        loader = Mock(return_value='vision encoder')
        encode = Mock()
        encode.encode.return_value = ('image embedding',)
        text = Mock()
        text.execute.return_value = ('image-aware prompt',)
        latent = Mock()
        latent.encode.return_value = ('conditioned positive', 'video latent')
        namespace = dict(self=owner, gen_data={'input_image': Image.new('RGB', (32, 32)),
            'width': 32, 'height': 32, 'original_image_number': 25}, np=np, torch=torch,
            comfy=SimpleNamespace(clip_vision=SimpleNamespace(load=loader)),
            resource_name=Mock(return_value='llava_llama3_vision.safetensors'),
            settings=SimpleNamespace(default_settings={}),
            path_manager=SimpleNamespace(get_folder_file_path=Mock(return_value='vision.safetensors')),
            CLIPVisionEncode=lambda: encode, TextEncodeHunyuanVideo_ImageToVideo=text,
            HunyuanImageToVideo=lambda: latent, positive_prompt='moving boat')
        code = self.branch()
        exec(code, namespace)
        loader.assert_called_once_with('vision.safetensors')
        self.assertEqual(encode.encode.call_args.kwargs['crop'], 'none')
        text.execute.assert_called_once_with('text encoder', 'image embedding', 'moving boat', image_interleave=4)
        self.assertEqual(latent.encode.call_args.kwargs['positive'], 'image-aware prompt')
        self.assertEqual(latent.encode.call_args.kwargs['guidance_type'], 'v2 (replace)')
        namespace['positive_prompt'] = 'different scene'
        namespace['gen_data']['input_image'] = Image.new('RGB', (32, 32), 'red')
        exec(code, namespace)
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(encode.encode.call_count, 2)
        self.assertEqual(text.execute.call_args.args[2], 'different scene')


if __name__ == '__main__':
    unittest.main()
