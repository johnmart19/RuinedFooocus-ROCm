"""Exercise native image-edit routing without loading diffusion weights."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image
import torch
from modules.settings_validation import validate_settings, MOSS_EDIT_FORMATS
from modules.image_guidance import reference_image_map


class ImageEditTests(unittest.TestCase):
    def test_moss_edit_settings_match_native_formats_and_strength_bounds(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "repositories/ComfyUI/comfy_extras/nodes_moss_image.py").read_text())
        formats = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "EDIT_FORMATS" for t in n.targets))
        self.assertEqual(set(MOSS_EDIT_FORMATS) - {"Disabled"}, set(ast.literal_eval(formats)))
        for value in (0, 1.0, 100):
            self.assertEqual(validate_settings({"moss_image_edit_strength": value})["moss_image_edit_strength"], value)
        for value in (-1, 101, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                validate_settings({"moss_image_edit_strength": value})
        with self.assertRaises(ValueError):
            validate_settings({"moss_image_edit_format": "unknown"})

    def encode(self, architecture, moss_format="Disabled", references=None):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "modules/image_pipeline.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "pipeline")
        process = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "process")
        branch = next(n for n in process.body if isinstance(n, ast.If) and ast.unparse(n.test) == "image_edit and input_images > 0")
        function = ast.parse("def encode(self): pass").body[0]
        function.body = ast.parse("input_images = 1; edit_latent = None").body + branch.body + ast.parse("return edit_latent, img2img_mode, input_images").body
        native_latent = {"samples": torch.zeros(1, 128, 24, 32)}
        mage = Mock()
        mage.execute.return_value = SimpleNamespace(result=("positive", "negative", native_latent))
        qwen = Mock()
        qwen.execute.side_effect = [("positive",), ("negative",)]
        moss = Mock()
        moss.execute.side_effect = [("positive",), ("negative",)]
        model = type(architecture, (), {})()
        owner = SimpleNamespace(xl_base_patched=SimpleNamespace(unet=SimpleNamespace(model=model), clip="clip", vae="vae"),
                                xl_base=SimpleNamespace(vae="vae"))
        image = Image.new("RGBA", (64, 48))
        namespace = dict(callback=None, clean_prompt_cond_caches=lambda: {"+": {}, "-": {}},
                         TextEncodeMageFlowEdit=lambda: mage, TextEncodeFibo2=lambda: mage,
                         TextEncodeQwenImageEdit=lambda: qwen,
                         TextEncodeMOSSImageEdit=lambda: moss, np=np, torch=torch,
                         positive_prompt="edit", negative_prompt="avoid", moss_edit_format=moss_format, moss_edit_strength=1.5,
                         input_image=torch.zeros(1, 48, 64, 3), edit_latent=None, img2img_mode=False,
                         input_images=1, gen_data={"input_image": image, "width": 512, "height": 384})
        namespace.update(additional_references=references or [], reference_image_map=reference_image_map,
                         reference_tensor=lambda image, *size: torch.zeros(1, image.height, image.width, 3))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "edit_route", "exec"), namespace)
        result = namespace["encode"](owner)
        return result, mage, qwen, moss, native_latent

    def test_mage_keeps_native_target_and_encodes_rgb_at_requested_size(self):
        result, mage, _, _, latent = self.encode("MageFlow")
        self.assertIs(result[0], latent)
        self.assertEqual(result[1:], (False, 0))
        args = mage.execute.call_args.kwargs
        self.assertEqual((args["width"], args["height"]), (512, 384))
        self.assertEqual(args["images"]["image_1"].shape[-1], 3)

    def test_fibo_keeps_native_target_and_forwards_all_edit_references(self):
        reference = Image.new('RGB', (32, 24))
        result, fibo, _, _, latent = self.encode('Fibo2', references=[reference])
        self.assertIs(result[0], latent)
        self.assertEqual(result[1:], (False, 0))
        args = fibo.execute.call_args.kwargs
        self.assertEqual((args['width'], args['height']), (512, 384))
        self.assertEqual(args['prompt'], 'edit')
        self.assertEqual(args['negative_prompt'], 'avoid')
        self.assertEqual(args['vae'], 'vae')
        self.assertEqual(list(args['images']), ['image_1', 'image_2'])
        self.assertEqual(args['images']['image_1'].shape, (1, 48, 64, 3))
        self.assertEqual(args['images']['image_2'].shape, (1, 24, 32, 3))
        native = ast.parse((Path(__file__).resolve().parents[1] /
            'repositories/ComfyUI/comfy_extras/nodes_fibo2.py').read_text(encoding='utf-8'))
        execute = next(node for node in ast.walk(native) if isinstance(node, ast.FunctionDef) and node.name == 'execute')
        self.assertTrue(set(args).issubset({arg.arg for arg in execute.args.args}))

    def test_edit_capabilities_use_upstream_options_for_all_native_families(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'modules/image_pipeline.py').read_text(encoding='utf-8'))
        info = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'known_model_info' for target in node.targets))
        for key, value in zip(info.keys, info.values):
            if key.value in ('Fibo2', 'MageFlow', 'QwenImage'):
                fields = dict(zip((field.value for field in value.keys), value.values))
                self.assertEqual(ast.literal_eval(fields['options']), {'has_image_edit': True})
                self.assertNotIn('flags', fields)

    def test_native_edit_does_not_validate_hidden_ordinary_denoise(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'modules/image_pipeline.py').read_text(encoding='utf-8'))
        process = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'process')
        validation = next(node for node in process.body if isinstance(node, ast.If)
                          and 'control_type in modules.controlnet.controlnet_models' in ast.unparse(node.test))
        scope = dict(control_type='img2img', input_image=object(), controlnet={'strength': 2},
                     modules=SimpleNamespace(controlnet=SimpleNamespace(controlnet_models={'img2img': None})))
        code = compile(ast.Module(body=[validation], type_ignores=[]), 'edit_preflight', 'exec')
        exec(code, scope | {'image_edit': True})
        with self.assertRaisesRegex(ValueError, 'denoise'):
            exec(code, scope | {'image_edit': False})

    def test_qwen_reference_does_not_enable_img2img_or_loopback(self):
        result, _, qwen, _, _ = self.encode("QwenImage")
        self.assertEqual(result, (None, False, 0))
        self.assertEqual(qwen.execute.call_count, 2)

    def test_native_mage_and_moss_receive_additional_images_without_sampling(self):
        reference = Image.new('RGB', (32, 24))
        _, mage, _, moss, _ = self.encode('MageFlow', references=[reference, reference])
        self.assertEqual(list(mage.execute.call_args.kwargs['images']), ['image_1', 'image_2', 'image_3'])
        _, _, _, moss, _ = self.encode('MOSSImage', 'ai-toolkit', references=[reference])
        for call in moss.execute.call_args_list:
            self.assertEqual(list(call.kwargs['images']), ['image_1', 'image_2'])
            self.assertEqual(call.kwargs['images']['image_2'].shape, (1, 24, 32, 3))

    def test_moss_passes_explicit_native_format_to_both_prompts(self):
        for edit_format in ("ai-toolkit", "ai-toolkit (kv_cache off)", "Moss-Image-Edit"):
            result, _, _, moss, _ = self.encode("MOSSImage", edit_format)
            self.assertEqual(result, (None, False, 0))
            self.assertEqual(moss.execute.call_count, 2)
            for call in moss.execute.call_args_list:
                self.assertEqual(call.kwargs["edit_format"], edit_format)
                self.assertEqual(call.kwargs["image_strength"], 1.5)
                self.assertEqual(call.kwargs["vae"], "vae")

    def test_mage_target_survives_latent_initialization(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "modules/image_pipeline.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "pipeline")
        process = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "process")
        branch = next(n for n in process.body if isinstance(n, ast.If) and ast.unparse(n.test) == "edit_latent is not None")
        function = ast.parse("def initialize(edit_latent): pass").body[0]
        function.body = [branch] + ast.parse("return latent, denoise").body
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "target_latent", "exec"), namespace)
        latent = {"samples": torch.zeros(1, 128, 24, 32)}
        result, denoise = namespace["initialize"](latent)
        self.assertIs(result, latent)
        self.assertIsNone(denoise)


if __name__ == "__main__":
    unittest.main()
