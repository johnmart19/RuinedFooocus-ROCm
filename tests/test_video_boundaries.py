"""Exercise timing boundaries without loading video models or a GPU."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
import torch

from modules.video_settings import VIDEO_FPS, frame_count

ROOT = Path(__file__).resolve().parents[1]


class VideoTimingTests(unittest.TestCase):
    def test_ltx_image_conditioning_preserves_both_native_outputs(self):
        tree = ast.parse((ROOT / "modules/ltx_video_pipeline.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "pipeline")
        process = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "process")
        branch = next(n for n in process.body if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "gen_data['input_image']")
        conditioning = next(n for n in process.body if isinstance(n, ast.Assign)
                            and isinstance(n.value, ast.Call)
                            and "LTXVConditioning" in ast.unparse(n.value))
        native = Mock()
        native.generate.return_value = ("native positive", "native negative", "native latent")
        apply = Mock()
        apply.execute.side_effect = lambda **values: (values["positive"], values["negative"])
        namespace = dict(np=np, torch=torch, fps=24,
                         LTXVImgToVideo=lambda: native, LTXVConditioning=lambda: apply,
                         gen_data={"input_image": [[[1, 2, 3]]], "width": 32, "height": 32, "frames": 9})
        function = ast.parse("def encode(self): pass").body[0]
        function.body = [branch, conditioning] + ast.parse("return positive, negative, latent_image").body
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "ltx_conditioning", "exec"), namespace)
        owner = SimpleNamespace(conditions={"+": {"cache": "text positive"}, "-": {"cache": "text negative"}},
                                model_base_patched=SimpleNamespace(vae="vae"))
        self.assertEqual(namespace["encode"](owner), ("native positive", "native negative", "native latent"))
        self.assertEqual(apply.execute.call_args.kwargs["negative"], "native negative")

    def test_silent_video_duration_and_save_defaults_agree(self):
        for filename, expected_fps in (("wan_video_pipeline.py", 16),
                                       ("hunyuan_video_pipeline.py", 24)):
            tree = ast.parse((ROOT / "modules" / filename).read_text())
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "pipeline")
            parse = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "parse_gen_data")
            process = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "process")
            save_fps = next(n for n in ast.walk(process) if isinstance(n, ast.Assign) and
                            any(isinstance(t, ast.Name) and t.id == "fps" for t in n.targets))
            namespace = dict(frame_count=frame_count)
            exec(compile(ast.Module(body=[parse], type_ignores=[]), filename, "exec"), namespace)
            data = namespace["parse_gen_data"](None, {"video_duration": 1})
            namespace["gen_data"] = data
            exec(compile(ast.Module(body=[save_fps], type_ignores=[]), filename, "exec"), namespace)
            self.assertEqual(namespace["fps"], expected_fps)
            self.assertEqual(data["original_image_number"], frame_count(1, namespace["fps"]))

    def test_legacy_ltx_requested_frames_survive_native_temporal_compression(self):
        tree = ast.parse((ROOT / "modules/ltx_video_pipeline.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "pipeline")
        process = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "process")
        timing = [n for n in process.body if isinstance(n, ast.Assign) and
                  (any(isinstance(t, ast.Name) and t.id == "fps" for t in n.targets) or
                   any(isinstance(t, ast.Subscript) and ast.unparse(t) == "gen_data['frames']" for t in n.targets))]
        self.assertEqual(len(timing), 2)
        for seconds in (0.5, 1, 2.3):
            data = {"original_image_number": seconds}
            namespace = dict(gen_data=data, settings=SimpleNamespace(default_settings={}),
                             VIDEO_FPS=VIDEO_FPS, frame_count=frame_count)
            exec(compile(ast.Module(body=timing, type_ignores=[]), "ltx_timing", "exec"), namespace)
            frames = data["frames"]
            self.assertEqual(namespace["fps"], 24)
            # EmptyLTXVLatentVideo and its VAE compress/expand time by eight.
            self.assertEqual(((frames - 1) // 8) * 8 + 1, frames)


if __name__ == "__main__":
    unittest.main()
