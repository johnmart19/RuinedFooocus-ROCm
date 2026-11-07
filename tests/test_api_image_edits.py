"""API image edits pick the checkpoint's native edit workflow when no tool is named."""

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from ruinedfooocus.features.models.image_components import native_edit_supported


def queued_control(family, name="local.safetensors", controlnet=None, moss_format="Disabled"):
    import ruinedfooocus.app.api_runtime as api_runtime

    perf = SimpleNamespace(
        CUSTOM_PERFORMANCE="Own settings",
        choices_for_model=lambda *args: ["Own settings"],
        selection_for_model=lambda *args: "Own settings",
        get_perf_options=lambda *args: dict(
            custom_steps=8, cfg=4, sampler_name="euler", scheduler="simple", clip_skip=1
        ),
    )
    shared = SimpleNamespace(
        settings=SimpleNamespace(
            default_settings={"performance": "Own settings", "moss_image_edit_format": moss_format}
        ),
        performance_settings=perf,
    )
    samplers = SimpleNamespace(KSampler=SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["simple"]))
    defaults = dict(
        base_model_name=name,
        performance_selection="Own settings",
        moss_image_edit_format=moss_format,
    )
    payload = dict(prompt="make it blue", size="512x512", input_image="upload")
    if controlnet:
        payload["controlnet"] = controlnet
    with (
        patch.dict(sys.modules, {"ruinedfooocus.app.state": shared, "comfy.samplers": samplers}),
        patch.object(api_runtime, "image_defaults", defaults),
        patch.object(api_runtime, "decode_image", return_value=Image.new("RGB", (64, 64))),
        patch.object(api_runtime, "submit", side_effect=lambda run: run),
    ):
        operation = api_runtime.images(payload, dict(family=family, name=name))
    cells = dict(
        zip(operation.__code__.co_freevars, (cell.cell_contents for cell in operation.__closure__))
    )
    return cells["data"]["controlnet"]


class ApiImageEditTests(unittest.TestCase):
    def test_edit_capable_checkpoints_use_native_editing_by_default(self):
        self.assertEqual(queued_control("Qwen"), {"type": "edit"})
        self.assertEqual(queued_control("MageFlow"), {"type": "edit"})
        self.assertEqual(
            queued_control("Flux.1 D", name="flux1-kontext-dev.safetensors"), {"type": "edit"}
        )

    def test_other_checkpoints_keep_gentle_image_to_image(self):
        self.assertEqual(queued_control("SDXL 1.0"), {"type": "img2img", "denoise": 0.25})
        self.assertEqual(queued_control("Flux.1 D"), {"type": "img2img", "denoise": 0.25})
        self.assertEqual(queued_control("MOSSImage"), {"type": "img2img", "denoise": 0.25})

    def test_an_explicit_tool_is_never_replaced(self):
        control = {"type": "img2img", "denoise": 0.6}
        self.assertEqual(queued_control("Qwen", controlnet=control), control)

    def test_ui_and_api_share_one_capability_rule(self):
        self.assertTrue(native_edit_supported("MOSS Image", moss_format="ai-toolkit"))
        self.assertFalse(native_edit_supported("MOSS Image"))
        self.assertFalse(native_edit_supported(None))


if __name__ == "__main__":
    unittest.main()
