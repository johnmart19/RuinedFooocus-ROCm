"""Grouped styles and replay semantics without encoders, model workers or inference."""

import ast
from pathlib import Path
import random
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr

from ruinedfooocus.features.prompting import ui_styles
from ruinedfooocus.features.prompting.style_catalog import (
    STYLE_ALIASES,
    merge_selection,
    style_templates,
)
from ruinedfooocus.features.gallery.image_metadata import read_metadata, render_values

ROOT = Path(__file__).resolve().parents[1]


def compose():
    templates = {"Style: Frame": ("frame {prompt}", "noise"), "None": ("{prompt}", "")}
    scope = dict(re=re, random=random, STYLE_ALIASES=STYLE_ALIASES, load_styles=lambda: templates)
    for filename, names in (
        ("sdxl_styles.py", ("apply_style",)),
        ("prompt_processing.py", ("process_prompt", "use_random_prompt", "process_wildcards")),
    ):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/prompting" / filename).read_text(encoding="utf-8")
        )
        nodes = [
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), scope)
    return scope["process_prompt"], scope["apply_style"], templates


class StyleWorkflowTests(unittest.TestCase):
    def test_normal_ui_chat_and_api_requests_apply_selection_once_and_record_input(self):
        process, _, _ = compose()
        original = 'Exactly two red apples. "Київ" {prompt}'
        for task in (None, "tool_call", "external_api"):
            data = {"task_type": task}
            selected = ["Style: Frame", ">>>>>> PAPERCRAFT"]
            positive, negative = process(selected, original + " <style:Frame>", "No text.", data)
            self.assertEqual(positive.count("frame "), 1)
            self.assertIn(original, positive)
            self.assertEqual(negative, "noise, No text.")
            self.assertEqual(selected, ["Style: Frame", ">>>>>> PAPERCRAFT"])
            self.assertEqual(data["raw_prompt"], original + " <style:Frame>")
            self.assertEqual(data["style_provenance"]["selected"], ["Style: Frame"])
            self.assertEqual(
                data["style_provenance"]["templates"]["Style: Frame"], ("frame {prompt}", "noise")
            )

    def test_rendered_metadata_replays_literally_without_a_second_style(self):
        original = "Exactly two apples."
        data = {}
        process, _, _ = compose()
        positive, negative = process(["Style: Frame"], original, "No text.", data)
        meta = read_metadata({"Prompt": positive, "Negative": negative, **data})
        restored = render_values(meta)
        self.assertEqual(restored["raw_prompt"], original)
        self.assertEqual(
            process(["Style: Frame"], positive, negative, {"_metadata_prompt": True}),
            (positive, negative),
        )

    def test_filtering_and_toggling_a_group_keeps_hidden_styles_and_order(self):
        selected = ["Style: First", "Style: Hidden", "Local chat model"]
        result = merge_selection(
            selected, ["Style: First", "Style: New"], ["Style: First", "Style: New"]
        )
        self.assertEqual(result, selected + ["Style: New"])
        self.assertEqual(
            merge_selection(result, ["Style: First"], []), selected[1:] + ["Style: New"]
        )
        self.assertEqual(selected, ["Style: First", "Style: Hidden", "Local chat model"])

    def test_baking_consumes_visual_directives_once_but_does_not_run_neural_tools(self):
        process, apply, templates = compose()
        templates["Local chat model"] = ("{prompt}", "")
        with (
            patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.features.prompting.sdxl_styles": SimpleNamespace(
                        apply_style=apply
                    )
                },
            ),
            patch.object(ui_styles, "load_styles", return_value=templates),
        ):
            positive, negative = ui_styles.preview_styles(
                "Two apples. <style:Frame>", "", ["Style: Frame"], ""
            )
            self.assertNotIn("<style:Frame>", positive)
            self.assertEqual(positive.count("frame "), 1)
            self.assertEqual(process([], positive, negative, {})[0], positive)
            preview, _ = ui_styles.preview_styles(
                "Two apples. <style:Local chat model>", "", [], ""
            )
            self.assertIn("<style:Local chat model>", preview)

    def test_real_ui_groups_have_no_selectable_headers_and_legacy_defaults_stay_inactive(self):
        with gr.Blocks() as app:
            controls = ui_styles.create_ui({"style": ["Style: sai-cinematic"]})
        self.assertEqual(controls.selection.value, [])
        groups = [c for c in app.blocks.values() if isinstance(c, gr.CheckboxGroup)]
        self.assertGreaterEqual(len(groups), 9)
        for group in groups:
            self.assertFalse(any(str(value).startswith(">>>") for _, value in group.choices))
        self.assertFalse(controls.selection.visible)

    def test_new_defaults_restore_real_styles_and_processor_controls(self):
        with gr.Blocks():
            controls = ui_styles.create_ui(
                {
                    "style": [">>>>>> PHOTO", "Style: sai-anime", "Local chat model"],
                    "style_application": 1,
                }
            )
            ui_styles.add_processor_controls(controls.selection)
        self.assertEqual(controls.selection.value, ["Style: sai-anime", "Local chat model"])

    def test_bake_clears_applied_styles_and_undo_protects_new_edits(self):
        with gr.Blocks() as app:
            prompt, negative, words = gr.Textbox(), gr.Textbox(), gr.Textbox()
            controls = ui_styles.create_ui({})
            ui_styles.bind_actions(controls, prompt, negative, words)
        functions = {fn.fn.__name__: fn.fn for fn in app.fns.values()}
        with patch.object(
            ui_styles, "preview_styles", return_value=("styled original", "styled negative")
        ):
            result = functions["bake"](
                "original", "negative", ["Style: sai-anime", "Local chat model"], ""
            )
        self.assertEqual(result[:3], ("styled original", "styled negative", ["Local chat model"]))
        with self.assertRaisesRegex(gr.Error, "overwrite your edits"):
            functions["undo"]("new edit", result[1], result[2], result[3])
        self.assertEqual(
            functions["undo"](result[0], result[1], result[2], result[3])[:3],
            ("original", "negative", ["Style: sai-anime", "Local chat model"]),
        )


if __name__ == "__main__":
    unittest.main()
