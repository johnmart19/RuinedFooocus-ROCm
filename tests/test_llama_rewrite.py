"""Prompt-rewrite presets and their output handling, without loading a model."""
import ast
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


def helpers():
    tree = ast.parse((ROOT / "modules/llama_pipeline.py").read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("clean_rewrite", "restore_tags")]
    scope = {"re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "llama_pipeline", "exec"), scope)
    return scope["clean_rewrite"], scope["restore_tags"]


class RewriteOutputTests(unittest.TestCase):
    def test_echoed_labels_and_wrapping_quotes_are_removed(self):
        clean, _ = helpers()
        self.assertEqual(clean('Output: "a fox in the snow"'), "a fox in the snow")
        self.assertEqual(clean("  'a fox'  "), "a fox")
        self.assertEqual(clean("a \"quoted\" word stays"), 'a "quoted" word stays')
        self.assertEqual(clean(None), "")

    def test_dropped_lora_and_style_tags_are_restored_once(self):
        _, restore = helpers()
        original = "a fox <lora:detail:0.5> in snow <style:Cinematic> <lora:detail:0.5>"
        self.assertEqual(restore(original, "Solitude"), "Solitude <lora:detail:0.5> <style:Cinematic>")
        kept = "a red fox, dusk <lora:detail:0.5> <style:Cinematic>"
        self.assertEqual(restore(original, kept), kept)
        self.assertEqual(restore("no tags here", "rewritten,"), "rewritten,")


class PresetFileTests(unittest.TestCase):
    def test_presets_have_a_name_rules_and_fit_the_context(self):
        presets = sorted((ROOT / "llamas").glob("*.txt"))
        self.assertTrue(presets)
        names = set()
        for path in presets:
            with self.subTest(preset=path.name):
                name, _, body = path.read_text(encoding="utf-8").partition("\n")
                self.assertTrue(name.strip())
                self.assertNotIn(name.strip(), names)
                names.add(name.strip())
                self.assertIn("Reply with", body)
                # The system prompt shares the context with the user's prompt and reply.
                self.assertLess(len(body.split()), 450)


if __name__ == "__main__":
    unittest.main()
