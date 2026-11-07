from ruinedfooocus.core.paths import asset_path, resource_path

"""Prompt expansion budgets/whitelists and UTF-8 styles without loading models."""
import ast
import csv
from collections import UserDict
from pathlib import Path
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, mock_open, patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def function(path, name, scope):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), scope)
    return scope[name]


class ExpansionTests(unittest.TestCase):
    def expand(self, length):
        vocab = {"Ġred": 0, "Ġblue": 1, "other": 2}
        tokenizer = Mock(
            side_effect=lambda *a, **k: UserDict(
                {
                    "input_ids": torch.zeros(1, length, dtype=torch.int64),
                    "attention_mask": torch.ones(1, length, dtype=torch.int64),
                }
            )
        )
        tokenizer.get_vocab.return_value = vocab
        tokenizer.batch_decode.return_value = ["expanded prompt"]
        model = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=1024),
            generate=Mock(return_value=torch.arange(length + 2).reshape(1, -1)),
        )
        owner = SimpleNamespace(
            tokenizer=tokenizer,
            model=model,
            patcher=SimpleNamespace(load_device="cpu"),
            logits_processor=object(),
        )
        execute = function(
            "ruinedfooocus/features/prompting/prompt_expansion.py",
            "__call__",
            {
                "torch": torch,
                "os": os,
                "SEED_LIMIT_NUMPY": 2**32,
                "set_seed": Mock(),
                "safe_str": lambda text: text.strip(),
                "fooocus_expansion_path": "unused",
                "neg_inf": -8192.0,
                "LogitsProcessorList": lambda processors: processors,
            },
        )
        source = mock_open(read_data="red\nblue\nred\n")
        original = " , Original words remain unchanged.  "
        with patch("builtins.open", source):
            result = execute(owner, original, 123)
        return owner, source, result, original

    def test_full_blocks_receive_positive_budget_within_context(self):
        for length, budget in ((1, 74), (74, 1), (75, 75), (150, 75), (1000, 24), (1023, 1)):
            with self.subTest(length=length):
                owner, source, result, original = self.expand(length)
                self.assertEqual(owner.model.generate.call_args.kwargs["max_new_tokens"], budget)
                self.assertEqual(owner.logits_bias.tolist(), [[0, 0, -8192]])
                self.assertEqual(result, original + ", expanded prompt")
                decoded = owner.tokenizer.batch_decode.call_args.args[0]
                self.assertEqual(decoded.tolist(), [[length, length + 1]])

    def test_no_remaining_context_returns_original_without_generation_or_word_loading(self):
        for length in (1024, 1025, 4096):
            owner, source, result, original = self.expand(length)
            owner.model.generate.assert_not_called()
            source.assert_not_called()
            self.assertEqual(result, original)

    def test_actual_tokenizer_keeps_literal_special_tokens_in_the_original_prefix(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(ROOT / "prompt_expansion", local_files_only=True)
        suffix_ids = tokenizer.encode(" beautiful light", add_special_tokens=False)
        model = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=1024),
            generate=lambda input_ids, **kwargs: torch.cat(
                (input_ids, torch.tensor([suffix_ids])), dim=1
            ),
        )
        owner = SimpleNamespace(
            tokenizer=tokenizer,
            model=model,
            patcher=SimpleNamespace(load_device="cpu"),
            logits_processor=object(),
        )
        execute = function(
            "ruinedfooocus/features/prompting/prompt_expansion.py",
            "__call__",
            {
                "torch": torch,
                "os": os,
                "SEED_LIMIT_NUMPY": 2**32,
                "set_seed": Mock(),
                "safe_str": lambda text: text.strip(" ,."),
                "fooocus_expansion_path": "unused",
                "neg_inf": -8192.0,
                "LogitsProcessorList": lambda processors: processors,
            },
        )
        original = " , Київ <|endoftext|> red cup  , "
        with patch("builtins.open", mock_open(read_data="beautiful\nlight\n")):
            result = execute(owner, original, 123)
        self.assertEqual(result, original + ", beautiful light")

    def test_style_wrapper_forwards_original_prefix_without_normalization(self):
        original = " , Київ <|endoftext|>  red cup.  "
        expansion = Mock(return_value=original + ", beautiful light")
        execute = function(
            "ruinedfooocus/features/prompting/prompt_expansion.py",
            "expand_prompt",
            {
                "FooocusExpansion": lambda: expansion,
                "remove_empty_str": lambda items, default: (
                    [item for item in items if item != ""] or [default]
                ),
                "random": SimpleNamespace(randint=lambda *args: 123),
            },
        )
        self.assertEqual(execute(original), original + ", beautiful light")
        expansion.assert_called_once_with(original, 123)


class StyleEncodingTests(unittest.TestCase):
    def test_utf8_styles_and_bom_preserve_words_across_load_merge_and_reload(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            default = Path(folder) / "default.csv"
            custom = Path(folder) / "styles.csv"
            default.write_text(
                "name,prompt,negative_prompt\nКиїв,猫 біля чашки,без шуму\n", encoding="utf-8-sig"
            )
            custom.write_text(
                "name,prompt,negative_prompt\nCustom,світло café,текст\n", encoding="utf-8-sig"
            )
            execute = function(
                "ruinedfooocus/features/prompting/sdxl_styles.py",
                "load_styles",
                {
                    "DEFAULT_STYLES_FILE": default,
                    "STYLES_FILE": custom,
                    "exists": os.path.exists,
                    "csv": csv,
                    "DictReader": csv.DictReader,
                    "locale": __import__("locale"),
                    "STYLE_ALIASES": {},
                },
            )
            styles = execute()
            self.assertEqual(styles["Київ"], ("猫 біля чашки", "без шуму"))
            self.assertEqual(styles["Custom"], ("світло café", "текст"))
            self.assertEqual(execute(), styles)
            self.assertIn("світло café", custom.read_text(encoding="utf-8"))

    def test_legacy_locale_styles_are_read_losslessly_without_incidental_rewrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            default = Path(folder) / "default.csv"
            custom = Path(folder) / "styles.csv"
            text = "name,prompt,negative_prompt\nCustom,світло,текст\n"
            default.write_text(text, encoding="utf-8")
            original = text.encode("cp1251")
            custom.write_bytes(original)
            execute = function(
                "ruinedfooocus/features/prompting/sdxl_styles.py",
                "load_styles",
                {
                    "DEFAULT_STYLES_FILE": default,
                    "STYLES_FILE": custom,
                    "exists": os.path.exists,
                    "csv": csv,
                    "DictReader": csv.DictReader,
                    "locale": SimpleNamespace(getpreferredencoding=lambda flag: "cp1251"),
                    "STYLE_ALIASES": {},
                },
            )
            with patch(
                "ruinedfooocus.features.prompting.style_catalog.locale.getpreferredencoding",
                return_value="cp1251",
            ):
                self.assertEqual(execute()["Custom"], ("світло", "текст"))
            self.assertEqual(custom.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
