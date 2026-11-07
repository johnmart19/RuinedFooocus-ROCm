from ruinedfooocus.core.paths import asset_path, resource_path

"""Workflow validation without starting model workers or downloading files."""
import ast
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, mock_open, patch

ROOT = Path(__file__).resolve().parents[1]


def function_from(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    )
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


class WorkflowControlsTests(unittest.TestCase):
    def test_generation_initial_update_works_after_changelog_setup(self):
        from ruinedfooocus.ui import hints

        namespace = dict(
            gr=SimpleNamespace(update=lambda **values: values),
            html=SimpleNamespace(make_progress_html=lambda *_: "Waiting"),
            asset_path=asset_path,
            hints=hints,
            ui_hints=hints,
        )
        for name in (
            "run_button",
            "stop_button",
            "progress_html",
            "gallery",
            "main_view",
            "inpaint_view",
            "hint_text",
        ):
            namespace[name] = name
        source = ROOT / "ruinedfooocus/ui/app.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        changelog = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Try)
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "open"
                and child.args
                and isinstance(child.args[0], ast.Constant)
                and child.args[0].value == "update_log.md"
                for statement in node.body
                for child in ast.walk(statement)
            )
        )
        with patch("builtins.open", mock_open(read_data="Recent changes")):
            exec(
                compile(ast.Module(body=[changelog], type_ignores=[]), str(source), "exec"),
                namespace,
            )
        update = function_from("ruinedfooocus/ui/app.py", "update_clicked", namespace)()
        self.assertTrue(update["hint_text"]["value"].startswith("**LPT:**"))
        self.assertFalse(update["run_button"]["interactive"])
        self.assertTrue(update["stop_button"]["interactive"])

    def test_rewrite_preview_requires_apply_and_does_not_overwrite_new_edits(self):
        gr = SimpleNamespace(Error=ValueError, update=lambda **values: values)
        model = Mock(return_value="A rewritten prompt")
        suggest = function_from(
            "ruinedfooocus/features/prompting/ui_rewrite.py",
            "suggest",
            {"gr": gr, "run_prompt": model},
        )
        preview, original, enabled = suggest("my-preset.txt", "Original prompt")
        self.assertEqual(original, "Original prompt")
        self.assertEqual(preview["value"], "A rewritten prompt")
        apply = function_from(
            "ruinedfooocus/features/prompting/ui_rewrite.py", "apply_rewrite", {"gr": gr}
        )
        with self.assertRaisesRegex(ValueError, "prompt changed"):
            apply("New user edit", original, preview["value"])
        result = apply(original, original, preview["value"])
        self.assertEqual(
            result[:2], ("A rewritten prompt", ("Original prompt", "A rewritten prompt"))
        )

    def test_rewrite_undo_protects_later_user_edits(self):
        gr = SimpleNamespace(Error=ValueError, update=lambda **values: values, skip=lambda: "skip")
        undo = function_from(
            "ruinedfooocus/features/prompting/ui_rewrite.py", "undo_rewrite", {"gr": gr}
        )
        record = ("Original", "Applied")
        with self.assertRaisesRegex(ValueError, "overwrite your edits"):
            undo("Later edit", record)
        self.assertEqual(undo("Applied", record)[0], {"value": "Original"})

    def test_image_submission_preserves_prompt_before_it_reaches_the_worker(self):
        policy = function_from(
            "ruinedfooocus/features/prompting/prompt_processing.py", "use_random_prompt", {}
        )
        add = Mock(return_value="task")
        submit = function_from(
            "ruinedfooocus/ui/app.py",
            "append_work",
            {
                "use_random_prompt": policy,
                "worker": SimpleNamespace(add_task=add),
                "get_promptlist": lambda data: data["prompt"].split("---"),
            },
        )
        config = dict(
            prompt="Keep the black hair and neon street.",
            image_number=2,
            input_image=object(),
            controlnet={"type": "img2img"},
            obp_assume_direct_control=True,
            obp_image_prompt=False,
        )
        self.assertEqual(submit(config), "task")
        self.assertEqual(add.call_args.args[0]["prompt"], config["prompt"])
        self.assertEqual(add.call_args.args[0]["image_total"], 2)
        submit(config | {"inpaint_toggle": True, "input_image": None})
        self.assertEqual(add.call_args.args[0]["prompt"], config["prompt"])
        submit(config | {"obp_image_prompt": True})
        self.assertEqual(add.call_args.args[0]["prompt"], "")
        submit(config | {"input_image": None})
        self.assertEqual(add.call_args.args[0]["prompt"], "")
        self.assertEqual(config["prompt"], "Keep the black hair and neon street.")

    def test_evolve_tokenizer_is_lazy_cached_and_respects_offline_mode(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/prompting/ui_evolve.py").read_text(encoding="utf-8")
        )
        node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evolve_tokenizer"
        )
        scope = dict(asset_path=asset_path, resource_path=resource_path, lru_cache=lru_cache)
        transformer = SimpleNamespace(CLIPTokenizer=SimpleNamespace(from_pretrained=Mock()))
        with patch.dict("sys.modules", {"transformers": transformer}):
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), "evolve_tokenizer", "exec"), scope
            )
            loader = transformer.CLIPTokenizer.from_pretrained
            loader.assert_not_called()
            first = scope["evolve_tokenizer"](True)
            self.assertIs(first, scope["evolve_tokenizer"](True))
            loader.assert_called_once_with("openai/clip-vit-large-patch14", local_files_only=True)
            scope["evolve_tokenizer"](False)
            self.assertEqual(loader.call_count, 2)
            loader.assert_called_with("openai/clip-vit-large-patch14", local_files_only=False)

    def test_evolve_offline_failure_is_actionable_and_token_conversion_is_preserved(self):
        tokenizer = SimpleNamespace(
            all_special_tokens=["SPECIAL"],
            get_vocab=lambda: ["word", "SPECIAL"],
            tokenize=lambda prompt: ["word"],
            convert_tokens_to_string=lambda tokens: "word</w>",
        )
        loader = Mock(return_value=tokenizer)
        scope = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            gr=SimpleNamespace(Error=ValueError),
            evolve_tokenizer=loader,
            os=SimpleNamespace(environ={}),
            random=SimpleNamespace(random=lambda: 1.0),
        )
        run = function_from(
            "ruinedfooocus/features/prompting/ui_evolve.py", "tokenize_and_randomize", scope
        )
        args = SimpleNamespace(offline=False)
        with patch.dict("sys.modules", {"ruinedfooocus.app.arguments": SimpleNamespace(args=args)}):
            for offline, environment, expected in [
                (False, {}, False),
                (True, {}, True),
                (False, {"RF_OFFLINE": "1"}, True),
            ]:
                args.offline = offline
                scope["os"].environ = environment
                self.assertEqual(run("word", 0), "word")
                loader.assert_called_with(expected)
            loader.side_effect = OSError("not cached")
            with self.assertRaisesRegex(ValueError, "Connect once to download"):
                run("word", 0)

    def submit(self, custom=False):
        return function_from(
            "ruinedfooocus/features/chat/ui.py",
            "llama_get_text",
            {
                "gr": SimpleNamespace(Error=ValueError),
                "custom_file": custom,
                "selected_file": "custom.gguf",
            },
        )

    def test_empty_and_stale_model_messages_are_rejected_before_submission(self):
        submit = self.submit()
        for message, active, selected in [
            ("  ", "a", "a"),
            ("hello", None, "a"),
            ("hello", "old", "new"),
        ]:
            with (
                self.subTest(message=message, active=active, selected=selected),
                self.assertRaises(ValueError),
            ):
                submit(message, True, active, selected)
        self.assertEqual(submit("hello", True, "a", "a"), ("hello", True))

    def test_configured_custom_model_remains_usable_without_catalogue_selection(self):
        self.assertEqual(
            self.submit(custom=True)("hello", False, "custom.gguf", None), ("hello", False)
        )
        with self.assertRaises(ValueError):
            self.submit(custom=True)("hello", False, "old.gguf", None)

    def test_invalid_submit_leaves_conversation_untouched_and_never_calls_worker(self):
        warning, respond = Mock(), Mock()
        submit = function_from(
            "ruinedfooocus/features/chat/ui.py",
            "submit_message",
            {
                "gr": SimpleNamespace(Warning=warning, skip=lambda: "skip"),
                "llama_msg": "message",
                "llama_get_text": self.submit(),
                "llama_respond": respond,
            },
        )
        self.assertEqual(
            list(submit("", False, "a", "", "", [], "a", False, False, None, False)),
            [{"message": "skip"}],
        )
        warning.assert_called_once()
        respond.assert_not_called()

    def test_reset_produces_a_fresh_sandbox_document_for_identical_history(self):
        from ruinedfooocus.features.chat.chat_code import python_runner

        first, second = python_runner([]), python_runner([])
        self.assertNotEqual(first, second)
        self.assertIn('sandbox="allow-scripts"', second)
        self.assertNotIn("allow-same-origin", second)

    def test_missing_chat_reference_preserves_message_without_submitting_or_loading(self):
        warning, respond = Mock(), Mock()
        submit = function_from(
            "ruinedfooocus/features/chat/ui.py",
            "submit_message",
            {
                "gr": SimpleNamespace(Warning=warning, skip=lambda: "skip"),
                "llama_msg": "message",
                "llama_get_text": self.submit(),
                "llama_respond": respond,
            },
        )
        for mode in ("Uploaded image", "Previous generated image"):
            with self.subTest(mode=mode):
                result = list(
                    submit(
                        "Keep this prompt.",
                        True,
                        "a",
                        "",
                        "",
                        [],
                        "a",
                        False,
                        False,
                        None,
                        False,
                        mode,
                    )
                )
                self.assertEqual(result, [{"message": "skip"}])
        self.assertEqual(warning.call_count, 2)
        respond.assert_not_called()

    def test_search_single_page_has_no_phantom_second_page(self):
        search = function_from(
            "ruinedfooocus/features/gallery/imagebrowser.py",
            "search_metadata",
            {"gr": SimpleNamespace(update=lambda **values: values), "Tuple": tuple, "List": list},
        )
        browser = SimpleNamespace(
            load_images=lambda page: (["image.png"], ""), num_images_pages=lambda: (1, 1)
        )
        result = search(browser, "test")
        self.assertEqual(result[1], {"value": 1, "maximum": 1})
        self.assertEqual(browser.filter, "test")


if __name__ == "__main__":
    unittest.main()
