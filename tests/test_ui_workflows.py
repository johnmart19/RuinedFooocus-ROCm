"""Workflow validation without starting model workers or downloading files."""
import ast
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def function_from(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), namespace)
    return namespace[name]


class WorkflowControlsTests(unittest.TestCase):
    def test_evolve_tokenizer_is_lazy_cached_and_respects_offline_mode(self):
        tree = ast.parse((ROOT / 'modules/ui/ui_evolve.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evolve_tokenizer')
        scope = dict(lru_cache=lru_cache)
        transformer = SimpleNamespace(CLIPTokenizer=SimpleNamespace(from_pretrained=Mock()))
        with patch.dict('sys.modules', {'transformers': transformer}):
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'evolve_tokenizer', 'exec'), scope)
            loader = transformer.CLIPTokenizer.from_pretrained
            loader.assert_not_called()
            first = scope['evolve_tokenizer'](True)
            self.assertIs(first, scope['evolve_tokenizer'](True))
            loader.assert_called_once_with('openai/clip-vit-large-patch14', local_files_only=True)
            scope['evolve_tokenizer'](False)
            self.assertEqual(loader.call_count, 2)
            loader.assert_called_with('openai/clip-vit-large-patch14', local_files_only=False)

    def test_evolve_offline_failure_is_actionable_and_token_conversion_is_preserved(self):
        tokenizer = SimpleNamespace(all_special_tokens=['SPECIAL'], get_vocab=lambda: ['word', 'SPECIAL'],
                                    tokenize=lambda prompt: ['word'], convert_tokens_to_string=lambda tokens: 'word</w>')
        loader = Mock(return_value=tokenizer)
        scope = dict(gr=SimpleNamespace(Error=ValueError), evolve_tokenizer=loader,
                     os=SimpleNamespace(environ={}), random=SimpleNamespace(random=lambda: 1.0))
        run = function_from('modules/ui/ui_evolve.py', 'tokenize_and_randomize', scope)
        args = SimpleNamespace(offline=False)
        with patch.dict('sys.modules', {'argparser': SimpleNamespace(args=args)}):
            for offline, environment, expected in [(False, {}, False), (True, {}, True), (False, {'RF_OFFLINE':'1'}, True)]:
                args.offline = offline
                scope['os'].environ = environment
                self.assertEqual(run('word', 0), 'word')
                loader.assert_called_with(expected)
            loader.side_effect = OSError('not cached')
            with self.assertRaisesRegex(ValueError, 'Connect once to download'):
                run('word', 0)

    def submit(self, custom=False):
        return function_from('modules/ui/ui_llama_chat.py', 'llama_get_text', {
            'gr': SimpleNamespace(Error=ValueError), 'custom_file': custom, 'selected_file': 'custom.gguf'})

    def test_empty_and_stale_model_messages_are_rejected_before_submission(self):
        submit = self.submit()
        for message, active, selected in [('  ', 'a', 'a'), ('hello', None, 'a'), ('hello', 'old', 'new')]:
            with self.subTest(message=message, active=active, selected=selected), self.assertRaises(ValueError):
                submit(message, True, active, selected)
        self.assertEqual(submit('hello', True, 'a', 'a'), ('hello', True))

    def test_configured_custom_model_remains_usable_without_catalogue_selection(self):
        self.assertEqual(self.submit(custom=True)('hello', False, 'custom.gguf', None),
                         ('hello', False))
        with self.assertRaises(ValueError):
            self.submit(custom=True)('hello', False, 'old.gguf', None)

    def test_invalid_submit_leaves_conversation_untouched_and_never_calls_worker(self):
        warning, respond = Mock(), Mock()
        submit = function_from('modules/ui/ui_llama_chat.py', 'submit_message', {
            'gr': SimpleNamespace(Warning=warning, skip=lambda: 'skip'), 'llama_msg': 'message',
            'llama_get_text': self.submit(), 'llama_respond': respond})
        self.assertEqual(list(submit('', False, 'a', '', '', [], 'a', False, False, None, False)), [{'message': 'skip'}])
        warning.assert_called_once()
        respond.assert_not_called()

    def test_reset_produces_a_fresh_sandbox_document_for_identical_history(self):
        from modules.chat_code import python_runner
        first, second = python_runner([]), python_runner([])
        self.assertNotEqual(first, second)
        self.assertIn('sandbox="allow-scripts"', second)
        self.assertNotIn("allow-same-origin", second)

    def test_search_single_page_has_no_phantom_second_page(self):
        search = function_from('modules/imagebrowser.py', 'search_metadata', {
            'gr': SimpleNamespace(update=lambda **values: values), 'Tuple': tuple, 'List': list})
        browser = SimpleNamespace(load_images=lambda page: (['image.png'], ''),
                                  num_images_pages=lambda: (1, 1))
        result = search(browser, 'test')
        self.assertEqual(result[1], {'value': 1, 'maximum': 1})
        self.assertEqual(browser.filter, 'test')


if __name__ == '__main__':
    unittest.main()

