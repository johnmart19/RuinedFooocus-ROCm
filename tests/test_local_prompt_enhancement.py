from ruinedfooocus.core.paths import asset_path, resource_path
"""Shared prompt enhancement preserves payloads, cancellation and helper ownership."""
import ast
from collections import defaultdict
from pathlib import Path
import re
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.features.prompting.prompt_enhancement import image_prompt_result, enhance_prompt

ROOT = Path(__file__).resolve().parents[1]


def function(path, name, scope):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), scope)
    return scope[name]


class Cancelled(Exception):
    pass


class LocalPromptTests(unittest.TestCase):
    def test_ui_enhancement_queues_and_worker_uses_existing_chat_helper(self):
        import ruinedfooocus.app.api_runtime as api_runtime
        worker = SimpleNamespace(worker_thread=None, check_interrupt=Mock())
        run = Mock(return_value='enhanced')
        with patch.dict(sys.modules, {'ruinedfooocus.app.async_worker':worker,
                                     'ruinedfooocus.features.chat.llama_pipeline':SimpleNamespace(run_llama=run)}), \
                patch('ruinedfooocus.app.async_worker',worker,create=True), \
                patch.object(api_runtime,'submit',return_value=object()) as submit, \
                patch.object(api_runtime,'wait_for_result',return_value='queued'):
            self.assertEqual(enhance_prompt('Two red cups.'),'queued')
            run.assert_not_called()
            worker.worker_thread = threading.current_thread()
            operation = submit.call_args.args[0]
            self.assertEqual(operation(object(),1),'enhanced')
            run.assert_called_once()
            self.assertEqual(run.call_args.args,(None,'Two red cups.'))

    def helper(self, chunks, load_error=None):
        owner = Mock()
        owner.load_base_model.side_effect = load_error
        owner.llm.handle_chat_completions.side_effect = lambda params, callback: [callback(c) for c in chunks]
        scope = dict(asset_path=asset_path, resource_path=resource_path, re=re, pipeline=Mock(return_value=owner), rewrite_template_kwargs=lambda: {},
            settings=SimpleNamespace(default_settings={'llm_hp_max_tokens':300}),
            model_management=SimpleNamespace(unload_all_models=Mock(), InterruptProcessingException=Cancelled))
        run = function('ruinedfooocus/features/chat/llama_pipeline.py', 'run_llama', scope)
        return run, owner, scope

    def test_stream_uses_fresh_messages_selected_runtime_and_configured_budget(self):
        prompt = 'Two red cups. Sign "Київ\\1". No people.'
        chunks = [{'choices':[{'delta':{'content':'Soft '}}]},
                  {'choices':[{'delta':{'content':'light.'},'finish_reason':'stop'}]}]
        run, owner, scope = self.helper(chunks)
        self.assertEqual(run(None,prompt,instruction='fidelity rules'),prompt+'\nSoft light.')
        owner.load_base_model.assert_called_once_with(vision=False)
        owner.unload.assert_called_once()
        scope['model_management'].unload_all_models.assert_called_once()
        params = owner.llm.handle_chat_completions.call_args.args[0]
        self.assertEqual(params['max_tokens'],300)
        self.assertEqual(params['messages'],[{'role':'system','content':'fidelity rules'},
                                           {'role':'user','content':prompt}])

    def test_truncation_empty_reasoning_and_load_failure_return_exact_original(self):
        prompt = ' , Keep "BLUE 42" and C:\\photos\\1. '
        for text, finish in (('partial','length'), ('partial',None), ('partial','content_filter'),
                             ('','stop'), ('<think>unfinished','stop')):
            with self.subTest(text=text):
                run, owner, _ = self.helper([{'choices':[{'delta':{'content':text},'finish_reason':finish}]}])
                self.assertEqual(run(None,prompt,instruction='rules'),prompt)
                owner.unload.assert_called_once()
        run, owner, _ = self.helper([],load_error=RuntimeError('load failed'))
        self.assertEqual(run(None,prompt,instruction='rules'),prompt)
        owner.unload.assert_called_once()
        owner.llm.handle_chat_completions.assert_not_called()

    def test_cancel_before_load_or_during_stream_propagates_and_cleans_up(self):
        for at in (1,3):
            with self.subTest(check=at):
                run, owner, _ = self.helper([{'choices':[{'delta':{'content':'partial'}}]}])
                calls = 0
                def cancel():
                    nonlocal calls
                    calls += 1
                    if calls == at:
                        raise Cancelled()
                with self.assertRaises(Cancelled):
                    run(None,'original',instruction='rules',check_cancelled=cancel)
                owner.unload.assert_called_once()
                if at == 1:
                    owner.load_base_model.assert_not_called()

    def processing(self, random_prompt):
        aliases = {'Erniehancer':'Local chat model','Style: Erniehancer':'Local chat model'}
        style_scope = dict(re=re, STYLE_ALIASES=aliases, styles={}, default_style=('{prompt}',''), allstyles=[])
        apply = function('ruinedfooocus/features/prompting/sdxl_styles.py','apply_style',style_scope)
        scope = dict(asset_path=asset_path, resource_path=resource_path, re=re, STYLE_ALIASES=aliases, available_styles={'Local chat model':('{prompt}','')},
                     apply_style=apply, use_random_prompt=lambda *args:random_prompt)
        scope['build_dynamic_prompt'] = lambda **kwargs:kwargs['enhancer']('original scene','Anime Model','')
        return function('ruinedfooocus/features/prompting/prompt_processing.py','process_prompt',scope)

    def test_legacy_style_uses_replacement_and_dedup_is_request_scoped(self):
        process = self.processing(False)
        enhance = Mock(side_effect=lambda text,*args,**kwargs:text+' enhanced')
        with patch('ruinedfooocus.features.prompting.prompt_enhancement.enhance_prompt',enhance):
            for _ in range(2):
                positive, negative = process([], '<style:Erniehancer>Two red cups.', 'no blur', {})
                self.assertEqual(positive,'Two red cups. enhanced')
                self.assertEqual(negative,'no blur')
        self.assertEqual(enhance.call_count,2)

    def test_onebutton_and_style_selection_enhance_once_in_one_request(self):
        process = self.processing(True)
        data = defaultdict(str,task_type='tool_call')
        enhance = Mock(return_value='enhanced scene')
        with patch('ruinedfooocus.features.prompting.prompt_enhancement.enhance_prompt',enhance):
            positive, _ = process(['Local chat model','Erniehancer'],'original','',data)
        self.assertEqual(positive,'enhanced scene')
        enhance.assert_called_once()

    def test_result_removes_completed_thoughts_without_rewriting_prompt_data(self):
        prompt='literal <style:Frame> {prompt} "Київ"'
        self.assertEqual(image_prompt_result({'finish_reason':'stop','message':{'content':'<think>private</think>Soft light.'}},prompt),prompt+'\nSoft light.')


if __name__ == '__main__':
    unittest.main()
