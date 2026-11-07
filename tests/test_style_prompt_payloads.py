"""Style templates may add text without rewriting placeholders in the user's text."""
import ast
from pathlib import Path
import random
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StylePayloadTests(unittest.TestCase):
    def apply(self, selected, prompt, keywords='metadata {prompt}'):
        tree = ast.parse((ROOT / 'modules/sdxl_styles.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'apply_style')
        scope = dict(random=random, re=re, STYLE_ALIASES={}, styles={
            'Frame': ('Frame {prompt}, , done', ''), 'Outer': ('Outer {prompt}', ''),
            'Keywords': ('{prompt} {lora_keywords}', ''), 'Negative only': (None,'noise')},
            default_style=('{prompt}',''), allstyles=[])
        exec(compile(ast.Module(body=[node], type_ignores=[]),'style_payload','exec'),scope)
        return scope['apply_style'](selected,prompt,'literal __negative__',keywords)

    def test_prompt_markers_and_punctuation_are_literal_across_multiple_styles(self):
        prompt = ' , Sign says {prompt}, , {lora_keywords}.  '
        positive, negative = self.apply(['Frame','Outer'],prompt)
        self.assertEqual(positive,'Outer Frame '+prompt+',  done, , ')
        self.assertTrue(negative.endswith('literal __negative__'))

    def test_keywords_are_expanded_only_in_templates_and_not_reexpanded(self):
        prompt = 'Sign says {lora_keywords} and {prompt}.'
        positive, _ = self.apply(['Keywords','Outer'],prompt)
        self.assertEqual(positive,'Outer '+prompt+' metadata {prompt}, , ')
        positive, _ = self.apply(['Keywords'],prompt,None)
        self.assertEqual(positive,prompt+' {lora_keywords}, ')

    def test_negative_only_and_empty_styles_leave_positive_payload_unchanged(self):
        prompt = ' , Sign says {prompt}, , {lora_keywords}.  '
        for selected in ([],['Negative only']):
            positive, _ = self.apply(selected,prompt)
            self.assertEqual(positive,prompt)
        positive, _ = self.apply(['Frame','Negative only'],prompt)
        self.assertEqual(positive,'Frame '+prompt+',  done, ')

    def test_styles_without_negative_text_leave_negative_prompt_exact(self):
        _, negative = self.apply(['Frame'],'Two cups.')
        self.assertEqual(negative,'literal __negative__')


if __name__ == '__main__':
    unittest.main()
