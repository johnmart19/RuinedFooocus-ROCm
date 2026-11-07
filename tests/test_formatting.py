"""Developer HTML formatting must preserve executable and user-visible literals."""

import unittest

from tools.format_code import canonical_html


class FormattingTests(unittest.TestCase):
    def test_normalizes_html_tokens_but_preserves_script_strings_and_svg(self):
        text = '<!doctype html>\n<meta charset="utf-8" />\n<script>const text = "<img />";</script>\n<svg><path d="M 0 1" /></svg>\n'
        result = canonical_html(text)
        self.assertTrue(result.startswith('<!DOCTYPE html>\n<meta charset="utf-8">'))
        self.assertIn('const text = "<img />";', result)
        self.assertIn('<path d="M 0 1" />', result)
        self.assertEqual(canonical_html(result), result)


if __name__ == "__main__":
    unittest.main()
