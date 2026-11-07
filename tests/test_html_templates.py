from ruinedfooocus.core.paths import asset_path, resource_path
from ruinedfooocus.core.paths import asset_path

"""App-owned HTML follows the HTML Living Standard basics; no server or browser needed."""
import ast
from html.parser import HTMLParser
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
PAGES = (
    str(asset_path("last_image.html")),
    str(asset_path("slideshow.html")),
    str(asset_path("chat_python.html")),
)
FRAGMENTS = (str(asset_path("chat_local_models.html")), str(asset_path("chat_model_remove.html")))
OBSOLETE = {"font", "center", "big", "strike", "marquee", "tt", "frame", "frameset"}
PRESENTATIONAL = {"bgcolor", "align", "border", "cellpadding", "cellspacing", "valign"}


class Collector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.doctype = None

    def handle_decl(self, decl):
        self.doctype = decl

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    handle_startendtag = handle_starttag


def parse(path):
    collector = Collector()
    collector.feed((ROOT / path).read_text(encoding="utf-8"))
    return collector


class HtmlTemplateTests(unittest.TestCase):
    def test_pages_declare_doctype_language_charset_and_title(self):
        for path in PAGES:
            with self.subTest(page=path):
                page = parse(path)
                self.assertEqual(page.doctype, "DOCTYPE html")
                self.assertEqual(page.tags[0], ("html", {"lang": "en"}))
                self.assertEqual(page.tags[1][0], "head")
                self.assertEqual(page.tags[2], ("meta", {"charset": "utf-8"}))
                self.assertIn("title", [tag for tag, _ in page.tags])

    def test_markup_has_no_obsolete_elements_or_presentational_attributes(self):
        for path in PAGES + FRAGMENTS:
            text = (ROOT / path).read_text(encoding="utf-8")
            with self.subTest(file=path):
                self.assertNotRegex(text, r"<[^>]*\s/>")  # void elements without trailing slashes
                for tag, attrs in parse(path).tags:
                    self.assertNotIn(tag, OBSOLETE)
                    self.assertFalse(PRESENTATIONAL & set(attrs), (tag, attrs))
                    if tag == "img":
                        self.assertIn("alt", attrs)
                    if tag == "button":
                        self.assertEqual(attrs.get("type"), "button")

    def test_python_built_markup_avoids_obsolete_tags(self):
        for path in (
            "ruinedfooocus/ui/app.py",
            "ruinedfooocus/features/prompting/ui_random.py",
            "ruinedfooocus/ui/html.py",
        ):
            with self.subTest(file=path):
                text = (ROOT / path).read_text(encoding="utf-8")
                self.assertNotRegex(text, r"<(?:font|center|big|strike|marquee)\b")
                for image in re.findall(r"<img\b[^>]*>", text):
                    self.assertIn("alt=", image)

    def test_progress_bar_is_labelled_and_escapes_status_text(self):
        source = (ROOT / "ruinedfooocus/ui/html.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        nodes = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and node.targets[0].id == "progress_html"
            or isinstance(node, ast.FunctionDef)
            and node.name == "make_progress_html"
        ]
        import html as html_text

        scope = {
            "asset_path": asset_path,
            "resource_path": resource_path,
            "state": {"last_progress": 0},
            "html_text": html_text,
        }
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), "ruinedfooocus/ui/html.py", "exec"),
            scope,
        )
        markup = scope["make_progress_html"](40, 'Loading <model> "a&b"')
        self.assertIn("&lt;model&gt; &quot;a&amp;b&quot;", markup)
        tags = [tag for tag, _ in parse_text(markup).tags]
        self.assertEqual(tags[0], "label")
        self.assertIn("progress", tags)
        self.assertNotIn("div", tags)  # phrasing content only inside <label>


def parse_text(text):
    collector = Collector()
    collector.feed(text)
    return collector


if __name__ == "__main__":
    unittest.main()
