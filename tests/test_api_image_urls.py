"""Exercise actual Windows path normalization without importing application services."""
import ast
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote, unquote
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(name, **extra):
    tree = ast.parse((ROOT / "modules/api.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(Path=Path, quote=quote, shared=SimpleNamespace(local_url="http://127.0.0.1:7860/"), **extra)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "modules/api.py", "exec"), scope)
    return scope[name]


class ImageURLTests(unittest.TestCase):
    def test_search_url_resolves_dot_segments_and_quotes_filename(self):
        path = ROOT / "modules" / ".." / "outputs" / "image #1 ?.png"
        url = function("image_file_url")(str(path))
        self.assertNotIn("\\", url)
        self.assertNotIn("/../", url)
        self.assertIn("image%20%231%20%3F.png", url)
        self.assertEqual(unquote(url.split("file=", 1)[1]), path.resolve().as_posix())

    def test_last_image_returns_canonical_path_for_display_and_file_access(self):
        path = ROOT / "modules" / ".." / "outputs" / "image.png"
        get = function("get_last_image", state={"last_image": str(path)})
        self.assertEqual(get(), path.resolve().as_posix())
        self.assertEqual(function("get_last_image", state={})(), (ROOT / "html/logo.png").as_posix())


if __name__ == "__main__":
    unittest.main()


class ModelIdTests(unittest.TestCase):
    def test_subfolder_model_ids_are_portable_and_accept_windows_separators(self):
        import sys
        from unittest.mock import patch
        from modules import api_runtime
        native = r"LTX25\ltx-2.5.safetensors"
        catalogue = {"video/LTX25/ltx-2.5.safetensors": {"name": native, "kind": "video"}}
        llama = SimpleNamespace(DEFAULT_MODEL="chat.gguf")
        with patch.object(api_runtime, "models", return_value=catalogue), \
                patch.dict(sys.modules, {"shared": SimpleNamespace(), "modules.llama_models": llama}):
            for requested in ("video/LTX25/ltx-2.5.safetensors", "video/" + native, native):
                with self.subTest(requested=requested):
                    model, selected = api_runtime.resolve_model(requested, "video")
                    self.assertEqual(model, "video/LTX25/ltx-2.5.safetensors")
                    self.assertEqual(selected["name"], native)
        self.assertEqual(api_runtime.model_id(r"image/a\b\c.safetensors"), "image/a/b/c.safetensors")
