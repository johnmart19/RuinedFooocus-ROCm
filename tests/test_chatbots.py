from ruinedfooocus.core.paths import asset_path, resource_path

"""Bundled chat bot definitions load and match what the Chat bots tab and context indexer accept."""
import ast
import json
import re
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


class ChatbotDefinitionTests(unittest.TestCase):
    def bots(self):
        folders = sorted(
            path
            for path in (ROOT / "ruinedfooocus/resources/assistants").iterdir()
            if path.is_dir()
        )
        self.assertTrue(folders)
        return [
            (folder, json.loads((folder / "info.json").read_text(encoding="utf-8")))
            for folder in folders
        ]

    def test_each_bot_has_a_name_greeting_system_prompt_and_avatar(self):
        names = set()
        for folder, info in self.bots():
            with self.subTest(bot=folder.name):
                for key in ("name", "greeting", "system"):
                    self.assertIsInstance(info.get(key), str)
                    self.assertTrue(info[key].strip())
                self.assertTrue((folder / "avatar.png").is_file())
                self.assertNotIn(info["name"].casefold(), names)
                names.add(info["name"].casefold())

    def test_bundled_bots_work_offline_from_local_sources(self):
        for folder, info in self.bots():
            for source in info.get("embed", []):
                with self.subTest(bot=folder.name, source=source):
                    self.assertEqual(len(source), 2)
                    # Bundled bots must not depend on another repository or the network.
                    self.assertIn(source[0], ("text", "file"))
                    if source[0] == "file":
                        self.assertTrue((ROOT / source[1]).is_file())


def index_sources():
    tree = ast.parse(
        (ROOT / "ruinedfooocus/features/chat/llama_pipeline.py").read_text(encoding="utf-8")
    )
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "pipeline")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "index_sources")
    scope = {
        "asset_path": asset_path,
        "resource_path": resource_path,
        "Path": Path,
        "re": re,
        "__file__": str(ROOT / "ruinedfooocus/features/chat/llama_pipeline.py"),
        "load_file_from_url": None,
        "url_to_filename": None,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "llama_pipeline", "exec"), scope)
    return scope["index_sources"]


class ContextSourceTests(unittest.TestCase):
    def run_index(self, sources):
        indexed = []

        class Embeddings:
            def __init__(self, **kwargs):
                pass

            def index(self, documents):
                indexed.extend(documents)

        stubs = {
            "txtai": types.SimpleNamespace(Embeddings=Embeddings),
            "ruinedfooocus.app.arguments": types.SimpleNamespace(
                args=SimpleNamespace(cpu=True, directml=None)
            ),
        }
        owner = SimpleNamespace()
        with mock.patch.dict(sys.modules, stubs):
            index_sources()(owner, sources)
        return indexed

    def test_file_sources_are_read_from_the_app_folder_and_split_into_passages(self):
        docs = self.run_index(
            [["file", "ruinedfooocus/resources/assistants/rf_support_troll/knowledge.md"]]
        )
        self.assertGreater(len(docs), 5)
        self.assertTrue(any("MetadataIncompleteBuffer" in doc for doc in docs))

    def test_file_sources_outside_the_app_folder_or_missing_are_rejected(self):
        for path in ("../outside.md", "chatbots/missing.md"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.run_index([["file", path]])


if __name__ == "__main__":
    unittest.main()
