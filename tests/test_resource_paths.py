"""Resource discovery must preserve customization without writing to resources."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.core import paths
from ruinedfooocus.core.path import PathManager
from ruinedfooocus.features.chat.assistants import assistant_choices, assistant_info, context_file
from ruinedfooocus.features.prompting.presets import rewrite_presets


class ResourcePathsTests(unittest.TestCase):
    def test_data_and_legacy_model_paths_are_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, RF_DATA_DIR=folder):
            self.assertEqual(
                paths.data_path("settings", "settings.json"),
                (Path(folder) / "settings/settings.json").resolve(),
            )
            owner = PathManager.__new__(PathManager)
            self.assertEqual(
                owner.get_abspath("../models/checkpoints"),
                (Path(folder) / "models/checkpoints").resolve(),
            )
            self.assertEqual(owner.get_abspath(folder), Path(folder))
            self.assertTrue(paths.resource_path("defaults", "styles.default").is_file())

    def test_rewrite_overrides_are_selected_once_and_never_modified(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, RF_DATA_DIR=folder):
            old = Path(folder) / "llamas/default.txt"
            old.parent.mkdir()
            old.write_text("My legacy preset\nKeep my words.", encoding="utf-8")
            before = old.read_bytes()
            choices = rewrite_presets()
            self.assertIn(("My legacy preset", str(old.resolve())), choices)
            new = Path(folder) / "prompt_presets/default.txt"
            new.parent.mkdir()
            new.write_text("My current preset\nKeep every count.", encoding="utf-8")
            choices = rewrite_presets()
            self.assertIn(("My current preset", str(new.resolve())), choices)
            self.assertNotIn(("My legacy preset", str(old.resolve())), choices)
            self.assertEqual(old.read_bytes(), before)

    def test_custom_assistant_overrides_bundle_and_general_chat_needs_no_persona(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, RF_DATA_DIR=folder):
            bot = Path(folder) / "assistants/cat"
            bot.mkdir(parents=True)
            definition = {"name": "My assistant", "greeting": "Hello", "system": "Be precise."}
            file = bot / "info.json"
            file.write_text(json.dumps(definition), encoding="utf-8")
            before = file.read_bytes()
            choices = assistant_choices()
            self.assertIn(("My assistant", str(bot.resolve())), choices)
            self.assertEqual(assistant_info(str(bot))["system"], "Be precise.")
            self.assertEqual(assistant_info("__normal__")["chatstart"], [])
            self.assertEqual(file.read_bytes(), before)

    def test_context_uses_legacy_bundled_path_and_rejects_escape(self):
        file = context_file("chatbots/rf_support_troll/knowledge.md")
        self.assertEqual(
            file, paths.resource_path("assistants/rf_support_troll/knowledge.md").resolve()
        )
        with self.assertRaises(ValueError):
            context_file("../outside.md")


if __name__ == "__main__":
    unittest.main()
