"""Optional prompt assets remain lazy and preserve custom files and weights."""

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.features.prompting.expansion_resources import (
    ensure_expansion_resources,
    expansion_model_path,
)


class ExpansionResourcesTests(unittest.TestCase):
    def test_failed_copy_removes_only_its_new_partial_file(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "pytorch_model.bin").write_bytes(b"original weights")
            with patch(
                "ruinedfooocus.features.prompting.expansion_resources.shutil.copyfileobj",
                side_effect=OSError("write failed"),
            ):
                with self.assertRaisesRegex(OSError, "write failed"):
                    ensure_expansion_resources(folder)
            self.assertEqual(list(folder.iterdir()), [folder / "pytorch_model.bin"])
            self.assertEqual((folder / "pytorch_model.bin").read_bytes(), b"original weights")

    def test_existing_weights_and_custom_tokenizer_are_not_modified_or_downloaded(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "pytorch_model.bin").write_bytes(b"original weights")
            (folder / "tokenizer.json").write_bytes(b"custom tokenizer")
            download = Mock(side_effect=AssertionError("No network allowed"))
            with patch.dict(
                "sys.modules",
                {"ruinedfooocus.core.util": SimpleNamespace(load_file_from_url=download)},
            ):
                ensure_expansion_resources(folder)
            download.assert_not_called()
            self.assertEqual((folder / "pytorch_model.bin").read_bytes(), b"original weights")
            self.assertEqual((folder / "tokenizer.json").read_bytes(), b"custom tokenizer")
            self.assertTrue((folder / "config.json").is_file())

    def test_discovery_does_not_create_or_download_a_missing_model(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, RF_DATA_DIR=folder):
            destination = Path(folder) / "prompt_expansion"
            chosen = expansion_model_path()
            self.assertFalse(destination.exists())
            # A legacy installed model can be reused instead of downloaded again.
            self.assertTrue(
                chosen == destination.resolve() or (chosen / "pytorch_model.bin").is_file()
            )


if __name__ == "__main__":
    unittest.main()
