"""An empty or changing library must not invalidate the current UI selection."""

import unittest
from unittest.mock import patch

import gradio as gr

from ruinedfooocus.features.models.selection import checkpoint_choices
from ruinedfooocus.features.models.ui_download import download_default


class ModelSelectionTests(unittest.TestCase):
    def test_missing_selection_is_retained_and_newly_installed_one_has_no_duplicate(self):
        self.assertEqual(
            checkpoint_choices([], "default.safetensors"),
            [("default.safetensors (not installed)", "default.safetensors")],
        )
        self.assertEqual(
            checkpoint_choices(["default.safetensors"], "default.safetensors"),
            ["default.safetensors"],
        )
        names = ["other.safetensors"]
        checkpoint_choices(names, "missing.safetensors")
        self.assertEqual(names, ["other.safetensors"])

    def test_offline_download_is_rejected_before_loading_state_or_fetching(self):
        with patch("ruinedfooocus.features.models.ui_download.offline", return_value=True):
            with self.assertRaisesRegex(gr.Error, "Offline mode"):
                download_default()


if __name__ == "__main__":
    unittest.main()
