"""Prompt helper device/lifetime checks with inert models; no neural inference."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.features.prompting.superprompter.runtime import Runtime


class SuperPromptRuntimeTests(unittest.TestCase):
    def test_cpu_uses_cpu_without_gpu_detection_or_model_download(self):
        resource = Runtime()
        resource.model = Mock()
        resource.model.generate.return_value = ["tokens"]
        resource.tokenizer = Mock()
        resource.tokenizer.return_value.input_ids.to.return_value = "cpu tokens"
        resource.tokenizer.decode.return_value = "  keep two apples  "
        arguments = SimpleNamespace(args=SimpleNamespace(cpu=True, directml=None))
        with (
            patch.dict("sys.modules", {"ruinedfooocus.app.arguments": arguments}),
            patch("ruinedfooocus.features.prompting.superprompter.runtime.release_except"),
            patch("ruinedfooocus.features.prompting.superprompter.runtime.torch.manual_seed"),
            patch("torch.cuda.is_available", side_effect=AssertionError("GPU probe prohibited")),
        ):
            self.assertEqual(
                resource.generate("two apples", seed=1, max_new_tokens=8), "keep two apples"
            )
        resource.model.to.assert_called_once_with("cpu")
        self.assertEqual(resource.device, "cpu")

    def test_releasing_device_weights_never_removes_original_files(self):
        with tempfile.TemporaryDirectory() as folder:
            weights = Path(folder) / "model.safetensors"
            weights.write_bytes(b"original weights")
            resource = Runtime(folder)
            resource.model = Mock()
            resource.device = "cuda:0"  # A label only; the model is a mock.
            resource.close()
            resource.model.to.assert_called_once_with("cpu")
            self.assertEqual(weights.read_bytes(), b"original weights")
            self.assertEqual(resource.device, "cpu")


if __name__ == "__main__":
    unittest.main()
