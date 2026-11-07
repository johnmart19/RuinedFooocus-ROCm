"""Native chat lifecycle checks with inert subprocess and HTTP replacements."""

import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.backends.llama import llama_server


class LlamaServerTests(unittest.TestCase):
    def test_server_logs_follow_writable_data_from_a_different_working_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            model = root / "original.gguf"
            model.write_bytes(b"fixture only; no model weights")
            application, data = root / "application", root / "data"
            application.mkdir()
            process = Mock()
            process.poll.return_value = None
            session = Mock()
            session.headers = {}
            session.get.return_value.ok = True
            current = Path.cwd()
            server = None
            try:
                os.chdir(application)
                with (
                    patch.dict(os.environ, RF_DATA_DIR=str(data)),
                    patch.object(llama_server.requests, "Session", return_value=session),
                    patch.object(llama_server.socket, "socket") as socket,
                    patch.object(llama_server, "server_path", return_value=Path("llama-server")),
                    patch.object(llama_server, "runtime_environment", return_value={}),
                    patch.object(llama_server, "rocm_target", return_value=None),
                    patch.object(llama_server, "projector_path", return_value=None),
                    patch.object(
                        llama_server.subprocess,
                        "run",
                        return_value=SimpleNamespace(returncode=0, stdout=""),
                    ),
                    patch.object(llama_server, "start_process", return_value=(process, None)),
                ):
                    socket.return_value.__enter__.return_value.getsockname.return_value = (
                        "127.0.0.1",
                        61234,
                    )
                    server = llama_server.Server(model, {}, gpu=False)
                    self.assertTrue((data / "cache/llama.cpp/server-61234.log").is_file())
                    self.assertFalse((application / "cache").exists())
                    self.assertEqual(model.read_bytes(), b"fixture only; no model weights")
            finally:
                if server is not None:
                    server.close()
                os.chdir(current)
            process.terminate.assert_called_once()
            session.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
