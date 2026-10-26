"""CPU-only plans and isolated runtime integrity; no packages or GPU work."""
import hashlib,json,sys,tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from modules import onnx_installer as installer

class OnnxInstallerTests(unittest.TestCase):
    def test_optional_install_is_isolated_pinned_and_does_not_replace_cpu_ort(self):
        with patch.object(installer.platform, 'system', return_value='Windows'), \
             patch.object(installer.platform, 'machine', return_value='AMD64'):
            plan = installer.install_plan('isolated-folder')
        self.assertIn('--target', plan)
        self.assertIn('--no-deps', plan)
        self.assertIn('onnxruntime-windowsml==' + installer.VERSION, plan)
        self.assertNotIn('onnxruntime', plan)
        with patch.object(installer.platform, 'system', return_value='Linux'):
            with self.assertRaises(RuntimeError):
                installer.install_plan()


    def test_marker_rejects_changed_binary_or_python_abi(self):
        with tempfile.TemporaryDirectory(dir=installer.ROOT / 'tmp') as folder:
            root = Path(folder)
            (root / 'binary.pyd').write_bytes(b'original')
            marker = dict(version=installer.VERSION, python=list(sys.version_info[:2]), binary='binary.pyd',
                          sha256=hashlib.sha256(b'original').hexdigest())
            (root / 'verified.json').write_text(json.dumps(marker))
            with patch.object(installer, 'runtime_path', return_value=root):
                self.assertTrue(installer.runtime_available())
                (root / 'binary.pyd').write_bytes(b'changed')
                self.assertFalse(installer.runtime_available())


