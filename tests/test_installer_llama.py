"""CPU-only installer plans; no downloads or installation."""
import unittest
from unittest.mock import patch
import os
from modules import llama_installer


class NativePackagePlanTests(unittest.TestCase):
    def test_supported_packages_match_published_asset_names(self):
        cases = [
            ('Windows', 'AMD64', False, 'Auto', 'win-cpu-x64.zip'),
            ('Windows', 'AMD64', True, 'Vulkan', 'win-vulkan-x64.zip'),
            ('Windows', 'AMD64', True, 'ROCm', 'win-rocm-10.0-x64.zip'),
            ('Windows', 'AMD64', True, 'CUDA 12.4', 'win-cuda-12.4-x64.zip'),
            ('Windows', 'AMD64', True, 'CUDA 13.4', 'win-cuda-13.4-x64.zip'),
            ('Windows', 'ARM64', False, 'CPU', 'win-cpu-arm64.zip'),
            ('Windows', 'ARM64', True, 'Auto', 'win-vulkan-arm64.zip'),
            ('Windows', 'ARM64', True, 'CUDA 13.4', 'win-cuda-13.4-arm64.zip'),
            ('Linux', 'x86_64', True, 'CUDA 12.8', 'ubuntu-cuda-12.8-x64.tar.gz'),
            ('Linux', 'x86_64', True, 'CUDA 13.4', 'ubuntu-cuda-13.4-x64.tar.gz'),
            ('Linux', 'aarch64', True, 'CUDA 13.4', 'ubuntu-cuda-13.4-arm64.tar.gz'),
            ('Linux', 'x86_64', False, 'CPU', 'ubuntu-x64.tar.gz'),
            ('Linux', 'x86_64', True, 'Vulkan', 'ubuntu-vulkan-x64.tar.gz'),
            ('Linux', 'x86_64', True, 'ROCm', 'ubuntu-rocm-10.0-x64.tar.gz'),
            ('Linux', 'aarch64', True, 'Vulkan', 'ubuntu-vulkan-arm64.tar.gz'),
            ('Darwin', 'arm64', True, 'Auto', 'macos-arm64.tar.gz'),
            ('Darwin', 'x86_64', False, 'CPU', 'macos-x64.tar.gz'),
        ]
        with patch('modules.llama_build.built_server', return_value=None):
            for system, machine, gpu, backend, suffix in cases:
                with self.subTest(system=system, machine=machine, backend=backend):
                    self.assertEqual(llama_installer.package_name(system, machine, gpu, backend=backend),
                                     f'llama-{llama_installer.VERSION}-bin-{suffix}')

    def test_explicit_cuda_requires_driver_and_architecture(self):
        for capabilities, driver in [((8.6,), (13, 3)), ((6.1,), (13, 4)), ((), None)]:
            with self.subTest(capabilities=capabilities, driver=driver), patch(
                    'modules.cuda_selection.nvidia_devices', return_value=(capabilities, driver)):
                with self.assertRaises(RuntimeError):
                    llama_installer.validate_cuda_backend('CUDA 13.4')
        with patch('modules.cuda_selection.nvidia_devices', return_value=((8.6,), (13, 4))):
            llama_installer.validate_cuda_backend('CUDA 13.4')

    def test_custom_server_bypasses_binary_download(self):
        from pathlib import Path
        executable = Path(__file__).resolve()
        with patch.dict(os.environ, {'LLAMA_SERVER': str(executable)}), patch.object(
                llama_installer.requests, 'get') as download:
            self.assertEqual(llama_installer.server_path(), executable)
            download.assert_not_called()

    def test_linux_cuda_runtime_uses_bundled_cuda_libraries(self):
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'LD_LIBRARY_PATH': '/opt/lib'}), patch.object(
                llama_installer.platform, 'system', return_value='Linux'), patch.object(
                llama_installer.platform, 'machine', return_value='x86_64'), patch.object(
                llama_installer, 'package_directory', return_value=Path(folder)):
            libraries = Path(folder) / 'cudart-llama-bin-ubuntu-cuda-13.4-x64'
            libraries.mkdir()
            (libraries / 'libcudart.so.13').write_bytes(b'')
            environment = llama_installer.runtime_environment(True, backend='CUDA 13.4')
        self.assertEqual(environment['LD_LIBRARY_PATH'], f'{libraries}:/opt/lib')
