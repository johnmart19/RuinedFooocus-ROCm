"""CPU-only installer plans; no downloads or installation."""
import unittest
from unittest.mock import patch
from modules import gpu_installer


class TorchPackagePlanTests(unittest.TestCase):
    def test_cpu_plan_uses_the_verified_patch_bundle(self):
        commands = gpu_installer.torch_install_commands(None, 'cpu', 'Windows')
        self.assertEqual(commands[0][:2], ['torch==2.14.1', 'torchvision==0.29.1'])
        self.assertEqual(commands[0][-2:], ['--index-url', 'https://download.pytorch.org/whl/cpu'])

    def test_cuda_channels_keep_cpu_audio_isolated_and_do_not_execute_commands(self):
        with patch('modules.cuda_selection.cuda_wheels_available', return_value=True), patch(
                'modules.cuda_selection.nvidia_devices', return_value=((8.6,), (13, 2))), patch.object(
                gpu_installer.subprocess, 'run') as execute:
            for channel in ('cu126', 'cu130', 'cu132'):
                with self.subTest(channel=channel):
                    commands = gpu_installer.torch_install_commands(None, channel, 'Windows')
                    self.assertEqual(commands[0][:2], [f'torch==2.14.1+{channel}', f'torchvision==0.29.1+{channel}'])
                    self.assertIn('--no-deps', commands[1])
                    self.assertIn('torchaudio==2.11.0+cpu', commands[1])
            execute.assert_not_called()

    def test_other_gpu_runtimes_use_vulkan_llm_wheels(self):
        for torch_platform, os_platform in [('xpu', 'Windows'), ('rocm6.2', 'Linux'), ('cu118', 'Linux')]:
            with self.subTest(torch_platform=torch_platform):
                self.assertTrue(gpu_installer.xllamacpp_index(torch_platform, os_platform).endswith('/vulkan'))
        self.assertEqual(gpu_installer.xllamacpp_index('cpu', 'Linux'), 'https://pypi.org/simple')
        self.assertEqual(gpu_installer.xllamacpp_index('mps', 'Darwin'), 'https://pypi.org/simple')


class TorchSetupTests(unittest.TestCase):
    def test_explicit_choice_resolves_before_replacing_torch(self):
        from modules import torch_setup
        with patch('modules.cuda_selection.cuda_wheels_available', return_value=True), patch(
                'modules.cuda_selection.nvidia_devices', return_value=((8.6,), (13, 2))), patch.object(
                torch_setup.subprocess, 'run') as execute, patch(
                'modules.runtime_support.inspect_installed_cuda',
                return_value={'cuda': '13.2', 'versions': {'torch': '2.14.1+cu132'}}):
            torch_setup.install('cu132')
        commands = [call.args[0] for call in execute.call_args_list]
        self.assertEqual(len(commands), 4)
        self.assertTrue(all('--dry-run' in command for command in commands[:2]))
        self.assertFalse(any('--dry-run' in command for command in commands[2:]))
        self.assertTrue(all('--force-reinstall' in command for command in commands))

    def test_unknown_or_incompatible_choice_changes_nothing(self):
        from modules import torch_setup
        with patch.object(torch_setup.subprocess, 'run') as execute:
            with self.assertRaises(SystemExit):
                torch_setup.install('rocm6.4')
            with patch('modules.cuda_selection.nvidia_devices', return_value=((6.1,), (13, 2))), \
                    self.assertRaises(RuntimeError):
                torch_setup.install('cu132')  # Pascal is not supported by CUDA 13 bundles.
        execute.assert_not_called()
