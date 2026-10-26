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
