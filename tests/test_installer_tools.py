"""Setup helpers resolve bundles before any package replacement."""
import unittest
from unittest.mock import patch


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


if __name__ == '__main__':
    unittest.main()
