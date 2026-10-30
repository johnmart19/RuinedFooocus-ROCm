"""CPU-only installer plans; no downloads or installation."""
import unittest
from unittest.mock import patch
import os
from modules import rocm_installer


class RocmPackagePlanTests(unittest.TestCase):
    def test_release_bundle_uses_validated_stable_versions_and_target_extra(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
                rocm_installer, 'detect_rocm_targets', return_value=['gfx1100']), patch.object(
                rocm_installer.platform, 'machine', return_value='AMD64'):
            plan = rocm_installer.automatic_rocm_plan('Windows', rocm_installer.ROCM_PLATFORM, force=True)
        self.assertEqual(plan.args[:3], ['torch[device-gfx1100]==2.14.0+rocm10.1.0',
            'torchvision[device-gfx1100]==0.29.0a0+rocm10.1.0', 'torchaudio==2.11.0.3+rocm10.1.0'])
        self.assertIn(rocm_installer.ROCM_INDEX, plan.args)

    def test_install_validates_native_vision_and_gpu_before_returning(self):
        with patch.object(rocm_installer.subprocess, 'run') as run:
            rocm_installer.install_rocm(['example-wheel'])
        self.assertEqual(run.call_count, 3)
        self.assertIn('--dry-run', run.call_args_list[0].args[0])
        smoke = run.call_args_list[-1].args[0][-1]
        self.assertIn('import torch, torchvision, torchaudio', smoke)
        self.assertIn('torchvision.ops.nms', smoke)

    def test_explicit_other_backend_does_not_detect_or_install_amd(self):
        with patch.dict(os.environ, {'TORCH_PLATFORM': 'cu128'}), patch.object(
                rocm_installer, 'detect_rocm_targets') as detect:
            self.assertIsNone(rocm_installer.automatic_rocm_plan('Windows', 'cu128'))
            detect.assert_not_called()

    def test_unsupported_gpu_keeps_automatic_selection_but_explicit_request_fails(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
                rocm_installer, 'detect_rocm_targets', return_value=['gfx803']), patch(
                'torchruntime.device_db.get_gpus', return_value=[type('Gpu', (), {'vendor_id': '1002'})()]):
            self.assertIsNone(rocm_installer.automatic_rocm_plan('Windows', 'directml'))
            self.assertIsNone(rocm_installer.automatic_rocm_plan('Linux', 'rocm6.2'))
            with self.assertRaises(RuntimeError):
                rocm_installer.automatic_rocm_plan('Windows', 'directml', force=True)
        with patch.dict(os.environ, {'TORCH_PLATFORM': rocm_installer.ROCM_PLATFORM}), patch.object(
                rocm_installer, 'detect_rocm_targets', return_value=['gfx803']):
            with self.assertRaises(RuntimeError):
                rocm_installer.automatic_rocm_plan('Windows', rocm_installer.ROCM_PLATFORM)
