import os
import argparse
import enum
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import Mock, patch

from modules.comfy_compat import configure_device_environment, initialize_comfy_memory
from modules import comfy_compat
from modules.comfy_args import add_comfy_arguments, COMFY_DESTINATIONS


class ComputeArgumentContractTests(unittest.TestCase):
    def test_compute_parser_matches_checked_out_backend(self):
        """Check every exposed compute action without importing Torch/backend code."""
        comfy = types.ModuleType("comfy")
        options = types.ModuleType("comfy.options")
        options.args_parsing = False
        comfy.options = options
        source = Path(__file__).resolve().parents[1] / "repositories/ComfyUI/comfy/cli_args.py"
        with patch.dict(sys.modules, {"comfy": comfy, "comfy.options": options}):
            upstream = runpy.run_path(str(source))
        local = argparse.ArgumentParser()
        add_comfy_arguments(local)
        backend = upstream["parser"]
        expected = {a.dest: a for a in backend._actions}

        def normalized(value):
            if isinstance(value, enum.Enum):
                return value.value
            return value

        for action in local._actions:
            if action.dest in ("help", "normalvram"):
                continue
            with self.subTest(dest=action.dest):
                other = expected[action.dest]
                self.assertIn(action.dest, COMFY_DESTINATIONS)
                for field in ("option_strings", "default", "const", "nargs", "required"):
                    self.assertEqual(normalized(getattr(action, field)), normalized(getattr(other, field)))
                if action.type is not None and issubclass(action.type, enum.Enum):
                    self.assertEqual([v.value for v in action.type], [v.value for v in other.type])
                else:
                    self.assertEqual(action.type, other.type)
                self.assertEqual(type(action), type(other))
        destinations = set(COMFY_DESTINATIONS)
        def groups(parser):
            return {frozenset(a.dest for a in group._group_actions if a.dest in destinations)
                    for group in parser._mutually_exclusive_groups} - {frozenset()}
        self.assertEqual(groups(local), groups(backend))

    def test_high_ram_and_compiler_implications_reach_backend(self):
        parser = argparse.ArgumentParser()
        add_comfy_arguments(parser)
        args = parser.parse_args(["--high-ram", "--disable-comfy-compiler", "--fast", "cublas_ops"])
        args.offline = False
        backend = types.SimpleNamespace()
        cli = types.ModuleType("comfy.cli_args")
        from modules.comfy_args import PerformanceFeature
        cli.PerformanceFeature = PerformanceFeature
        cli.LatentPreviewMethod = types.SimpleNamespace(from_string=lambda value: value)
        with patch.dict(sys.modules, {"comfy.cli_args": cli}):
            comfy_compat.configure_comfy_runtime(args, backend)
        self.assertTrue(backend.high_ram)
        self.assertTrue(backend.cache_classic)
        self.assertTrue(backend.disable_cuda_graphs)
        self.assertEqual(backend.fast, {PerformanceFeature.CublasOps})


def device_args(**overrides):
    return types.SimpleNamespace(**(dict(cuda_device=None, default_device=0,
        gpu_device_id=None, oneapi_device_selector=None, deterministic=False) | overrides))


class DeviceSelectionTests(unittest.TestCase):
    def test_invalid_memory_controls_fail_before_device_probe_or_environment_changes(self):
        for name in ('reserve_vram', 'vram_headroom', 'async_offload'):
            for value in (-1, float('nan'), float('inf')):
                with self.subTest(name=name, value=value), patch.dict(os.environ, {}, clear=True), \
                        patch.object(comfy_compat, '_physical_device_count') as probe:
                    with self.assertRaisesRegex(ValueError, name.replace('_', '-')):
                        configure_device_environment(device_args(**{name: value}))
                    probe.assert_not_called()
                    self.assertNotIn('CUDA_VISIBLE_DEVICES', os.environ)
                    self.assertNotIn('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', os.environ)

    def test_zero_and_positive_memory_controls_are_valid(self):
        with patch.dict(os.environ, {}, clear=True):
            configure_device_environment(device_args(default_device=None, reserve_vram=0,
                                                     vram_headroom=4.0, async_offload=0))

    def test_backend_startup_defaults_and_user_overrides(self):
        for existing in ({}, {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "0", "MIMALLOC_PURGE_DELAY": "10"}):
            with patch.dict(os.environ, existing, clear=True), patch.object(
                    comfy_compat.platform, 'system', return_value='Windows'):
                configure_device_environment(device_args(default_device=None))
                self.assertEqual(os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'],
                                 existing.get('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '1'))
                self.assertEqual(os.environ['MIMALLOC_PURGE_DELAY'], existing.get('MIMALLOC_PURGE_DELAY', '0'))

    def test_single_nvidia_default_does_not_create_phantom_devices(self):
        from modules.cuda_selection import nvidia_devices
        responses = ["CUDA Version: 13.2", "GPU-test, 00000000:01:00.0, 8.6"] * 2
        with patch.dict(os.environ, {}, clear=True), patch(
                "modules.cuda_selection.subprocess.check_output", side_effect=responses):
            configure_device_environment(device_args())
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(nvidia_devices(), ((8.6,), (13, 2)))

    def test_multi_gpu_reorders_only_real_devices(self):
        with patch.dict(os.environ, {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}, clear=True), patch(
                "modules.cuda_selection.nvidia_devices", return_value=((8.6, 8.9, 8.6), (13, 2))):
            configure_device_environment(device_args(default_device=1))
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1,0,2")

    def test_ambiguous_nvidia_order_rejected_and_environment_restored(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-old"}, clear=True), patch(
                "modules.cuda_selection.nvidia_devices", return_value=((8.6, 8.9), (13, 2))):
            with self.assertRaisesRegex(ValueError, "PCI_BUS_ID"):
                configure_device_environment(device_args())
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-old")

    def test_amd_count_probed_without_visibility(self):
        def probe(*args, **kwargs):
            self.assertNotIn("HIP_VISIBLE_DEVICES", os.environ)
            return "2\n"
        with patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": "1"}, clear=True), patch(
                "modules.cuda_selection.nvidia_devices", return_value=((), None)), patch(
                "modules.comfy_compat.subprocess.check_output", side_effect=probe):
            configure_device_environment(device_args(default_device=1))
            self.assertEqual(os.environ["HIP_VISIBLE_DEVICES"], "1,0")

    def test_unavailable_index_does_not_change_visibility(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=True), patch(
                "modules.comfy_compat._physical_device_count", return_value=1):
            with self.assertRaisesRegex(ValueError, "unavailable"):
                configure_device_environment(device_args(default_device=1))
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")

    def test_explicit_selection_skips_default_probe(self):
        with patch.dict(os.environ, {}, clear=True), patch(
                "modules.comfy_compat._physical_device_count") as probe:
            configure_device_environment(device_args(cuda_device="2,0"))
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2,0")
            configure_device_environment(device_args(gpu_device_id=1))
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1")
            probe.assert_not_called()


class DynamicMemoryTests(unittest.TestCase):
    def setUp(self):
        names = ("comfy", "comfy.cli_args", "comfy.model_management",
                 "comfy.memory_management", "comfy.model_patcher", "comfy_aimdo", "comfy_aimdo.control")
        self.modules = {name: types.ModuleType(name) for name in names}
        for name, module in self.modules.items():
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(self.modules[parent], child, module)
        self.args = types.SimpleNamespace(reserve_vram=1.5, disable_nvml_pressure=False,
            vram_headroom=0.5, directml=None, cpu=False, enable_dynamic_vram=False,
            disable_dynamic_vram=False, highvram=False, gpu_only=False, novram=False)
        cli = self.modules["comfy.cli_args"]
        cli.args = self.args
        cli.enables_dynamic_vram = lambda: self.args.enable_dynamic_vram or not any(
            getattr(self.args, key) for key in ("disable_dynamic_vram", "highvram", "gpu_only", "novram", "cpu"))
        self.control = self.modules["comfy_aimdo.control"]
        self.control.init = Mock()
        self.control.init_devices = Mock(return_value=True)
        self.management = self.modules["comfy.model_management"]
        self.management.is_nvidia = Mock(return_value=True)
        self.management.is_amd = Mock(return_value=False)
        self.management.rocm_version = (7, 15)
        self.management.torch_version_numeric = (2, 14)
        self.management.get_all_torch_devices = Mock(return_value=[types.SimpleNamespace(index=0)])
        self.memory = self.modules["comfy.memory_management"]
        self.memory.aimdo_enabled = False
        self.patcher = self.modules["comfy.model_patcher"]
        self.legacy = object()
        self.patcher.CoreModelPatcher = self.legacy
        self.patcher.ModelPatcherDynamic = object()
        self.scope = patch.dict(sys.modules, self.modules)
        self.scope.start()
        self.addCleanup(self.scope.stop)

    def test_controller_failure_never_selects_dynamic_patcher(self):
        self.control.init.return_value = False
        self.assertFalse(initialize_comfy_memory())
        self.control.init_devices.assert_not_called()
        self.assertIs(self.patcher.CoreModelPatcher, self.legacy)

    def test_backend_initializes_once_before_shared_state(self):
        order = []
        support = types.SimpleNamespace(inspect_installed_rocm=lambda platform: None,
            inspect_installed_cuda=lambda platform: None, select_torch_platform=lambda *args: "rocm10.1")
        with patch.dict(os.environ, {}, clear=True), \
                patch.dict(sys.modules, {"torchruntime": types.SimpleNamespace(), "modules.runtime_support": support}), \
                patch.object(comfy_compat, "_runtime_platform", None), \
                patch.object(comfy_compat, "configure_torch_allocator", side_effect=lambda *args: order.append("allocator")), \
                patch.object(comfy_compat, "configure_torch_compatibility", side_effect=lambda: order.append("compatibility")), \
                patch.object(comfy_compat, "configure_comfy_runtime", side_effect=lambda *args: order.append("flags")), \
                patch.object(comfy_compat, "initialize_comfy_memory", side_effect=lambda: order.append("memory")):
            first = comfy_compat.initialize_backend(self.args)
            self.assertEqual(comfy_compat.initialize_backend(self.args), first)
            self.assertEqual(os.environ["OCL_SET_SVM_SIZE"], "4194304")
        self.assertEqual(order, ["allocator", "compatibility", "flags", "memory"])

    def test_nvidia_activation_applies_headroom_and_patcher(self):
        self.assertTrue(initialize_comfy_memory())
        self.control.init.assert_called_once_with(simple_vram_headroom=int(1.5 * 1024**3), nvml_pressure=True)
        self.control.init_devices.assert_called_once_with([(0, 512 * 1024**2)])
        self.assertTrue(self.memory.aimdo_enabled)
        self.assertIs(self.patcher.CoreModelPatcher, self.patcher.ModelPatcherDynamic)

    def test_modern_rocm_activates(self):
        self.management.is_nvidia.return_value = False
        self.management.is_amd.return_value = True
        self.assertTrue(initialize_comfy_memory())

    def test_unsupported_hardware_and_old_torch_skip_even_when_forced(self):
        for kind in ("old_rocm", "directml", "cpu", "old_torch", "other"):
            with self.subTest(kind=kind):
                self.args.enable_dynamic_vram = True
                self.args.cpu = kind == "cpu"
                self.args.directml = 0 if kind == "directml" else None
                self.management.is_nvidia.return_value = kind in ("directml", "cpu", "old_torch")
                self.management.is_amd.return_value = kind == "old_rocm"
                self.management.rocm_version = (7, 2)
                self.management.torch_version_numeric = (2, 6) if kind == "old_torch" else (2, 14)
                self.assertFalse(initialize_comfy_memory())
                self.assertFalse(self.memory.aimdo_enabled)
                self.assertIs(self.patcher.CoreModelPatcher, self.legacy)
        self.control.init_devices.assert_not_called()

    def test_disabled_memory_modes_skip_initialization(self):
        for key in ("disable_dynamic_vram", "highvram", "gpu_only", "novram", "cpu"):
            with self.subTest(key=key):
                setattr(self.args, key, True)
                self.assertFalse(initialize_comfy_memory())
                setattr(self.args, key, False)
        self.control.init.assert_not_called()

    def test_activation_sets_controller_log_level_like_native_startup(self):
        self.modules["comfy.cli_args"].get_console_log_level = lambda outputs: outputs[0][0] if outputs else "INFO"
        for verbose, setter in (([], "set_log_info"), ([("WARNING", None)], "set_log_warning"),
                                ([("DETAIL", None)], "set_log_info")):  # DETAIL falls back when unsupported
            with self.subTest(verbose=verbose):
                for name in ("set_log_info", "set_log_warning"):
                    setattr(self.control, name, Mock())
                self.args.verbose = verbose
                self.assertTrue(initialize_comfy_memory())
                getattr(self.control, setter).assert_called_once_with()

    def test_device_failure_retains_legacy(self):
        for outcome in (False, RuntimeError("driver unavailable")):
            with self.subTest(outcome=outcome):
                if isinstance(outcome, Exception):
                    self.control.init_devices.side_effect = outcome
                else:
                    self.control.init_devices.return_value = outcome
                self.assertFalse(initialize_comfy_memory())
                self.assertFalse(self.memory.aimdo_enabled)
                self.assertIs(self.patcher.CoreModelPatcher, self.legacy)


class AimdoLogShutdownTests(unittest.TestCase):
    def setUp(self):
        import ctypes
        self.callback_type = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)
        self.callback = self.callback_type(lambda level, message: None)
        self.library = Mock()
        self.control = types.SimpleNamespace(lib=self.library, _LOG_CALLBACK=self.callback_type,
            _log_callback=self.callback, devctxs=[123])
        self.callbacks = []
        self.state = patch.object(comfy_compat, "_aimdo_log_detach_registered", False)
        self.state.start()
        self.addCleanup(self.state.stop)
        self.registration = patch.object(comfy_compat.atexit, "register", side_effect=self.callbacks.append)
        self.registration.start()
        self.addCleanup(self.registration.stop)

    def test_keeps_callback_alive_and_detaches_only_logging_at_exit(self):
        import gc
        import weakref
        callback_ref = weakref.ref(self.callback)
        self.assertTrue(comfy_compat.register_aimdo_log_shutdown(self.control, "Windows", True))
        self.library.set_log_callback.assert_not_called()
        self.control.lib = None
        self.control._log_callback = None
        self.callback = None
        gc.collect()
        self.assertIsNotNone(callback_ref())
        self.callbacks[0]()
        self.library.set_log_callback.assert_called_once()
        self.assertFalse(bool(self.library.set_log_callback.call_args.args[0]))
        self.library.cleanup.assert_not_called()
        self.library.plat_cleanup.assert_not_called()
        self.assertEqual(self.control.devctxs, [123])

    def test_registers_once(self):
        self.assertTrue(comfy_compat.register_aimdo_log_shutdown(self.control, "Windows", True))
        self.assertFalse(comfy_compat.register_aimdo_log_shutdown(self.control, "Windows", True))
        self.assertEqual(len(self.callbacks), 1)

    def test_other_platforms_and_uninitialized_abi_are_untouched(self):
        for system, amd in (("Linux", True), ("Windows", False), ("Darwin", True)):
            with self.subTest(system=system, amd=amd):
                self.assertFalse(comfy_compat.register_aimdo_log_shutdown(self.control, system, amd))
        self.assertFalse(comfy_compat.register_aimdo_log_shutdown(types.SimpleNamespace(lib=None), "Windows", True))
        self.assertFalse(comfy_compat.register_aimdo_log_shutdown(types.SimpleNamespace(lib=self.library), "Windows", True))
        self.assertEqual(self.callbacks, [])

    def test_activation_registers_only_after_success(self):
        # Inspect call ordering through the existing isolated controller fixture.
        fixture = DynamicMemoryTests(methodName="test_modern_rocm_activates")
        fixture.setUp()
        try:
            fixture.management.is_nvidia.return_value = False
            fixture.management.is_amd.return_value = True
            order = []
            fixture.control.init_devices.side_effect = lambda devices: order.append("devices") or True
            with patch.object(comfy_compat, "register_aimdo_log_shutdown", side_effect=lambda *args: order.append("shutdown")):
                self.assertTrue(initialize_comfy_memory())
            self.assertEqual(order, ["devices", "shutdown"])
            order.clear()
            fixture.control.init_devices.side_effect = lambda devices: order.append("devices") or False
            with patch.object(comfy_compat, "register_aimdo_log_shutdown") as register:
                self.assertFalse(initialize_comfy_memory())
            register.assert_not_called()
        finally:
            fixture.doCleanups()


if __name__ == "__main__":
    unittest.main()
