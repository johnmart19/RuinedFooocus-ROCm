"""Local #! commands must not leak into shared generation defaults."""
import importlib
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class HashbangTests(unittest.TestCase):
    def test_add_performance_saves_preset_without_changing_defaults(self):
        defaults = dict(custom_steps=30, cfg=8, sampler_name="dpmpp_2m_sde_gpu", scheduler="karras", clip_skip=1)
        saved = {}
        settings = SimpleNamespace(default_settings=defaults, load_performance=lambda: {},
                                   save_performance=lambda options: saved.update(options))
        shared = SimpleNamespace(performance_settings=settings, update_cfg=Mock())
        samplers = SimpleNamespace(KSampler=SimpleNamespace(SAMPLERS=["uni_pc"], SCHEDULERS=["simple"]))
        stubs = {"shared": shared, "modules.async_worker": Mock(), "comfy.samplers": samplers}
        with patch.dict(sys.modules, stubs):
            sys.modules.pop("modules.hashbang_pipeline", None)
            module = importlib.import_module("modules.hashbang_pipeline")
            prompt = "#!add performance\nname: WAN\nsteps: 12\ncfg: 4\nscheduler: simple\nsampler: uni_pc"
            with patch.object(module.Image, "open", return_value=Mock()):
                module.pipeline().process(dict(task_id=1, prompt=prompt))
        sys.modules.pop("modules.hashbang_pipeline", None)
        self.assertEqual(saved["WAN"], dict(custom_steps=12, cfg=4.0, sampler_name="uni_pc",
                                            scheduler="simple", clip_skip=1))
        self.assertEqual(defaults, dict(custom_steps=30, cfg=8, sampler_name="dpmpp_2m_sde_gpu",
                                        scheduler="karras", clip_skip=1))


if __name__ == "__main__":
    unittest.main()
