import json
import tempfile
import time
import sys
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.model_recommendations import ModelRecommendations, example_settings


SAMPLERS = ["euler", "euler_ancestral"]
SCHEDULERS = ["normal", "simple"]


def examples(count=3, **values):
    return [{"id": i + 1, "meta": dict(steps=8, cfgScale=1, sampler="Euler",
                                     scheduler="normal", **values)} for i in range(count)]


class RecommendationTests(unittest.TestCase):
    def test_moss_family_fills_missing_scheduler_without_replacing_checkpoint_recipe(self):
        from modules.performance import PerformanceSettings

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model.json").write_text(json.dumps({"baseModel": "MOSSImage",
                "generationConfig": {"steps": 50, "cfgScale": 3, "sampler_name": "er_sde"}}),
                encoding="utf-8")
            performance = PerformanceSettings.__new__(PerformanceSettings)
            performance.performance_options = json.loads(
                PerformanceSettings.DEFAULT_PERFORMANCE_FILE.read_text(encoding="utf-8"))
            performance.model_recommendations = ModelRecommendations(root, root / "settings.json")
            # Registry-only dependency: this resolver must not need a GPU context.
            registry = SimpleNamespace(KSampler=SimpleNamespace(
                SAMPLERS=["er_sde"], SCHEDULERS=["simple", "karras"]))
            with patch.dict(sys.modules, {"comfy.samplers": registry}):
                values = performance.get_perf_options("Model Defaults", "model.safetensors")
            self.assertEqual((values["custom_steps"], values["cfg"], values["sampler_name"], values["scheduler"]),
                             (50, 3, "er_sde", "simple"))

    def _performance(self, root):
        from modules.performance import PerformanceSettings
        performance = PerformanceSettings.__new__(PerformanceSettings)
        performance.performance_options = json.loads(
            PerformanceSettings.DEFAULT_PERFORMANCE_FILE.read_text(encoding="utf-8"))
        performance.model_recommendations = ModelRecommendations(root, root / "settings.json")
        return performance

    def test_distilled_families_use_native_template_fallbacks(self):
        registry = SimpleNamespace(KSampler=SimpleNamespace(SAMPLERS=["euler", "dpmpp_2m_sde_gpu"],
                                                            SCHEDULERS=["simple", "karras"]))
        cases = [("Krea 2", "desire_krea2.safetensors", (8, 1)),
                 ("MageFlow", "mage_flow.safetensors", (30, 5)),
                 ("MageFlow", "mage_flow_edit_turbo_int8.safetensors", (4, 1))]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            performance = self._performance(root)
            for family, checkpoint, expected in cases:
                with self.subTest(checkpoint=checkpoint):
                    (root / (Path(checkpoint).stem + ".json")).write_text(
                        json.dumps({"baseModel": family}), encoding="utf-8")
                    with patch.dict(sys.modules, {"comfy.samplers": registry}):
                        values = performance.get_perf_options("Model Defaults", checkpoint)
                    self.assertEqual((values["custom_steps"], values["cfg"], values["sampler_name"],
                                      values["scheduler"]), (*expected, "euler", "simple"))

    def test_generation_fetches_published_recipe_once_without_ui(self):
        registry = SimpleNamespace(KSampler=SimpleNamespace(SAMPLERS=SAMPLERS, SCHEDULERS=SCHEDULERS))
        version = {"id": 7, "modelId": 3, "baseModel": "Krea 2", "images": examples(4)}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model.json").write_text(json.dumps({"id": 7, "modelId": 3, "baseModel": "Krea 2"}),
                                             encoding="utf-8")
            performance = self._performance(root)
            with patch.dict(sys.modules, {"comfy.samplers": registry}), patch(
                    "modules.model_sources.civitai_metadata",
                    side_effect=lambda path: version if path == "model-versions/7" else None) as fetch:
                first = performance.get_perf_options("Model Defaults", "model.safetensors")
                calls = fetch.call_count
                performance.get_perf_options("Model Defaults", "model.safetensors")
            self.assertEqual((first["custom_steps"], first["cfg"], first["sampler_name"]), (8, 1, "euler"))
            self.assertGreater(calls, 0)
            self.assertEqual(fetch.call_count, calls)  # Cached afterwards.

    def test_majority_and_conflicting_fields(self):
        images = examples(10)
        for image in images[-3:]:
            image["meta"]["scheduler"] = "simple"
        self.assertEqual(example_settings({"images": images}, SAMPLERS, SCHEDULERS),
                         dict(custom_steps=8, cfg=1, sampler_name="euler", scheduler="normal"))
        for image in images[5:]:
            image["meta"]["scheduler"] = "simple"
        self.assertNotIn("scheduler", example_settings({"images": images}, SAMPLERS, SCHEDULERS))

    def test_sparse_duplicate_and_invalid_examples(self):
        for images in (examples(2), [examples(1)[0]] * 10,
                       [{"meta": {"steps": True, "cfgScale": float("nan"),
                                  "sampler": "unknown", "scheduler": []}}] * 5):
            self.assertEqual(example_settings({"images": images}, SAMPLERS, SCHEDULERS), {})
        self.assertEqual(example_settings({"images": "invalid"}, SAMPLERS, SCHEDULERS), {})

    def test_refresh_migrates_cache_preserves_override_and_card_priority(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model.json").write_text(json.dumps({"id": 123}), encoding="utf-8")
            store = ModelRecommendations(root, root / "settings.json")
            store.save_override("model.safetensors", dict(custom_steps=12, cfg=2,
                                sampler_name="euler", scheduler="simple", clip_skip=1),
                                SAMPLERS, SCHEDULERS)
            data = store.load()
            data["model.safetensors:123"].update(schema=2, checked=time.time(),
                status="Public version metadata refreshed.", recommended={})
            store.path.write_text(json.dumps(data), encoding="utf-8")
            remote = dict(id=123, modelId=456, baseModel="Example", generationConfig={"steps": 10},
                          images=examples())
            card = dict(id=456, modelVersions=[remote], generationConfig={"cfgScale": 7, "steps": 20})
            with patch("modules.model_sources.civitai_metadata", side_effect=[remote, card]) as fetch:
                store.refresh("model.safetensors", SAMPLERS, SCHEDULERS)
                self.assertEqual(fetch.call_count, 2)
            details = store.details("model.safetensors", SAMPLERS, SCHEDULERS)
            self.assertEqual(details["recommended"]["custom_steps"], 10)
            self.assertEqual(details["recommended"]["cfg"], 7)
            self.assertEqual(details["override"]["cfg"], 2)
            self.assertIn("example consensus", details["source"])
            store.reset_override("model.safetensors")
            self.assertEqual(store.recipe("model.safetensors", SAMPLERS, SCHEDULERS)[0]["cfg"], 7)

    def test_display_sampler_alias(self):
        images = examples()
        for image in images:
            image['meta']['sampler'] = 'Euler a'
        self.assertEqual(example_settings({'images': images}, SAMPLERS, SCHEDULERS)['sampler_name'],
                         'euler_ancestral')

    def test_wrong_version_and_offline_do_not_infer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model.json").write_text('{"id":123}', encoding="utf-8")
            store = ModelRecommendations(root, root / "settings.json")
            with patch("modules.model_sources.civitai_metadata", return_value=dict(id=999, images=examples())):
                store.refresh("model.safetensors", SAMPLERS, SCHEDULERS)
            self.assertEqual(store.details("model.safetensors", SAMPLERS, SCHEDULERS)["recommended"], {})
            store.offline = True
            with patch("modules.model_sources.civitai_metadata") as fetch:
                store.refresh("model.safetensors", SAMPLERS, SCHEDULERS, force=True)
                fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
