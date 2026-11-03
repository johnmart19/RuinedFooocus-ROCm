import json
import tempfile
import time
import sys
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.model_recommendations import ModelRecommendations, example_settings, extract


SAMPLERS = ["euler", "euler_ancestral"]
SCHEDULERS = ["normal", "simple"]


def examples(count=3, **values):
    return [{"id": i + 1, "meta": dict(steps=8, cfgScale=1, sampler="Euler",
                                     scheduler="normal", **values)} for i in range(count)]


class RecommendationTests(unittest.TestCase):
    def test_hidream_sdk_guidance_is_not_native_cfg_zero(self):
        for value,expected in ((0,1),(.5,1),(1,1),(5,5)):
            metadata=dict(baseModel='HiDreamO1',description=f'guidance_scale = {value},')
            self.assertEqual(extract(metadata,SAMPLERS,SCHEDULERS)['cfg'],expected)
            metadata['generationConfig']={'guidance_scale':value}
            self.assertEqual(extract(metadata,SAMPLERS,SCHEDULERS)['cfg'],expected)
        # Explicit native CFG and other families keep their documented values.
        self.assertEqual(extract(dict(baseModel='HiDreamO1',description='CFG: 0\nguidance_scale = 5,'),SAMPLERS,SCHEDULERS)['cfg'],0)
        self.assertEqual(extract(dict(baseModel='HiDreamO1',generationConfig={'cfgScale':0}),SAMPLERS,SCHEDULERS)['cfg'],0)
        self.assertEqual(extract(dict(baseModel='Krea2',description='guidance_scale = 0,'),SAMPLERS,SCHEDULERS)['cfg'],0)

    def test_hidream_hf_refresh_translates_sdk_guidance_and_preserves_override(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            metadata=dict(baseModel='HiDreamO1',hf_repo_id='HiDream-ai/example',_rf_local_sha256='a'*64)
            (root/'o1_dev.json').write_text(json.dumps(metadata))
            store=ModelRecommendations(root,root/'settings.json')
            store.path.write_text(json.dumps({store.identity('o1_dev.safetensors'):dict(schema=4,
                checked=time.time(),status='Public version metadata refreshed.',recommended=dict(cfg=0),override=dict(cfg=2))}))
            remote=dict(description='guidance_scale = 0,\nnum_inference_steps = 28,',source='Hugging Face example',
                        source_url='https://huggingface.co/HiDream-ai/example')
            with patch('modules.model_sources.huggingface_recommendations',return_value=remote) as fetch:
                store.refresh('o1_dev.safetensors',SAMPLERS,SCHEDULERS)
                fetch.assert_called_once()
            details=store.details('o1_dev.safetensors',SAMPLERS,SCHEDULERS)
            self.assertEqual(details['recommended'],dict(cfg=1,custom_steps=28))
            self.assertEqual(store.recipe('o1_dev.safetensors',SAMPLERS,SCHEDULERS)[0]['cfg'],2)

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
                 ("Krea2", "desire_krea2_backend.safetensors", (8, 1)),
                 ("MageFlow", "mage_flow.safetensors", (30, 5)),
                 ("MageFlow", "mage_flow_edit_turbo_int8.safetensors", (4, 1)),
                 ("QwenImage", "qwen_image_edit_fp8.safetensors", (20, 2.5)),
                 ("QwenImage", "qwen_image_fp8.safetensors", (50, 4))]
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

    def test_krea_backend_family_keeps_provider_and_saved_overrides(self):
        registry = SimpleNamespace(KSampler=SimpleNamespace(SAMPLERS=SAMPLERS, SCHEDULERS=SCHEDULERS))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'krea.json').write_text(json.dumps({'baseModel': 'Krea 2',
                'generationConfig': {'steps': 12, 'cfgScale': 1.5}}), encoding='utf-8')
            performance = self._performance(root)
            with patch.dict(sys.modules, {'comfy.samplers': registry}):
                values = performance.get_perf_options('Model Defaults', 'krea.safetensors', 'Krea2')
                self.assertEqual((values['custom_steps'], values['cfg']), (12, 1.5))
                performance.model_recommendations.save_override('krea.safetensors',
                    dict(custom_steps=10, cfg=2, sampler_name='euler', scheduler='normal', clip_skip=1),
                    SAMPLERS, SCHEDULERS)
                values = performance.get_perf_options('Model Defaults', 'krea.safetensors', 'Krea2')
                self.assertEqual((values['custom_steps'], values['cfg'], values['scheduler']), (10, 2, 'normal'))

    def test_qwen_edit_published_and_user_settings_override_native_fallback(self):
        registry = SimpleNamespace(KSampler=SimpleNamespace(SAMPLERS=SAMPLERS, SCHEDULERS=SCHEDULERS))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'qwen_image_edit.json').write_text(json.dumps({'baseModel': 'QwenImage',
                'generationConfig': {'steps': 32, 'cfgScale': 3.2}}))
            performance = self._performance(root)
            with patch.dict(sys.modules, {'comfy.samplers': registry}):
                values = performance.get_perf_options('Model Defaults', 'qwen_image_edit.safetensors')
                self.assertEqual((values['custom_steps'], values['cfg']), (32, 3.2))
                performance.model_recommendations.save_override('qwen_image_edit.safetensors',
                    dict(custom_steps=50, cfg=4, sampler_name='euler', scheduler='normal', clip_skip=1),
                    SAMPLERS, SCHEDULERS)
                values = performance.get_perf_options('Model Defaults', 'qwen_image_edit.safetensors')
                self.assertEqual((values['custom_steps'], values['cfg'], values['scheduler']), (50, 4, 'normal'))

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

    def test_lora_gallery_recipes_do_not_set_bare_checkpoint_defaults(self):
        accelerated = examples(6)
        for index, image in enumerate(accelerated):
            image['id'] += 100
            image['meta']['steps'] = 4
            image['meta']['additionalResources' if index % 2 else 'resources'] = [
                {'type': 'LoRA', 'name': 'Lightning.safetensors', 'strength': 1}]
        clean = examples(3)
        for image in clean:
            image['meta'].update(steps=30, cfgScale=4)
        self.assertEqual(example_settings({'images': accelerated + clean}, SAMPLERS, SCHEDULERS),
                         dict(custom_steps=30, cfg=4, sampler_name='euler', scheduler='normal'))
        self.assertEqual(example_settings({'images': accelerated}, SAMPLERS, SCHEDULERS), {})
        clean[0]['meta']['resources'] = [None, 1, {'type': 'checkpoint'}]
        self.assertEqual(example_settings({'images': clean}, SAMPLERS, SCHEDULERS)['custom_steps'], 30)

    def test_old_accelerated_consensus_is_refreshed_without_losing_override(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'model.json').write_text(json.dumps({'id': 123}), encoding='utf-8')
            store = ModelRecommendations(root, root / 'settings.json')
            store.path.write_text(json.dumps({'model.safetensors:123': dict(schema=3,
                checked=time.time(), status='Public version metadata refreshed.',
                recommended=dict(custom_steps=4, cfg=1), override=dict(clip_skip=2))}))
            images = examples(4)
            for image in images:
                image['meta']['additionalResources'] = [{'type': 'lora', 'name': 'Lightning'}]
            remote = dict(id=123, images=images)
            with patch('modules.model_sources.civitai_metadata', return_value=remote) as fetch:
                store.refresh('model.safetensors', SAMPLERS, SCHEDULERS)
                fetch.assert_called_once()
            details = store.details('model.safetensors', SAMPLERS, SCHEDULERS)
            self.assertEqual(details['recommended'], {})
            self.assertEqual(details['override'], {'clip_skip': 2})

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
