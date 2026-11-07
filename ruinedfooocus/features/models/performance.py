from ruinedfooocus.core.paths import data_path, resource_path
import json
from pathlib import Path
import re
import shutil
from ruinedfooocus.core.config_io import save_json

MODEL_DEFAULT_SELECTIONS = ("Model Defaults", "Model settings", "Model defaults")

class PerformanceSettings:
    DEFAULT_PERFORMANCE_FILE = Path("ruinedfooocus/resources/defaults/performance.default")
    PERFORMANCE_FILE = data_path("settings", "performance.json")
    CUSTOM_PERFORMANCE = "Own settings"
    MODEL_PERFORMANCE = MODEL_DEFAULT_SELECTIONS[0]
    LEGACY_MODEL_PERFORMANCE = MODEL_DEFAULT_SELECTIONS[1]

    RETIRED_PRESETS = {
        "Speed", "Quality", "Lcm", "Lightning", "SD3", "Flux",
        "SDXL - Balanced", "Illustrious XL - Balanced", "Animagine XL 3.1",
        "NoobAI XL - EPS", "SDXL Lightning - 2 steps",
        "SDXL Lightning - 4 steps", "SDXL Lightning - 8 steps",
        "FLUX.1 dev - 20 steps", "SD3.5 Large", "SD3.5 Large Turbo",
    }

    default_settings = {
        "custom_steps": 30,
        "cfg": 8,
        "sampler_name": "dpmpp_2m_sde_gpu",
        "scheduler": "karras",
        "clip_skip": 1,
    }

    def __init__(self, cache_dir="cache/checkpoints", offline=False):
        from ruinedfooocus.features.models.model_recommendations import ModelRecommendations
        self.model_recommendations = ModelRecommendations(cache_dir=cache_dir, offline=offline)
        self.performance_options = self.load_performance()

    def load_performance(self):
        default_data = self._load_data(self.DEFAULT_PERFORMANCE_FILE)
        data = self._load_data(self.PERFORMANCE_FILE)

        # Retain old recipes for saved selections, but keep them out of the menu.
        self.legacy_options = {name: value for name, value in data.items()
                               if name in self.RETIRED_PRESETS}
        current = {name: value for name, value in data.items()
                   if name not in self.RETIRED_PRESETS}
        options = {name: defaults | current.get(name, {}) for name, defaults in default_data.items()}
        for name, defaults in default_data.items():
            options[name]["recommended_resolutions"] = defaults.get("recommended_resolutions", "")
        options.update({name: value for name, value in current.items() if name not in options})
        self._save_data(self.PERFORMANCE_FILE, self.legacy_options | options)
        return options

    def _load_data(self, file_path):
        if not file_path.exists():
            shutil.copy(self.DEFAULT_PERFORMANCE_FILE, file_path)
        with open(file_path, encoding="utf-8") as handle:
            return json.load(handle)

    def _save_data(self, file_path, data):
        save_json(file_path, data)

    def save_performance(self, perf_options):
        self._save_data(self.PERFORMANCE_FILE, self.legacy_options | perf_options)
        self.performance_options = self.load_performance()

    def get_perf_options(self, name, checkpoint=None, family=None):
        if name in MODEL_DEFAULT_SELECTIONS and checkpoint:
            from comfy.samplers import KSampler
            metadata = self.model_recommendations.metadata(checkpoint)
            family = family or metadata.get("baseModel", "Unknown")
            from ruinedfooocus.features.video.video_settings import VIDEO_FPS, is_ltx25_distilled
            if family in VIDEO_FPS:
                presets = [options for title, options in self.performance_options.items()
                           if family in options.get("video_models", [])
                           and (title != "LTX 2.5 Distilled" or is_ltx25_distilled(family, checkpoint))]
                return self.default_settings | (presets[0] if presets else {})
            preset = self.family_preset(family, checkpoint)
            fallback = self.performance_options.get(preset, {})
            # API and queued jobs may use a checkpoint never opened in the UI. Fetch its
            # published recipe once (cached, offline-aware); failures keep cached settings.
            try:
                self.model_recommendations.refresh(checkpoint, KSampler.SAMPLERS, KSampler.SCHEDULERS)
            except Exception as error:
                print(f"Model recommendation lookup failed: {error}")
            recipe, _ = self.model_recommendations.recipe(checkpoint, KSampler.SAMPLERS, KSampler.SCHEDULERS)
            return self.default_settings | fallback | recipe
        # Stale/deleted names must return usable settings, never a string.
        options = self.performance_options.get(name, self.legacy_options.get(name, {}))
        return self.default_settings | options

    def family_preset(self, family, checkpoint):
        from ruinedfooocus.features.models.model_recommendations import FAMILY_PRESETS
        preset = FAMILY_PRESETS.get(family, family)
        if preset == 'HiDream O1':
            from ruinedfooocus.features.generation.image_sampling import hidream_o1_dev
            return 'HiDream O1 Dev' if hidream_o1_dev(checkpoint) else preset
        # Upstream filenames distinguish edit and turbo sampling recipes.
        name = str(checkpoint or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
        if preset == "Qwen Image" and "edit" in name and "Qwen Image Edit" in self.performance_options:
            return "Qwen Image Edit"
        if "turbo" in name and f"{preset} Turbo" in self.performance_options:
            return f"{preset} Turbo"
        if preset == 'HiDream I1 Full':
            variant = re.search(r'(?:^|[_. -])(dev|fast)(?:[_. -]|$)', name)
            return 'HiDream I1 ' + (variant[1].capitalize() if variant else 'Full')
        return preset

    def apply(self, gen_data, family=None):
        result = gen_data.copy()
        result.pop('_model_default_sampling_shift', None)
        name = result.get("performance_selection")
        if name and name not in (self.CUSTOM_PERFORMANCE, "Custom..."):
            options = self.get_perf_options(name, result.get("base_model_name"), family)
            result.update({key: options[key] for key in self.default_settings})
            if name in MODEL_DEFAULT_SELECTIONS and options.get('sampling_shift') is not None:
                result['_model_default_sampling_shift'] = options['sampling_shift']
        return result

    def describe(self, name):
        resolution = self.performance_options.get(name, {}).get("recommended_resolutions", "")
        return f"Recommended: {resolution}" if resolution else ""

    def describe_workflow(self, family, checkpoint, selection, steps, cfg, sampler, scheduler):
        from ruinedfooocus.features.video.video_settings import VIDEO_FPS, fixed_video_settings, is_ltx25_distilled
        values = dict(custom_steps=steps, cfg=cfg, sampler_name=sampler, scheduler=scheduler)
        if selection != self.CUSTOM_PERFORMANCE:
            values.update(self.get_perf_options(selection, checkpoint, family))
        values.update(fixed_video_settings(family, checkpoint))
        schedule = values["scheduler"]
        if family in VIDEO_FPS and family.startswith("LTX"):
            schedule = "LTX model schedule"
        if is_ltx25_distilled(family, checkpoint):
            schedule = "Fixed 8-step schedule + 3-step refinement (2× upscale)"
        guidance = "Guidance" if family == "Hunyuan Video" else "CFG"
        text = (f"Steps: {values['custom_steps']} · {guidance}: {values['cfg']}  \n"
                f"Sampler: {values['sampler_name']} · Schedule: {schedule}\n\n")
        if family in VIDEO_FPS:
            text += "Set output size in Aspect Ratios and length in Video duration. Custom exposes adjustable sampling and FPS.\n\n"
            text += "For image-to-video, put the first frame in **PowerUp → Input image**. Leave it empty for text-to-video; use weights trained for that mode. Keep Cheat Code set to None."
            if family == "Wan Video":
                text += "\n\nWan 2.1 T2V and I2V require different checkpoints. Wan 2.2 14B dual-expert workflows are not implemented."
            if selection == self.CUSTOM_PERFORMANCE:
                text += "\n\nNo preset is active. Verify the recipe for your exact checkpoint; these values are not a model recommendation."
        else:
            options = self.get_perf_options(selection, checkpoint, family)
            text += f"Clip Skip: {options['clip_skip']}" if selection != self.CUSTOM_PERFORMANCE else ""
        return text

    def choices_for_model(self, model_base, checkpoint=None):
        return [self.MODEL_PERFORMANCE, self.CUSTOM_PERFORMANCE]

    def selection_for_model(self, family, checkpoint, current):
        return self.CUSTOM_PERFORMANCE if current in (self.CUSTOM_PERFORMANCE, "Custom...") else self.MODEL_PERFORMANCE
