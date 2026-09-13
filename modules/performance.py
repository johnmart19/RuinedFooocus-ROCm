import json
from pathlib import Path
import shutil

class PerformanceSettings:
    DEFAULT_PERFORMANCE_FILE = Path("settings/performance.default")
    PERFORMANCE_FILE = Path("settings/performance.json")
    CUSTOM_PERFORMANCE = "Custom..."

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

    def __init__(self):
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
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_performance(self, perf_options):
        self._save_data(self.PERFORMANCE_FILE, self.legacy_options | perf_options)
        self.performance_options = self.load_performance()

    def get_perf_options(self, name):
        # Stale/deleted names must return usable settings, never a string.
        options = self.performance_options.get(name, self.legacy_options.get(name, {}))
        return self.default_settings | options

    def apply(self, gen_data):
        result = gen_data.copy()
        name = result.get("performance_selection")
        if name and name != self.CUSTOM_PERFORMANCE:
            options = self.get_perf_options(name)
            result.update({key: options[key] for key in self.default_settings})
        return result

    def describe(self, name):
        resolution = self.performance_options.get(name, {}).get("recommended_resolutions", "")
        return f"Recommended: {resolution}" if resolution else ""

    def describe_workflow(self, family, checkpoint, selection, steps, cfg, sampler, scheduler):
        from modules.video_settings import VIDEO_FPS, fixed_video_settings, is_ltx25_distilled
        values = dict(custom_steps=steps, cfg=cfg, sampler_name=sampler, scheduler=scheduler)
        if selection != self.CUSTOM_PERFORMANCE:
            values.update(self.get_perf_options(selection))
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
            text += "Custom exposes steps, CFG, sampler, scheduler and Clip Skip.\n\n"
            text += "**PowerUp → Cheat Code**: Img2Img, SDXL controls, upscaling or background removal. Supply an Input image. For adjustable image-to-image denoise, choose Custom → Img2img → Denoise.\n\n"
            text += "Select LoRAs in Models. A sampler does not automatically load its required LCM/DMD2 LoRA."
        return text

    def choices_for_model(self, model_base, checkpoint=None):
        from modules.video_settings import VIDEO_FPS, is_ltx25_distilled
        video = model_base in VIDEO_FPS
        return [name for name, options in self.performance_options.items()
                if (model_base in options.get("video_models", []) if video
                    else not options.get("video_models"))
                and (name != "LTX 2.5 Distilled" or checkpoint is None
                     or is_ltx25_distilled(model_base, checkpoint))] + [self.CUSTOM_PERFORMANCE]

    def selection_for_model(self, family, checkpoint, current):
        from modules.video_settings import VIDEO_FPS
        choices = self.choices_for_model(family, checkpoint)
        if family in VIDEO_FPS:
            return choices[0]
        return current if current in choices else self.CUSTOM_PERFORMANCE
