import json
import shutil
from pathlib import Path
from modules.config_io import save_json


class ResolutionSettings:
    DEFAULT_RESOLUTIONS_FILE = Path("settings/resolutions.default")
    VIDEO_RESOLUTIONS_FILE = Path("settings/video_resolutions.default")
    RESOLUTIONS_FILE = Path("settings/resolutions.json")
    CUSTOM_RESOLUTION = "Custom..."

    def __init__(self):
        self.load_resolutions()

    def load_resolutions(self):
        self.base_ratios = {}

        if not self.RESOLUTIONS_FILE.is_file():
            shutil.copy(self.DEFAULT_RESOLUTIONS_FILE, self.RESOLUTIONS_FILE)

        with self.RESOLUTIONS_FILE.open() as f:
            data = json.load(f)

        # Append new defaults without replacing the user's saved resolutions.
        with self.DEFAULT_RESOLUTIONS_FILE.open() as f:
            for ratio, res in json.load(f).items():
                data.setdefault(ratio, res)

        with self.VIDEO_RESOLUTIONS_FILE.open() as f:
            for ratio, res in json.load(f).items():
                # Older installations stored these video entries in the main list.
                data.setdefault(ratio, res)
                data[ratio].setdefault("video_models", res["video_models"])
        self.video_models = {name: res.get("video_models", []) for name, res in data.items()}
        self.image_models = {name: res.get("image_models", []) for name, res in data.items()}
        with self.DEFAULT_RESOLUTIONS_FILE.open() as f:
            self.builtin_names = set(json.load(f))
        with self.VIDEO_RESOLUTIONS_FILE.open() as f:
            self.builtin_names.update(json.load(f))

        # Square, landscape, portrait; then ratio and pixel area within each group.
        def order(item):
            name, res = item
            width, height = res["width"], res["height"]
            return (0 if width == height else 1 if width > height else 2,
                    max(width, height) / min(width, height), width * height, name.casefold())
        for ratio, res in sorted(data.items(), key=order):
            self.base_ratios[ratio] = (res["width"], res["height"])

        self.aspect_ratios = {
            f"{v[0]}x{v[1]} ({k})": v for k, v in self.base_ratios.items()
        }

        return self.base_ratios

    def save_resolutions(self, res_options):
        formatted_options = {}
        for k in res_options:
            formatted_options[k] = {
                "width": res_options[k][0],
                "height": res_options[k][1],
            }
            if self.image_models.get(k):
                formatted_options[k]["image_models"] = self.image_models[k]
            if self.video_models.get(k):
                formatted_options[k]["video_models"] = self.video_models[k]

        save_json(self.RESOLUTIONS_FILE, formatted_options)

        return self.load_resolutions()

    def get_base_aspect_ratios(self, name):
        return self.base_ratios[name]

    def choices_for_model(self, family):
        from modules.video_settings import VIDEO_FPS
        video = family in VIDEO_FPS
        choices = [f"{width}x{height} ({name})"
                for name, (width, height) in self.base_ratios.items()
                if (family in self.video_models[name] if video else
                    not self.video_models[name] and
                    (not self.image_models[name] or family in self.image_models[name]) and
                    (not name.startswith("Qwen Image") or family in ("QwenImage", "Qwen", "Qwen-Image")))
                ]
        recommended = self.recommended_sizes(family)
        for width, height in recommended:
            if not any(self.aspect_ratios[value] == (width, height) for value in choices):
                label = f"{width}x{height} ({family} recommended)"
                self.aspect_ratios[label] = (width, height)
                choices.append(label)
        choices.sort(key=lambda value: self.aspect_ratios[value] not in recommended)
        return choices + [self.CUSTOM_RESOLUTION]

    def recommended_sizes(self, family):
        import re
        from modules.model_recommendations import FAMILY_PRESETS
        with Path("settings/performance.default").open() as f:
            recipes = json.load(f)
        recipe = recipes.get(FAMILY_PRESETS.get(family, family), {})
        if not recipe:
            recipe = next((value for value in recipes.values() if family in value.get("video_models", [])), {})
        return [(int(w), int(h)) for w, h in re.findall(r"(\d+)x(\d+)", recipe.get("recommended_resolutions", ""))]

    def recommendation_text(self, family):
        sizes = self.recommended_sizes(family)
        return "Recommended (family recipe): " + ", ".join(f"{w} × {h}" for w,h in sizes) if sizes else "No recommended size published in the local family recipe."

    def is_custom(self, selection):
        return any(f"{w}x{h} ({name})" == selection and name not in self.builtin_names
                   for name, (w, h) in self.base_ratios.items())

    def remove_custom(self, selection):
        name = next((k for k,v in self.base_ratios.items() if f"{v[0]}x{v[1]} ({k})" == selection), None)
        if name is None or name in self.builtin_names:
            raise ValueError("Only saved custom resolutions can be removed.")
        options = dict(self.base_ratios)
        del options[name]
        self.save_resolutions(options)

    def selection_for_model(self, family, current):
        choices = self.choices_for_model(family)
        return current if current in choices else choices[0]

    def get_aspect_ratios(self, name):
        return self.aspect_ratios[name]
