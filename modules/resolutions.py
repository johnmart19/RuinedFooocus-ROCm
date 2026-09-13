import json
import shutil
from pathlib import Path


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
            if self.video_models.get(k):
                formatted_options[k]["video_models"] = self.video_models[k]

        with open(self.RESOLUTIONS_FILE, "w") as f:
            json.dump(formatted_options, f, indent=2)

        return self.load_resolutions()

    def get_base_aspect_ratios(self, name):
        return self.base_ratios[name]

    def choices_for_model(self, family):
        from modules.video_settings import VIDEO_FPS
        video = family in VIDEO_FPS
        return [f"{width}x{height} ({name})"
                for name, (width, height) in self.base_ratios.items()
                if (family in self.video_models[name] if video else not self.video_models[name])
                ] + [self.CUSTOM_RESOLUTION]

    def selection_for_model(self, family, current):
        choices = self.choices_for_model(family)
        return current if current in choices else choices[0]

    def get_aspect_ratios(self, name):
        return self.aspect_ratios[name]
