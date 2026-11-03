import json
from pathlib import Path
import shutil
from modules.config_io import save_json


class OneButtonPresets:
    DEFAULT_OBP_FILE = Path("random_prompt/presets/obp_presets.default")
    OBP_FILE = Path("random_prompt/userfiles/obp_presets.json")
    CUSTOM_OBP = "Custom..."
    RANDOM_PRESET_OBP = "All (random)..."

    def __init__(self):
        self.opb_presets = self.load_obp_presets()

    def load_obp_presets(self):
        try:
            default_data = self._load_data(self.DEFAULT_OBP_FILE)
        except:
            print(f"ERROR: Failed to load {self.DEFAULT_OBP_FILE}")
            default_data = {}
        try:
            data = self._load_data(self.OBP_FILE)
        except:
            print(f"ERROR: Failed to load {self.OBP_FILE}")
            data = {}

        for name, settings in default_data.items():
            if name not in data:
                data[name] = settings
            elif "enhancement_focus" in settings:
                data[name].setdefault("enhancement_focus", settings["enhancement_focus"])

        # Sanity check
        for name, settings in data.items():
            if settings['subject'] == '------ all':
                settings['subject'] = 'all'
            # Upgrade unchanged bundled anime text, preserving custom preset edits.
            if name in ("Waifu's", "Husbando's"):
                for key, old in {
                    "prefixprompt": "(((masterpiece))), (((best quality))), anime style, 2d,",
                    "suffixprompt": "key visual",
                }.items():
                    if settings.get(key) == old:
                        settings[key] = default_data[name][key]

        try:
            self._save_data(self.OBP_FILE, data)
        except:
            print(f"ERROR: Failed to save {self.OBP_FILE}")
        return data

    def _load_data(self, file_path):
        if not file_path.exists():
            shutil.copy(self.DEFAULT_OBP_FILE, file_path)
        with open(file_path, encoding="utf-8") as file:
            return json.load(file)

    def _save_data(self, file_path, data):
        save_json(file_path, data)

    def save_obp_preset(self, perf_options):
        self._save_data(self.OBP_FILE, perf_options)
        self.opb_presets = self.load_obp_presets()

    def get_obp_preset(self, name):
        return self.opb_presets[name]
