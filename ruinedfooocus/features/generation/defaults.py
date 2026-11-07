"""Save deliberate startup choices without storing uploads or generation state."""

import copy
import math

from ruinedfooocus.core.settings_validation import validate_settings

DEFAULT_CONTROLS = {
    "base_model": "base_model_name",
    "performance": "performance_selection",
    "resolution": "aspect_ratios_selection",
    "prompt": "prompt",
    "negative_prompt": "negative",
    "style": "style_selection",
    "seed": "seed",
    "seed_random": "seed_random",
    "image_number": "image_number",
    "auto_negative_prompt": "auto_negative",
    "custom_width": "custom_width",
    "custom_height": "custom_height",
}
SAMPLING_KEYS = ("custom_steps", "cfg", "sampler_name", "scheduler", "clip_skip")


def startup_loras(settings):
    if "startup_loras" in settings:
        return [(entry["name"], entry["weight"]) for entry in settings["startup_loras"]]
    return [
        (settings[f"lora_{i}_model"], settings.get(f"lora_{i}_weight", 0.5))
        for i in range(1, 6)
        if settings.get(f"lora_{i}_model") not in (None, "", "None")
    ]


def generation_defaults(values):
    if values.get("preset_selection"):
        raise ValueError("Clear the generation preset before saving current controls as defaults.")
    result = {
        key: copy.deepcopy(values[control])
        for key, control in DEFAULT_CONTROLS.items()
        if control in values
    }
    # Forever is an action, not a safe startup batch count.
    result["image_number"] = max(1, int(result.get("image_number", 1)))
    loras = []
    for item in values.get("loras") or []:
        caption = item.get("caption") if isinstance(item, dict) else item[1]
        weight, name = caption.split(" - ", 1)
        weight = float(weight)
        if not math.isfinite(weight):
            raise ValueError("LoRA weights must be finite numbers.")
        if name and name != "None":
            loras.append({"name": name, "weight": weight})
    result["startup_loras"] = loras
    sampling = {key: values[key] for key in SAMPLING_KEYS if key in values}
    validate_settings(sampling)
    result["custom_sampling"] = sampling
    return validate_settings(result)
