"""Validate Settings form values before changing or writing configuration."""

import math


INTEGER_RANGES = {
    "image_number": (1, None), "image_number_max": (1, None),
    "seed": (-1, None), "preset_mode": (1, 7),
    "images_per_page": (1, 1000), "hint_chance": (0, 100),
    "llm_n_predict": (-1, None), "llm_n_ctx": (0, None),
    "llm_n_gpu_layers": (-1, None), "llm_chat_history": (0, None),
    "llm_hp_max_tokens": (1, None),
}
FLOAT_RANGES = {"video_fps": (0.01, None), "update_interval": (0.01, None)}
LIST_FIELDS = {"archive_folders", "path_checkpoints", "path_loras", "path_wildcards"}


def validate_settings(values):
    result = dict(values)
    for key, value in result.items():
        bounds = INTEGER_RANGES.get(key) or FLOAT_RANGES.get(key)
        if key.endswith("_shift"):
            bounds = (0.01, None)
        if (key.startswith("lora_") and key.endswith("_weight")) or key in ("lora_min", "lora_max"):
            bounds = (None, None)
        if bounds is not None and value in (None, "") and not key.endswith("_shift"):
            raise ValueError(f"{key}: enter a number.")
        if bounds is not None and value not in (None, ""):
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{key}: enter a number.") from None
            if not math.isfinite(number):
                raise ValueError(f"{key}: enter a finite number.")
            low, high = bounds
            if (low is not None and number < low) or (high is not None and number > high):
                raise ValueError(f"{key}: value must be between {low} and {high or 'no upper limit'}.")
            if key in INTEGER_RANGES:
                if not number.is_integer():
                    raise ValueError(f"{key}: enter a whole number.")
                number = int(number)
            result[key] = number
        if key in LIST_FIELDS:
            lines = value.splitlines() if isinstance(value, str) else (value or [])
            result[key] = [line.strip() for line in lines if line.strip()]
            if key in ("path_checkpoints", "path_loras") and not result[key]:
                raise ValueError(f"{key}: provide at least one folder.")
        if key in ("path_outputs", "path_inbox") and not str(value or "").strip():
            raise ValueError(f"{key}: provide a folder.")
    if result.get("image_number", 1) > result.get("image_number_max", 50):
        raise ValueError("Image Number cannot exceed Image Number Max.")
    if result.get("lora_min", 0) >= result.get("lora_max", 2):
        raise ValueError("LoRA weight min must be smaller than max.")
    name = result.get("ui_settings_name")
    if name and (name in (".", "..") or any(c in name for c in '\\/:*?"<>|')):
        raise ValueError("Settings profile name must be a folder name, without path separators.")
    return result
