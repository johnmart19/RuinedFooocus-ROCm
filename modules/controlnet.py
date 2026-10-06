import os
import shutil
import json
from os.path import exists
from shared import state
from shared import path_manager
from modules.config_io import save_json

DEFAULT_CNSETTINGS_FILE = "settings/powerup.default"
CNSETTINGS_FILE = "settings/powerup.json"
NEWCN = "Custom..."

# https://huggingface.co/stabilityai/control-lora/tree/main/control-LoRAs-rank128
controlnet_models = {
    "canny": "control-lora-canny-rank128.safetensors",
    "depth": "control-lora-depth-rank128.safetensors",
    "recolour": "control-lora-recolor-rank128.safetensors",
    "sketch": "control-lora-sketch-rank128-metadata.safetensors",
    "img2img": None,
    "upscale": None,
    "rembg": None,
    "faceswap": None,
}

def load_cnsettings():
    settings = {}
    if not os.path.isfile(CNSETTINGS_FILE):
        shutil.copy(DEFAULT_CNSETTINGS_FILE, CNSETTINGS_FILE)

    if os.path.exists(CNSETTINGS_FILE):
        with open(CNSETTINGS_FILE) as f:
            settings.update(json.load(f))

    with open(DEFAULT_CNSETTINGS_FILE) as f:
        default_settings = json.load(f)

    # Update settings with any missing keys from default_settings
    settings_updated = False
    for key, value in default_settings.items():
        if key not in settings:
            settings[key] = value
            settings_updated = True

    # If settings were updated, write them back to the file
    if settings_updated:
        save_json(CNSETTINGS_FILE, settings)

    # Ignore unknown settings
    for key in list(settings.keys()):
        if settings[key]["type"] not in controlnet_models:
            del settings[key]

    return settings


def save_cnsettings(cn_save_options):
    global controlnet_settings, cn_options
    save_json(CNSETTINGS_FILE, cn_save_options)
    controlnet_settings = cn_save_options
    cn_options = {f"{k}": v for k, v in controlnet_settings.items()}


def modes():
    return controlnet_settings.keys()


def get_model(type):
    if type not in controlnet_models or controlnet_models[type] is None:
        return None
    return path_manager.get_file_path(f"cn_{type}", default=None)


def get_settings(gen_data):
    if gen_data.get("automatic_upscale_target") == "Input image only":
        from modules.automatic_upscale import model_for
        model = model_for(gen_data.get("automatic_upscale"))
        if model is None:
            raise ValueError("Choose an upscale model before upscaling the input image.")
        return {"type": "upscale", "upscaler": model}
    if gen_data.get("controlnet") is not None:
        return dict(gen_data["controlnet"])
    if "cn_selection" not in gen_data:
        return {}
    if gen_data["cn_selection"] == NEWCN:
        return {
            "type": gen_data["cn_type"].lower(),
            "edge_low": gen_data["cn_edge_low"],
            "edge_high": gen_data["cn_edge_high"],
            "start": gen_data["cn_start"],
            "stop": gen_data["cn_stop"],
            "strength": gen_data["cn_strength"],
            "upscaler": gen_data["cn_upscale"],
        }
    else:
        options = dict(controlnet_settings.get(gen_data["cn_selection"], {}))
        if options.get("type") == "img2img" and "cn_strength" in gen_data:
            options["denoise"] = gen_data["cn_strength"]
        elif options.get("type") in ("canny", "depth", "sketch", "recolour"):
            for key in ("strength", "start", "stop"):
                if "cn_" + key in gen_data:
                    options[key] = gen_data["cn_" + key]
        return options


controlnet_settings = load_cnsettings()
cn_options = {f"{k}": v for k, v in controlnet_settings.items()}
