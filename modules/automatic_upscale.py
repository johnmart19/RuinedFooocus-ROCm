"""Optional model-based image enlargement after native VAE decoding."""

PRESETS = {
    "Off": (None, 1),
    "2× Anime (Futsuu)": ("2x-Futsuu-Anime.pth", 2),
    "2× Anime cleanup (AniScale)": ("2x-AniScale.pth", 2),
    "2× Anime sharp (AnimeSharp V4 Fast)": ("2x-AnimeSharpV4_Fast_RCAN_PU.safetensors", 2),
    "2× General (RealESRGAN)": ("RealESRGAN_x2plus.pth", 2),
    "4× Anime (RealESRGAN)": ("RealESRGAN_x4plus_anime_6B.pth", 4),
    "4× General (RealESRGAN)": ("RealESRGAN_x4plus.pth", 4),
    "4× UltraSharp": ("4x-UltraSharp.pth", 4),
}
# Saved selections from the first version remain valid without family guessing.
ALIASES = {"2×": "2× General (RealESRGAN)", "4×": "4× General (RealESRGAN)"}


def choices():
    from shared import path_manager
    names = set(path_manager.upscaler_filenames) | {
        name for name, entry in path_manager.DOWNLOADABLE_FILES.items()
        if entry["path"] == "path_upscalers"}
    preset_models = {model for model, _ in PRESETS.values()}
    return list(PRESETS) + sorted(name for name in names
        if name not in preset_models and filename_scale(name) is not None)


def filename_scale(name):
    import re
    match = re.search(r"(?:^|[-_])([2348])x|(?:^|[-_])x([2348])", name, re.IGNORECASE)
    return int(match.group(1) or match.group(2)) if match else None


def resolve(selection):
    selection = ALIASES.get(selection, selection) or "Off"
    if selection in PRESETS:
        return PRESETS[selection]
    if selection not in choices():
        raise ValueError("Choose a listed upscale model.")
    return selection, filename_scale(selection)


def scale_for(selection):
    return resolve(selection)[1]


def describe_size(selection, resolution, width, height, resolutions):
    scale = scale_for(selection)
    if resolution in resolutions:
        width, height = resolutions[resolution]
    width, height = int(width), int(height)
    return f"{width * scale} × {height * scale} pixels" + (
        f" (from {width} × {height}, {scale}× upscale)" if scale > 1 else "")


def model_for(selection, family=None):
    return resolve(selection)[0]


def upscale_decoded(owner, image, gen_data, family):
    gen_data.pop("automatic_upscale_model", None)
    name = model_for(gen_data.get("automatic_upscale"), family)
    if name is None:
        return image
    from modules.upscale_pipeline import pipeline, InterruptibleUpscaler
    from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
    import modules.async_worker as worker
    worker.check_interrupt(gen_data)
    worker.add_result(gen_data["task_id"], "preview", (-1, "Automatic upscaling ...", None))
    if getattr(owner, "automatic_upscaler_name", None) != name:
        owner.automatic_upscaler = pipeline().load_upscaler_model(name)
        owner.automatic_upscaler_name = name
    model = owner.automatic_upscaler
    if model.scale != scale_for(gen_data["automatic_upscale"]):
        raise ValueError(f"Unexpected scale for automatic upscaler {name}: {model.scale}")
    result = ImageUpscaleWithModel().upscale(InterruptibleUpscaler(model, gen_data), image)[0]
    worker.check_interrupt(gen_data)
    gen_data["automatic_upscale_model"] = name
    return result
