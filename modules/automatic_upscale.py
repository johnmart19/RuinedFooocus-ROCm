"""Optional model-based image enlargement after native VAE decoding."""

CHOICES = ["Off", "2×", "4×"]


def scale_for(selection):
    if selection in (None, "Off"):
        return 1
    if selection not in CHOICES:
        raise ValueError("Automatic upscale must be Off, 2× or 4×.")
    return 2 if selection == "2×" else 4


def describe_size(selection, resolution, width, height, resolutions):
    scale = scale_for(selection)
    if resolution in resolutions:
        width, height = resolutions[resolution]
    width, height = int(width), int(height)
    return f"Output: **{width * scale} × {height * scale}** pixels" + (
        f" · generated at {width} × {height}, then upscaled {scale}×" if scale > 1 else "")


def model_for(selection, family):
    scale = scale_for(selection)
    if scale == 1:
        return None
    if scale == 2:
        return "RealESRGAN_x2plus.pth"
    return ("RealESRGAN_x4plus_anime_6B.pth" if family in
            ("Anima", "Illustrious", "NoobAI", "Animagine") else "RealESRGAN_x4plus.pth")


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
