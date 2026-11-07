"""Depth ControlNet preparation through the pinned native geometry nodes."""

import math

DEPTH_MODELS = {size: f"depth_anything_3_{size}.safetensors" for size in ("small", "base")}
DEPTH_DEFAULTS = dict(
    depth_preprocess="auto",
    depth_model="auto",
    depth_resolution=504,
    depth_resize_method="upper_bound_resize",
    depth_normalization="v2_style",
    depth_dtype="default",
)
DEPTH_CHOICES = dict(
    depth_preprocess=("auto", "none"),
    depth_model=("auto", "small", "base"),
    depth_resize_method=("upper_bound_resize", "lower_bound_resize"),
    depth_normalization=("v2_style", "min_max"),
    depth_dtype=("default", "fp16", "bf16", "fp32"),
)


def depth_settings(values):
    # Old presets/metadata supplied a depth map directly. Only new defaults opt in.
    options = {
        key: values.get(key, "none" if key == "depth_preprocess" else default)
        for key, default in DEPTH_DEFAULTS.items()
    }
    for key, choices in DEPTH_CHOICES.items():
        if options[key] not in choices:
            raise ValueError(f"{key}: choose a supported depth option.")
    resolution = options["depth_resolution"]
    if isinstance(resolution, bool):
        raise ValueError("Depth resolution must be a multiple of 14 from 140 to 2520.")
    try:
        resolution = float(resolution)
    except (TypeError, ValueError):
        raise ValueError("Depth resolution must be a multiple of 14 from 140 to 2520.") from None
    if not math.isfinite(resolution) or not 140 <= resolution <= 2520 or resolution % 14:
        raise ValueError("Depth resolution must be a multiple of 14 from 140 to 2520.")
    options["depth_resolution"] = int(resolution)
    return options


def depth_model_name(selection):
    if selection == "auto":
        from comfy import model_management as memory

        device = memory.get_torch_device()
        selection = (
            "base"
            if device.type != "cpu" and memory.get_total_memory(device) >= 8 * 1024**3
            else "small"
        )
    return DEPTH_MODELS[selection]


def prepare_depth(owner, image, options):
    options = depth_settings(options)
    if options["depth_preprocess"] == "none":
        return image
    import folder_paths
    from pathlib import Path
    from ruinedfooocus.app.state import path_manager
    from ruinedfooocus.features.models.model_identity import file_signature
    from comfy_extras.nodes_depth_anything_3 import LoadDA3Model, DA3Inference, DA3Render

    model_name = depth_model_name(options["depth_model"])
    path = path_manager.get_folder_file_path("geometry_estimation", model_name)
    if path is None:
        raise FileNotFoundError(f"Depth preprocessing requires {model_name}.")
    path = Path(path)
    signature = (file_signature(path), options["depth_dtype"])
    if signature != getattr(owner, "depth_model_signature", None):
        folder_paths.add_model_folder_path("geometry_estimation", str(path.parent), is_default=True)
        model = LoadDA3Model.execute(model_name=path.name, weight_dtype=options["depth_dtype"])[0]
        owner.depth_model = model
        owner.depth_model_signature = signature
    owner.depth_model_name = model_name
    geometry = DA3Inference.execute(
        da3_model=owner.depth_model,
        image=image,
        resolution=options["depth_resolution"],
        resize_method=options["depth_resize_method"],
        mode={"mode": "mono"},
    )[0]
    depth = DA3Render.execute(
        da3_geometry=geometry,
        output=dict(
            output="depth", normalization=options["depth_normalization"], apply_sky_clip=False
        ),
    )[0]
    if tuple(depth.shape) != tuple(image.shape):
        raise ValueError(
            "Depth preprocessing changed the input dimensions; refusing to stretch it."
        )
    return depth
