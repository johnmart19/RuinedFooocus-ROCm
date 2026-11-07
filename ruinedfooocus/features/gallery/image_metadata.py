"""Read saved rendering settings without importing worker state or image handles."""

import json

RENDER_FIELDS = frozenset(
    {
        "prompt",
        "negative",
        "raw_prompt",
        "raw_negative",
        "style_provenance",
        "steps",
        "custom_steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "clip_skip",
        "width",
        "height",
        "custom_width",
        "custom_height",
        "aspect_ratios_selection",
        "seed",
        "base_model_name",
        "base_model_hash",
        "loras",
        "start_step",
        "denoise",
        "sampling_denoise",
        "controlnet",
        "inpaint_toggle",
        "inpaint_denoise",
        "inpaint_mask_grow",
        "rembg_force",
        "automatic_upscale",
        "automatic_upscale_target",
        "generation_width",
        "generation_height",
        "upscaler",
        "image_guidance",
        "image_geometry",
        "depth_preprocessing",
        "reference_image_count",
        "sampling_shift_supported",
        "comment",
        "software",
        "video_duration",
        "video_fps",
        "frames",
        "fps",
    }
)


def read_metadata(value):
    if isinstance(value, str):
        if not value.lstrip().startswith("{"):
            return None
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        return None
    metadata = {key.lower(): item for key, item in value.items()}
    return metadata if "prompt" in metadata else None


def render_values(metadata):
    from ruinedfooocus.features.image_editing.image_guidance import FIELD_SCHEMA

    return {
        key: value
        for key, value in metadata.items()
        if key in RENDER_FIELDS or key in FIELD_SCHEMA or key.startswith(("cn_", "effective_"))
    }


def control_value(key, value):
    if key == "hidream_o1_seam_smoothing":
        if value is None:
            return "auto"
        if value is True:
            return "on"
        if value is False:
            return "off"
    return value
