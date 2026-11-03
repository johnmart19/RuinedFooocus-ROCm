"""Recovered MiniMax encoder selection; keep upstream recovery mathematics intact."""
from pathlib import Path

ENCODER = "qwen3vl_8b_minimax_h3_recovered_int8_convrot.safetensors"
MANIFEST = "minimax_h3_recovered_8b_manifest.json"
PACKAGE_FILES = (MANIFEST, "ara.safetensors", "conditioning_adapter.safetensors", ENCODER)


def load_recovered_encoder(path_manager):
    from modules.vendor.minimax_text_encoders import _load_manifest, _load_recovered
    files = [Path(path_manager.get_folder_file_path("clip", f"minimax_recovered_8b/{name}"))
             for name in PACKAGE_FILES]
    directory = files[0].parent
    if any(path.parent != directory for path in files):
        raise RuntimeError("MiniMax recovered encoder components must share one minimax_recovered_8b directory.")
    package = _load_manifest(directory, MANIFEST)
    clip = _load_recovered(directory, package, "recovered_8b_int8_convrot")
    clip._rf_minimax_recovered = True
    return clip


def use_recovered_encoder(default_settings, has_image=False):
    # Explicit official encoder selections retain their existing behavior.
    return not has_image and default_settings.get(
        "minimax_recovered_encoder", "clip_qwen3vl_32b" not in default_settings
    )
