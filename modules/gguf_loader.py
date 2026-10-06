"""Load local diffusion GGUFs through the backend's quantization-aware loader."""

from pathlib import Path

import folder_paths
from molbal_comfyui_gguf.nodes import UnetLoaderGGUF


def load_diffusion_model(filename):
    path = Path(filename).resolve(strict=True)
    folder_paths.add_model_folder_path("diffusion_models", str(path.parent), is_default=True)
    return UnetLoaderGGUF().load_unet(path.name, patch_on_device=True)[0]
