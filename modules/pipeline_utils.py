import latent_preview
import folder_paths
from comfy.cli_args import args, LatentPreviewMethod
from pathlib import Path
import logging
import comfy.sd
from comfy.model_patcher import ModelPatcher
from modules.gguf_loader import load_diffusion_model as load_gguf_model


def load_diffusion_model(filename, input_unet=None):
    """Preserve native patchers and precision; wrap only actual GGUF files."""
    if Path(filename).suffix.lower() == ".gguf":
        return load_gguf_model(filename)
    if isinstance(input_unet, ModelPatcher):
        return input_unet
    if input_unet is not None:
        return comfy.sd.load_diffusion_model_state_dict(input_unet)
    return comfy.sd.load_diffusion_model(filename, model_options={})


def clean_prompt_cond_caches():
    return {sign: {"text": None, "cache": None} for sign in ("+", "-", "switch")}


def get_previewer(device, latent_format):
    if args.preview_method == LatentPreviewMethod.NoPreviews:
        return None
    import shared
    folder_paths.add_model_folder_path('vae_approx',
        str(shared.path_manager.model_paths['vae_approx_path']), is_default=True)
    method = args.preview_method
    try:
        # The model worker is serial. Preserve our Auto preference for installed
        # approximate VAEs while letting ComfyUI own loading and device policy.
        decoder = latent_format.taesd_decoder_name
        if method == LatentPreviewMethod.Auto and decoder and any(
                name.startswith(decoder) for name in folder_paths.get_filename_list('vae_approx')):
            args.preview_method = LatentPreviewMethod.TAESD
        try:
            previewer = latent_preview.get_previewer(device, latent_format)
        except Exception:
            logging.exception('Could not load preview decoder; using latent RGB preview')
            args.preview_method = LatentPreviewMethod.Latent2RGB
            previewer = latent_preview.get_previewer(device, latent_format)
    finally:
        args.preview_method = method
    if previewer is not None:
        def preview_function(x0, step, total_steps):
            # Image models can use a temporal approximate VAE for one frame.
            if isinstance(previewer, latent_preview.TAEHVPreviewerImpl) and x0.ndim == 4:
                x0 = x0.unsqueeze(2)
            return previewer.decode_latent_to_preview(x0)
        previewer.preview = preview_function
    return previewer
