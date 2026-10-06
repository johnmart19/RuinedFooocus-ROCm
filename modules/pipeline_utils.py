import torch
import os
import einops
from latent_preview import Latent2RGBPreviewer
import latent_preview
from comfy.cli_args import args, LatentPreviewMethod
from pathlib import Path
import logging
import numpy as np


def clean_prompt_cond_caches():
    conditions = {}
    conditions["+"] = {}
    conditions["-"] = {}
    conditions["switch"] = {}
    conditions["+"]["text"] = None
    conditions["+"]["cache"] = None
    conditions["-"]["text"] = None
    conditions["-"]["cache"] = None
    conditions["switch"]["text"] = None
    conditions["switch"]["cache"] = None
    return conditions


def set_timestep_range(conditioning, start, end):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]

        if "pooled_output" in n[1]:
            n[1]["start_percent"] = start
            n[1]["end_percent"] = end

        c.append(n)

    return c


def get_previewer(device, latent_format):
    if args.preview_method == LatentPreviewMethod.NoPreviews:
        return None
    previewer = None
    # Prefer an installed approximate VAE in Auto mode. Linear RGB projection
    # alone is particularly coarse for image models with video VAE latents.
    if args.preview_method in (LatentPreviewMethod.Auto, LatentPreviewMethod.TAESD) and getattr(latent_format, 'taesd_decoder_name', None):
        import shared
        from comfy.sd import VAE
        from comfy.utils import load_torch_file
        from comfy.taesd.taesd import TAESD
        directory = Path(shared.path_manager.model_paths['vae_approx_path'])
        decoder = next((p for p in sorted(directory.glob(latent_format.taesd_decoder_name + '*'))
                        if p.suffix in ('.safetensors', '.pth', '.pt')), None)
        if decoder is not None:
            try:
                if latent_format.taesd_decoder_name in latent_preview.VIDEO_TAES:
                    vae = VAE(load_torch_file(str(decoder)))
                    vae.throw_exception_if_invalid()
                    vae.first_stage_model.show_progress_bar = False
                    previewer = latent_preview.TAEHVPreviewerImpl(vae)
                else:
                    vae = TAESD(None, str(decoder), latent_channels=latent_format.latent_channels).to(device)
                    previewer = latent_preview.TAESDPreviewerImpl(vae)
            except Exception:
                logging.exception('Could not load preview decoder %s; using latent RGB preview', decoder.name)
    if previewer is None and args.preview_method == LatentPreviewMethod.TAESD:
        previewer = latent_preview.get_previewer(device, latent_format)
    if previewer is not None:
        def preview_function(x0, step, total_steps):
            # Video approximate VAEs also preview still images: one frame.
            if isinstance(previewer, latent_preview.TAEHVPreviewerImpl) and x0.ndim == 4:
                x0 = x0.unsqueeze(2)
            return previewer.decode_latent_to_preview(x0)
        previewer.preview = preview_function
        return previewer
    if latent_format.latent_rgb_factors is None:
        return None
    previewer = Latent2RGBPreviewer(
        latent_rgb_factors=latent_format.latent_rgb_factors,
        latent_rgb_factors_bias=latent_format.latent_rgb_factors_bias,
        latent_rgb_factors_reshape=latent_format.latent_rgb_factors_reshape
    )
    def preview_function(x0, step, total_steps):
        return previewer.decode_latent_to_preview(x0)
    previewer.preview = preview_function
    return previewer
