"""LTX 2.5 distilled spatial upscaling and audiovisual refinement."""

from pathlib import Path
import numpy as np
import torch
import folder_paths
from comfy import model_management
from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader
from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler
from comfy_extras.nodes_lt import LTXVSeparateAVLatent, LTXVConcatAVLatent, LTXVImgToVideoInplace
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, KSamplerSelect
from modules import async_worker as worker
from modules.video_preview import video_callback


UPSCALER = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"


def refine(latent, guider, vae, upscaler_path, gen_data, seed):
    worker.check_interrupt(gen_data)
    worker.add_result(gen_data["task_id"], "preview", (-1, "Upscaling video latents ...", None))
    model_management.unload_model_and_clones(guider.model_patcher)
    path = Path(upscaler_path)
    folder_paths.add_model_folder_path("latent_upscale_models", str(path.parent), is_default=True)
    upscaler = LatentUpscaleModelLoader.execute(path.name)[0]
    try:
        if latent["samples"].is_nested:
            video, audio = LTXVSeparateAVLatent.execute(latent)
        else:
            video, audio = latent, None
        video = LTXVLatentUpsampler.execute(video, upscaler, vae)[0]
    finally:
        model_management.unload_model_and_clones(upscaler)
    worker.check_interrupt(gen_data)
    if gen_data["input_image"] is not None:
        image = torch.from_numpy(np.asarray(gen_data["input_image"], dtype=np.float32) / 255.0)[None]
        video = LTXVImgToVideoInplace.execute(vae, image, video, strength=1.0)[0]
    latent = LTXVConcatAVLatent.execute(video, audio)[0] if audio is not None else video
    worker.check_interrupt(gen_data)
    noise = Noise_RandomNoise(seed + 1)
    samples = guider.sample(
        noise.generate_noise(latent), latent["samples"],
        KSamplerSelect().get_sampler("euler")[0],
        torch.tensor([0.909375, 0.725, 0.421875, 0.0]),
        denoise_mask=latent.get("noise_mask"),
        callback=video_callback(guider.model_patcher, gen_data, "Refining video (stage 2/2)"),
        disable_pbar=False, seed=noise.seed,
    )
    return {**latent, "samples": samples.to(model_management.intermediate_device())}
