"""Independent official Comfy node references for LTX AV and MiniMax H3.

Entry point ``tools.comfy_video_reference:generate`` accepts explicit local split
components. family=ltx_av supports LTX 2/2.3/2.5 using caller-selected components;
family=minimax uses the official MiniMax node. Distilled validates stage one, or both stages with refinement_upscaler.
No model downloads, app pipelines, web server or graph executor are used.
"""
from tools.comfy_feature_reference import _register, _image


def generate(family, diffusion_model, text_encoder, vae_model, audio_vae_model,
             text_encoder2=None, positive="a landscape", negative="", seed=1,
             steps=20, cfg=4.0, sampler="euler", scheduler="normal", width=512,
             height=320, length=9, fps=24.0, source_image=None, distilled=False,
             weight_dtype="default", lora=None, lora_strength=1.0, output_video=None,
             refinement_upscaler=None, gguf_loader="upstream"):
    import torch
    from nodes import VAELoader, LoraLoader, CLIPTextEncode, VAEDecodeTiled
    from comfy_extras.nodes_custom_sampler import (RandomNoise, BasicScheduler,
        KSamplerSelect, BasicGuider, CFGGuider, SamplerCustomAdvanced)
    from comfy_extras.nodes_lt import (EmptyLTXVLatentVideo, LTXVImgToVideo,
        LTXVConditioning, LTXVConcatAVLatent, LTXVSeparateAVLatent, LTXVScheduler)
    from comfy_extras.nodes_lt_audio import LTXVEmptyLatentAudio
    from comfy_extras.nodes_audio import VAEDecodeAudio
    if family not in ("ltx_av", "minimax"):
        raise ValueError("family must be ltx_av or minimax")
    if family == "minimax" and (distilled or fps != 24.0):
        raise ValueError("MiniMax requires 24 fps and does not use LTX distilled scheduling")
    model = _diffusion(diffusion_model, weight_dtype, gguf_loader)
    clip_type = "minimax" if family == "minimax" else "ltxv"
    clip = _clip(text_encoder, text_encoder2, clip_type)
    vae = VAELoader().load_vae(_register("vae", vae_model))[0]
    audio_vae = VAELoader().load_vae(_register("vae", audio_vae_model))[0]
    if lora:
        model, clip = LoraLoader().load_lora(model, clip, _register("loras", lora), lora_strength, lora_strength)
    final_width, final_height = width, height
    if refinement_upscaler:
        if family != "ltx_av" or not distilled:
            raise ValueError("Spatial refinement requires distilled LTX AV")
        width, height = width // 2, height // 2
    if family == "minimax":
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
        pos, latent = MiniMaxH3ImageToVideo.execute(clip, vae, positive, width, height, length,
            first_frame=_image(source_image, width, height) if source_image else None).result
        neg = None
    else:
        pos = CLIPTextEncode().encode(clip, positive)[0]
        neg = CLIPTextEncode().encode(clip, negative)[0] if cfg != 1.0 else []
        if source_image:
            pos, neg, video = LTXVImgToVideo.execute(positive=pos, negative=neg, vae=vae,
                image=_image(source_image, final_width, final_height), width=width, height=height,
                length=length, batch_size=1, strength=1.0).result
        else:
            video = EmptyLTXVLatentVideo.execute(width, height, length, 1)[0]
        audio = LTXVEmptyLatentAudio.execute(audio_vae=audio_vae, frames_number=length, frame_rate=fps, batch_size=1)[0]
        latent = LTXVConcatAVLatent.execute(video, audio)[0]
        pos, neg = LTXVConditioning.execute(pos, neg, fps).result
    if distilled:
        sigmas = torch.tensor([1.0, .99375, .9875, .98125, .975, .909375, .725, .421875, 0.0])
        sampler = "euler_ancestral"
    elif family == "ltx_av":
        sigmas = LTXVScheduler.execute(steps=steps, max_shift=2.05, base_shift=.95,
            stretch=True, terminal=.1, latent=latent)[0]
    else:
        sigmas = BasicScheduler.execute(model, scheduler, steps, denoise=1.0)[0]
    guider = (BasicGuider.execute(model, pos)[0] if family == "minimax"
              else CFGGuider.execute(model, pos, neg, cfg)[0])
    result = SamplerCustomAdvanced.execute(RandomNoise.execute(seed)[0], guider,
        KSamplerSelect.execute(sampler)[0], sigmas, latent)[0]
    if refinement_upscaler:
        from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader
        from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler
        from comfy_extras.nodes_lt import LTXVImgToVideoInplace
        import comfy.model_management
        video, audio = LTXVSeparateAVLatent.execute(result).result
        upscaler = LatentUpscaleModelLoader.execute(_register("latent_upscale_models", refinement_upscaler))[0]
        try:
            video = LTXVLatentUpsampler.execute(video, upscaler, vae)[0]
        finally:
            comfy.model_management.unload_model_and_clones(upscaler)
        if source_image:
            video = LTXVImgToVideoInplace.execute(vae,
                _image(source_image, final_width, final_height), video, strength=1.0)[0]
        latent = LTXVConcatAVLatent.execute(video, audio)[0]
        sigmas = torch.tensor([.85, .7250, .4219, 0.0])
        result = SamplerCustomAdvanced.execute(RandomNoise.execute(seed + 1)[0], guider,
            KSamplerSelect.execute("euler_ancestral")[0], sigmas, latent)[0]
    width, height = final_width, final_height
    if family == "ltx_av":
        video, audio = LTXVSeparateAVLatent.execute(result).result
    else:
        parts = result["samples"].tensors
        video = {"samples": parts[0]}
        audio = {"samples": parts[1]} if len(parts) > 1 else None
    frames = VAEDecodeTiled().decode(vae=vae, samples=video, tile_size=512, overlap=64,
                                    temporal_size=64, temporal_overlap=16)[0]
    sound = VAEDecodeAudio.execute(audio_vae, audio)[0] if audio else None
    if output_video:
        _save_video(frames, sound, fps, output_video)
    return {"latent": {"video": video["samples"], "audio": audio["samples"] if audio else None}, "conditioning": {"positive": pos, "negative": neg},
            "sigmas": sigmas, "output": frames, "audio": sound,
            "settings": {"seed": seed, "steps": 3 if refinement_upscaler else (8 if distilled else steps),
            "cfg": 1.0 if family == "minimax" else cfg, "sampler": sampler,
            "scheduler": ("LTXDistilledRefinement" if refinement_upscaler else ("LTXDistilled" if distilled else "LTXVScheduler")) if family == "ltx_av" else scheduler,
            "width": width, "height": height, "length": int(frames.shape[0]), "fps": fps,
            "lora_strength": lora_strength if lora else None}}


def legacy(family, diffusion_model, text_encoder, vae_model, text_encoder2=None,
           positive="a landscape", negative="", seed=1, steps=20, cfg=4.0,
           sampler="euler", scheduler="normal", width=512, height=320, length=9,
           fps=24.0, source_image=None, weight_dtype="default", clip_vision_model=None,
           sampling_shift=None, lora=None, lora_strength=1.0, output_video=None,
           gguf_loader="upstream"):
    """Official-node legacy LTX 0.9.6, Wan 2.1/2.2 5B, and Hunyuan recipes.

    Wan 2.2 dual-expert 14B is excluded. Components are split local files; the
    caller explicitly chooses checkpoints/encoders appropriate to t2v versus i2v.
    """
    from nodes import VAELoader, CLIPTextEncode, LoraLoader, VAEDecodeTiled
    from comfy_extras.nodes_custom_sampler import (RandomNoise, BasicScheduler,
        KSamplerSelect, BasicGuider, CFGGuider, SamplerCustomAdvanced)
    from comfy_extras.nodes_hunyuan import EmptyHunyuanLatentVideo, HunyuanImageToVideo
    from comfy_extras.nodes_lt import EmptyLTXVLatentVideo, LTXVImgToVideo, LTXVConditioning, LTXVScheduler
    from comfy_extras.nodes_model_advanced import ModelSamplingSD3
    types = {"ltx_legacy": "ltxv", "wan21": "wan", "wan22_5b": "wan", "hunyuan": "hunyuan_video"}
    if family not in types:
        raise ValueError("family must be ltx_legacy, wan21, wan22_5b, or hunyuan")
    model = _diffusion(diffusion_model, weight_dtype, gguf_loader)
    clip = _clip(text_encoder, text_encoder2, types[family])
    vae = VAELoader().load_vae(_register("vae", vae_model))[0]
    if lora:
        model, clip = LoraLoader().load_lora(model, clip, _register("loras", lora), lora_strength, lora_strength)
    pos = None if family == 'hunyuan' and source_image else CLIPTextEncode().encode(clip, positive)[0]
    neg = CLIPTextEncode().encode(clip, negative)[0]
    image = _image(source_image, width, height) if source_image else None
    if family == "ltx_legacy":
        if image is not None:
            pos, neg, latent = LTXVImgToVideo.execute(pos, neg, image, vae, width, height, length, 1, 1.0).result
        else:
            latent = EmptyLTXVLatentVideo.execute(width, height, length, 1)[0]
        pos, neg = LTXVConditioning.execute(pos, neg, fps).result
    elif family == "wan22_5b":
        from comfy_extras.nodes_wan import Wan22ImageToVideoLatent
        latent = Wan22ImageToVideoLatent.execute(vae, width, height, length, 1, start_image=image)[0]
    elif image is not None and family == "wan21":
        from comfy_extras.nodes_wan import WanImageToVideo
        vision = None
        if clip_vision_model:
            from nodes import CLIPVisionLoader, CLIPVisionEncode
            encoder = CLIPVisionLoader().load_clip(_register("clip_vision", clip_vision_model))[0]
            vision = CLIPVisionEncode().encode(encoder, image, crop="center")[0]
        pos, neg, latent = WanImageToVideo.execute(pos, neg, vae, width, height, length, 1,
                                                  clip_vision_output=vision, start_image=image).result
    elif image is not None and family == "hunyuan":
        if not clip_vision_model:
            raise ValueError('Hunyuan I2V requires the LLaVA vision encoder')
        from nodes import CLIPVisionLoader, CLIPVisionEncode
        from comfy_extras.nodes_hunyuan import TextEncodeHunyuanVideo_ImageToVideo
        encoder = CLIPVisionLoader().load_clip(_register("clip_vision", clip_vision_model))[0]
        vision = CLIPVisionEncode().encode(encoder, image, crop="none")[0]
        pos = TextEncodeHunyuanVideo_ImageToVideo.execute(clip, vision, positive, image_interleave=4)[0]
        pos, latent = HunyuanImageToVideo.execute(pos, vae, width, height, length, 1,
                                                   guidance_type="v2 (replace)", start_image=image).result
    else:
        latent = EmptyHunyuanLatentVideo.execute(width, height, length, 1)[0]
    if family != "ltx_legacy":
        shift = sampling_shift if sampling_shift is not None else (7.0 if family == "hunyuan" else 8.0)
        model = ModelSamplingSD3().patch(model, shift)[0]
    if family == "hunyuan":
        from comfy_extras.nodes_flux import FluxGuidance
        pos = FluxGuidance.execute(pos, cfg)[0]
        guider = BasicGuider.execute(model, pos)[0]
    else:
        guider = CFGGuider.execute(model, pos, neg, cfg)[0]
    if family in ("wan21", "wan22_5b"):
        # Stock Wan workflows use KSampler, which adjusts schedules for uni_pc
        # and other discard-penultimate samplers; BasicScheduler is not equivalent.
        result, sigmas = _wan_sample(model, seed, steps, cfg, sampler, scheduler, pos, neg, latent)
    else:
        sigmas = (LTXVScheduler.execute(steps, 2.05, .95, True, .1, latent)[0] if family == "ltx_legacy"
                  else BasicScheduler.execute(model, scheduler, steps, 1.0)[0])
        result = SamplerCustomAdvanced.execute(RandomNoise.execute(seed)[0], guider,
            KSamplerSelect.execute(sampler)[0], sigmas, latent)[0]
    if family == "hunyuan":
        neg = None  # BasicGuider only consumes positive conditioning.
    if family == "ltx_legacy":
        frames = VAEDecodeTiled().decode(vae, result, tile_size=512, overlap=64,
                                        temporal_size=64, temporal_overlap=5)[0]
    else:
        frames = VAEDecodeTiled().decode(vae, result,
            tile_size=128 if family == "hunyuan" else 512, overlap=64)[0]
    if output_video:
        _save_video(frames, None, fps, output_video)
    return {"latent": result["samples"], "conditioning": _conditioning_artifact({"positive": pos, "negative": neg}),
            "sigmas": sigmas, "output": frames, "settings": {"family": family, "seed": seed,
            "steps": steps, "cfg": cfg, "sampler": sampler, "scheduler": scheduler,
            "width": width, "height": height, "length": int(frames.shape[0]), "fps": fps,
            "sampling_shift": sampling_shift, "lora_strength": lora_strength if lora else None}}


def _save_video(frames, audio, fps, filename):
    from pathlib import Path
    from comfy_extras.nodes_video import CreateVideo
    from comfy_api.latest import Types
    target = Path(filename).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    video = CreateVideo.execute(images=frames, audio=audio, fps=fps)[0]
    video.save_to(target, format=Types.VideoContainer.MP4, codec=Types.VideoCodec("auto"))


def _conditioning_artifact(value):
    """Expose vision output tensors without dropping conditioning evidence."""
    from comfy.clip_vision import Output
    if isinstance(value, Output):
        return {key: _conditioning_artifact(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: _conditioning_artifact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_conditioning_artifact(item) for item in value]
    return value


def _diffusion(filename, weight_dtype, gguf_loader):
    """Select an upstream loader; compact fallback is explicitly nonindependent.

    ``shared_compact`` shares the application's quantized weight loader. Such a
    run can compare pipeline logic but cannot establish loader equivalence. The
    selection is retained in the validation JSON spec, and a warning is emitted.
    No fallback is automatic: an upstream OOM remains a failed adapter run.
    """
    from pathlib import Path
    from nodes import UNETLoader
    if gguf_loader not in ("upstream", "shared_compact"):
        raise ValueError("gguf_loader must be upstream or shared_compact")
    if Path(filename).suffix.lower() != ".gguf":
        if gguf_loader != "upstream":
            raise ValueError("shared_compact is only for an explicitly selected GGUF")
        return UNETLoader().load_unet(_register("diffusion_models", filename), weight_dtype)[0]
    if weight_dtype != "default":
        raise ValueError("GGUF references require weight_dtype=default")
    if gguf_loader == "shared_compact":
        import warnings
        from modules.gguf_loader import load_diffusion_model
        warnings.warn("NONINDEPENDENT LOADER: reference shares app quantized weight loader; pipeline-only comparison", RuntimeWarning)
        return load_diffusion_model(filename)
    from molbal_comfyui_gguf.nodes import UnetLoaderGGUF
    # The plugin resolves the native `unet` category, an alias of diffusion_models.
    name = _register("diffusion_models", filename)
    _register("unet", filename)
    return UnetLoaderGGUF().load_unet(name, patch_on_device=True)[0]


def _clip(first, second, clip_type):
    from pathlib import Path
    from nodes import CLIPLoader, DualCLIPLoader
    quantized = any(Path(name).suffix.lower() == ".gguf" for name in (first, second) if name)
    if quantized:
        from molbal_comfyui_gguf.nodes import CLIPLoaderGGUF, DualCLIPLoaderGGUF
        single, dual = CLIPLoaderGGUF, DualCLIPLoaderGGUF
    else:
        single, dual = CLIPLoader, DualCLIPLoader
    first_name = _register("text_encoders", first)
    if second:
        return dual().load_clip(first_name, _register("text_encoders", second), clip_type)[0]
    return single().load_clip(first_name, clip_type)[0]


def _wan_sample(model, seed, steps, cfg, sampler, scheduler, positive, negative, latent):
    """Execute the canonical node and observe its effective sampler schedule."""
    from unittest.mock import patch
    from nodes import KSampler
    from comfy.samplers import CFGGuider
    original = CFGGuider.sample
    observed = {}

    def sample(self, noise, latent_image, selected_sampler, sigmas, *args, **kwargs):
        observed["sigmas"] = sigmas
        return original(self, noise, latent_image, selected_sampler, sigmas, *args, **kwargs)

    with patch.object(CFGGuider, "sample", sample):
        result = KSampler().sample(model, seed, steps, cfg, sampler, scheduler,
                                   positive, negative, latent, denoise=1.0)[0]
    return result, observed["sigmas"]
