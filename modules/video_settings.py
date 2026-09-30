"""Video families and frame timing shared by the UI and pipelines."""

VIDEO_FPS = {
    "Wan Video": 16,
    "Wan 2.2 5B": 24,
    "Hunyuan Video": 24,
    "LTXV": 24,
    "LTXV2": 24,
    "LTXV 2.3": 24,
    "LTXV 2.5": 24,
    "MiniMaxH3": 24,
}


def is_ltx25_distilled(family, checkpoint):
    # Tensor shapes identify the family, not whether its weights were distilled.
    return family == "LTXV 2.5" and "distilled" in str(checkpoint).replace("\\", "/").rsplit("/", 1)[-1].lower()


def fixed_video_settings(family, checkpoint=""):
    if family not in VIDEO_FPS:
        return {}
    fixed = {"clip_skip": 1}
    if family.startswith("LTX"):
        fixed["scheduler"] = "simple"  # The pipeline uses LTXVScheduler instead.
    if family == "MiniMaxH3":
        fixed.update(cfg=1, video_fps=24)
    if is_ltx25_distilled(family, checkpoint):
        fixed.update(custom_steps=8, cfg=1, sampler_name="euler_ancestral")
    return fixed


def uses_negative_prompt(family, checkpoint=""):
    return family not in ("MiniMaxH3", "Hunyuan Video") and not is_ltx25_distilled(family, checkpoint)


def constrain_video_settings(data, family):
    if family not in VIDEO_FPS:
        return data
    result = dict(data)
    result.update(fixed_video_settings(family, result.get("base_model_name", "")))
    if not uses_negative_prompt(family, result.get("base_model_name", "")):
        result.update(auto_negative=False, negative="", negative_prompt="")
    return result


def frame_count(seconds, fps, multiple=4):
    return max(1, round(float(seconds) * float(fps) / multiple) * multiple + 1)


VIDEO_PIPELINES = {
    "Wan Video": "wan_video", "Wan 2.2 5B": "wan_video",
    "Hunyuan Video": "hunyuan_video", "LTXV": "ltx_video",
    "LTXV2": "video_pipeline", "LTXV 2.3": "video_pipeline",
    "LTXV 2.5": "video_pipeline", "MiniMaxH3": "video_pipeline",
}


def detect_video_model(tensors):
    # Check architecture keys for both standalone and bundled checkpoints.
    for prefix in ("", "model.diffusion_model.", "diffusion_model."):
        def has(key):
            return prefix + key in tensors
        def shape(key):
            return tensors.get(prefix + key, {}).get("shape", [])
        if has("head.modulation") and has("patch_embedding.weight"):
            return "Wan 2.2 5B" if shape("head.head.weight")[:1] == [192] else "Wan Video"
        if has("txt_in.individual_token_refiner.blocks.0.norm1.weight"):
            return "Hunyuan Video"
        if has("video_patch_proj.weight") and has("audio_patch_proj.weight"):
            return "MiniMaxH3"
        if has("adaln_single.emb.timestep_embedder.linear_1.bias") and not has("pos_embed.proj.bias"):
            if has("audio_adaln_single.linear.weight"):
                if 4096 in shape("keyframes_abs_pos_embedding"):
                    return "LTXV 2.5"
                if has("transformer_blocks.0.attn1.to_gate_logits.weight"):
                    return "LTXV 2.3"
                return "LTXV2"
            return "LTXV"
    return None
