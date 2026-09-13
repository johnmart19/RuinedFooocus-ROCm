# Model settings

## Performance

Choose SDXL, Illustrious XL, Pony XL, NoobAI XL, FLUX.1 dev, FLUX.1 schnell,
or SD3.5. Use **Custom...** for checkpoint-specific or distilled settings.
Retired recipes open as Custom; saved values and user presets are preserved.
The selector shows recommended resolutions.

Qwen Image uses 50 steps and CFG 4 from its [model card](https://huggingface.co/Qwen/Qwen-Image),
with the listed square and widescreen resolutions. This recipe is for the base
text-to-image model, not accelerated LoRAs or image-editing variants.
Z-Image Turbo uses the [ComfyUI workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_z_image_turbo.json)
recipe: 8 steps, CFG 1, res_multistep/simple. It is not a recipe for Z-Image Base.

Recognized video models show matching video presets. The LTX 2.5 Distilled
recipe is for distilled checkpoints only; see [video settings](video.md).
Video presets are selected on model changes and startup. Hunyuan Video and legacy
LTX Video use the matching [Hunyuan](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/hunyuan_video_text_to_video.json)
and [LTX](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/ltxv_text_to_video.json)
ComfyUI examples. Families without a matching recipe use Custom; hidden controls
are enforced by the backend, not merely removed from view.

Illustrious defaults to 30 steps, CFG 5 and CLIP Skip 2, matching
[JANKU v7.77's range](https://civitai.red/models/1277670/janku-trained-chenkin-and-noobai-rouwei-illustrious-xl).
[FLUX.1 dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) demonstrates 1024x1024 generation.
A checkpoint author's recipe takes precedence over family defaults.

## Aspect ratios

Image models show the image sizes from `settings/resolutions.default`. Video
models show their family's sizes from `settings/video_resolutions.default`.
The list switches at startup and when selecting a model. An incompatible previous
size is replaced by the first matching entry; Custom remains available.

Saved dimensions are preserved. Older bundled video entries are classified
automatically, and new custom sizes saved for a video belong to that family.
The combined internal lookup remains available for older presets and API requests.
Additional video sizes follow the ComfyUI [LTX 2](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_ltx2_t2v.json)
latent dimensions and [MiniMax](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json)
native canvas. A listed resolution does not guarantee that a model fits in VRAM.
