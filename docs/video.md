# Video generation

Select a recognized video checkpoint to show **Video duration (seconds)**.
Choose **Custom...** under Performance to adjust frame rate and sampling.
Image checkpoints hide these controls and retain Image Number.

Local Safetensors and GGUF detection covers the existing Wan, Hunyuan Video,
LTX Video/LTX-2, and MiniMax H3 pipelines. Detection uses tensor signatures,
including checkpoints with diffusion-model prefixes, rather than filenames.

Wan 2.2 TI2V 5B uses 20 steps, CFG 5, UniPC/simple, shift 8 and 24 FPS.
Its preset recommends 1280x704 or 704x1280 and loads `wan2.2_vae.safetensors`.
See the [ComfyUI workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_wan2_2_5B_ti2v.json).
The 14B dual-expert Wan 2.2 workflows require additional pipeline support;
detecting a video family does not implement those workflows.

The Wan 2.1 preset uses UniPC, 30 steps and CFG 6, matching the
[ComfyUI workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/text_to_video_wan.json).
For the 1.3B model, use 832x480 or 480x832.
Text-to-video uses UMT5 and the Wan VAE with denoise 1.0; CLIP Vision is only
needed for supported image-to-video models.

Frame counts are rounded to the model's temporal layout. For example, Wan
uses 49 frames for 3 seconds at 16 FPS, so actual duration can differ slightly.
Longer videos and higher resolutions require more VRAM.

Wan offloads diffusion weights before VAE decoding and uses 512-pixel decoding
tiles, matching ComfyUI's tiled decoder default. The model reloads when sampling
the next video; its file stays on disk.

While sampling, the progress bar shows elapsed time, estimated time remaining,
and seconds per step. The estimate updates after each step. Stop interrupts
generation at the next processing checkpoint. The logo remains visible until
the first live preview; the completed animation replaces it after saving.

GGUF video types are read locally from model metadata. Preview artwork and
source information belong in `cache/checkpoints`; offline mode does not fetch
missing artwork.

For an LTX 2.5 **distilled** checkpoint, select **LTX 2.5 Distilled**. It uses
CFG 1 and the publisher's fixed eight-step schedule with ancestral Euler, skipping
unused negative-prompt encoding. It generates at half the selected resolution,
applies the learned 2x latent upscaler, then refines video and audio with three
deterministic Euler steps. The selected resolution is the final output size;
1536x1024 starts at 768x512. Dimensions align to multiples of 64.

The matching upscaler downloads automatically to `models/latent_upscale_models`
(configurable with `path_latent_upscalers`). Hugging Face access/login is required
if the file is missing. Progress identifies both sampling stages and upscaling;
Stop remains available between phases and during sampling. Diffusion weights are
offloaded before upscaling and final decoding. Image input is reapplied at the
refinement resolution, while the generated audio continues through both stages.

Custom keeps this two-stage recipe for a recognized distilled checkpoint; it only
exposes duration, FPS and output size. Steps, CFG, sampler and schedule stay fixed,
including for imported settings or API requests. The `distilled` filename marker
identifies the recipe: architecture alone cannot distinguish distilled weights.
Unidentified variants remain Custom rather than being assigned a distilled recipe.
See the [model card](https://huggingface.co/Lightricks/LTX-2.5) and
[pipeline constants](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py).

Video selection automatically picks its matching Performance recipe, including at
startup. Custom hides Clip Skip for all videos, Scheduler for LTX pipelines, and
unused CFG/negative-prompt controls for MiniMax. MiniMax FPS stays at 24. Hunyuan's
CFG control is labeled Guidance and its unused negative prompt is hidden. Image
models retain their existing presets and controls. Adjustable sizes and durations
can still exceed available memory; these controls do not guarantee VRAM fit.

Video LoRAs use the selected weight during both LTX sampling stages. Loading a
LoRA preserves the text encoder and video/audio VAEs; changing LoRAs invalidates
cached prompt conditioning. A missing or unreadable LoRA stops the job with its
filename instead of silently producing an unstyled result.
