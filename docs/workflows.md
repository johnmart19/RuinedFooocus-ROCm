# Workflow controls

The application exposes fixed pipelines, not ComfyUI's full node graph. A model
download or a valid sampler name does not establish workflow support.

Open **Setting → Performance → Workflow settings** to inspect the current recipe
and find its input controls. Select **Custom...** to edit supported sampling
parameters. Fixed video schedules remain fixed.

| Workflow | UI route | Boundary |
| --- | --- | --- |
| Text-to-image | Models checkpoint, Performance, prompt, Generate | Use the recipe for the exact model variant |
| Image-to-image | PowerUp → Custom → Img2img, Input image, Denoise | Denoise ranges from 0.01 to 1 |
| Image editing | Matching edit checkpoint and PowerUp Input image | Only implemented edit architectures; not every image model |
| SDXL ControlNet | PowerUp Canny, Depth, Sketch or Recolour | Catalog controls target SDXL |
| Upscale / restore | PowerUp preset, or Custom → Upscale → Upscaler | GFPGAN restores a cropped face; it is not face swapping |
| Background removal | PowerUp → RemBG, Input image | Separate image-processing pipeline |
| Wan 2.1 | Matching T2V/I2V checkpoint, Wan 2.1 preset | I2V requires image-trained weights; T2V weights are not interchangeable |
| Wan 2.2 TI2V 5B | Wan 2.2 5B preset, optional PowerUp Input image | Does not implement 14B dual-expert sampling |
| Hunyuan Video / original LTX Video | Matching preset, duration, optional Input image | Use weights supporting the requested mode |
| LTX 2.5 distilled | LTX 2.5 Distilled preset, duration, size, optional Input image | Automatic two-stage sampling and latent upscale |
| LTX 2 / 2.3, non-distilled 2.5, MiniMax H3 | Custom, duration, size, component overrides in Settings | Loader routes exist; no verified bundled recipe or complete generation validation for every variant |
| LCM / DMD2 SDXL | Models LoRA plus Custom sampling settings | No automatic combined LoRA-and-sampling recipe |

For video image input, keep **PowerUp → Cheat Code** set to **None**. Upscale and
RemBG select a different pipeline. Video duration and FPS appear only for video
models; FPS is under Custom where adjustable. Use Models for LoRA selection.

ComfyUI advanced sigma editors, arbitrary node workflows, Wan dual-expert model
switching and Hunyuan 1.5 are not implemented by merely listing their components.
They need dedicated integration and validation. Do not treat Custom as a way to
enable an absent pipeline.

For optional random prompts with an input image, enable **PowerUp → Generate
random prompt**. Preset, enhancement, prompt style and subject controls appear
and share the existing One Button settings. For video, describe desired motion
in Subject. This replaces the submitted prompt; it does not analyze the image.
The option defaults to off and has no effect without an input image or inpainting.

**Image to image (preserve details)** is a separate, conservative preset: denoise
0.25, no selected style templates, and each batch image starts from the uploaded
reference rather than the previous output. Explicit prompts, inline styles,
negative prompts, LoRAs and opt-in random prompting still apply. It does not
extract a character description or guarantee identity preservation. The original
Img2Img preset is unchanged. Lower denoise generally retains more reference
structure; see [ComfyUI's img2img guide](https://docs.comfy.org/tutorials/basic/image-to-image).

This is a source-level control audit. Registry/configuration checks do not replace
generation tests. See [video](video.md), [model presets](model-presets.md) and
[downloads](model-downloads.md) for model-specific contracts.
