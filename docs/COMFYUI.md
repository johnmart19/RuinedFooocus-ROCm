# Embedded ComfyUI backend

RuinedFooocus pins [ComfyUI v0.39.0](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.39.0)
at `b0b743566f65daafc423b4fea8a2fbda94b3384a` and uses Comfy Kitchen 0.2.37.
Its Gradio UI calls backend models and nodes directly; it does not launch the
ComfyUI workflow editor or asset server.

| Release change | RuinedFooocus integration |
| --- | --- |
| Expanded ComfyAttention/tensor-container support | Existing image/video model implementations inherit the backend attention changes. |
| MiniMax-H3 embedding-memory and VAE offload fixes | Used by the existing MiniMax-H3 video pipeline without replacing its controls. |
| Integrated-GPU pinned-memory policy | Backend detects integrated GPUs automatically; discrete GPUs retain the normal policy. |
| Offline/partner-node flag split | `--offline` and `RF_OFFLINE=1` forward offline mode to the embedded backend and disable its partner nodes. This does not add a network firewall. |
| Compiler/graph controls | `--disable-comfy-compiler` disables both the model compiler and CUDA graphs before backend initialization. |
| Host memory override | `--disable-pinned-memory` forwards an explicit override before backend initialization. |
| Asset scanning/database fixes | Included in the pinned source; unused by the separate RuinedFooocus model cache/gallery. |
| Workflow editor/templates, paid partner nodes, bounding-box preview and EXR nodes | Included in the source, but not exposed as new Gradio workflows. Backend node availability alone is not application workflow support. |

Normal online startup updates the pinned backend and dependency. Offline startup
uses the installed checkout/packages and performs no upgrade. Existing Torch/ROCm
selection and compatibility handling remain in the launcher.

## Command-line argument support

Compute arguments are defined in `modules/comfy_args.py` and applied before
backend model-management imports. Their types, defaults, optional values and
mutually exclusive groups match the pinned upstream parser. They cover:

* Precision for diffusion, VAE, text encoders and intermediate tensors.
* Attention implementations/upcasting, xformers, Triton and FP8 capability overrides.
* VRAM modes/headroom, async offload, DynamicVRAM, fast disk and smart memory.
* Compiler/CUDA graphs, deterministic execution, nonblocking operations and mmap.
* Pinned memory, experimental `--fast` features and CUDA allocator selection.

`--force-fp16` also implies `--fp16-unet`; `--fast` without values enables all
upstream performance features. `--disable-comfy-compiler` also disables graphs.
The legacy `--normalvram` disables DynamicVRAM. Image and video pipelines respect
backend smart memory: keep reusable diffusion, encoder and VAE weights resident
when the active stage's weights and workspace fit; offload only when more memory
is needed. Video sampling, decoding and LTX refinement do not force diffusion
or encoder unloading at every stage boundary. Temporary refinement upscalers
are released after use. Explicit VRAM modes and `--cpu-vae` retain their meaning;
automatic residency does not force GPU-only loading or CPU VAE decoding.

Under the image resolution controls, **Automatic Upscale** offers Off (default),
explicit RealESRGAN general/anime choices, UltraSharp and catalogued/local
upscalers with a named 2×/3×/4×/8× scale. Output size uses a wrapping field in
the same control group. Additional controls hide when Off. The default list
contains everyday anime/general models; **Show advanced upscale models** adds
specialist filenames. Select **Generated image** to upscale after generation,
or **Input image only** to switch the main preview to an upload area without
diffusion sampling. The latter estimates dimensions from the uploaded image;
switching back restores the session's generated preview (or the startup image).
Input-only progress/results do not overwrite that generated preview. Upscale shortcuts have
moved out of Cheat Code, while saved legacy presets remain supported. It runs
ComfyUI's tiled `ImageUpscaleWithModel` after VAE decoding and before saving;
it does not change the diffusion resolution or run another sampling pass.
Model choice is explicit and independent of checkpoint family. Weights download through the existing upscaler catalog
on first use and are cached between prompts. Stop is checked between tiles;
download/backend failures follow the normal worker error recovery. PNG metadata
records the final size, original generation size, scale and upscaler. Video
generation is unaffected; the legacy standalone backend remains available.

Native 2× anime choices include [Futsuu Anime](https://openmodeldb.info/models/2x-Futsuu-Anime)
(compact general animation, WTFPL), [AniScale](https://openmodeldb.info/models/2x-AniScale)
(compact artifact cleanup, CC-BY-NC-4.0), and
[AnimeSharp V4 Fast RCAN PU](https://openmodeldb.info/models/2x-AnimeSharpV4-Fast-RCAN-PU)
(sharpening/detail and compression cleanup, CC-BY-NC-SA-4.0). The latter two
carry noncommercial licenses. Author/source/license metadata is retained in
the download catalog. These are true 2× models, without a 4× resize intermediate.
The pinned DynamicVRAM controller is initialized
without launching the ComfyUI server.

`--gpu-device-id` retains single-GPU selection. `--cuda-device` accepts a comma
list or `all` (preserves existing visibility); these two options are exclusive.
`--default-device` orders a preferred physical index first when no explicit
CUDA selection is supplied. Visibility is set for CUDA/HIP before Torch startup;
`--oneapi-device-selector` sets the Intel selector. Deterministic mode preserves
an existing CUBLAS configuration or supplies the backend's default.

Explicit `--cuda-malloc` / `--disable-cuda-malloc` override the allocator backend,
disable expandable segments and retain other allocator fields. Without these
flags the application's existing allocator policy and user environment remain.
Attention accelerators, kernels, FP8, async allocators and compiler features still
require compatible packages/hardware; accepting a flag does not install them or
establish support on every device.

### ComfyUI-only options

The following flags are deliberately rejected instead of accepted without effect:

| Scope | Upstream options / RuinedFooocus equivalent |
| --- | --- |
| Workflow execution cache | `--cache-ram`, `--cache-classic`, `--cache-lru`, `--cache-none`, `--high-ram`; RF uses direct pipelines and its own caches. |
| Model/path directories | `--base-directory`, `--extra-model-paths-config`, `--models-directory`, `--input-directory`, `--output-directory`, `--temp-directory`, `--user-directory`; use RF path settings. |
| Output metadata | `--disable-metadata`, `--default-hashing-function`; RF has its own preview, metadata and model-cache logic. |
| Server transport | `--tls-keyfile`, `--tls-certfile`, `--enable-cors-header`, `--max-upload-size`, `--enable-compress-response-body`, `--multi-user`; ComfyUI's server is not started. RF's `--listen`/`--port` retain RF semantics and authentication. |
| Editor/Manager | `--front-end-root`, `--front-end-version`, `--enable-manager`, `--disable-manager-ui`, `--enable-manager-legacy-ui`, `--feature-flag`, `--list-feature-flags`; no Comfy workflow editor is served. |
| Partner/custom nodes | `--comfy-api-base`, `--disable-api-nodes`, `--disable-partner-nodes`, `--disable-all-custom-nodes`, `--whitelist-custom-nodes`; RF does not discover those nodes. Use RF `--offline` for offline startup. |
| Asset service | `--database-url`, `--enable-assets`, `--enable-asset-hashing`; RF uses its separate cache/gallery. |
| Server startup/logging | `--auto-launch`, `--disable-auto-launch`, `--windows-standalone-build`, `--dont-print-server`, `--quick-test-for-ci`, `--debug-hang`, `--verbose`, `--log-stdout`; ComfyUI main hooks are not run. RF browser control is `--nobrowser`. |

## Image and video pipeline contracts

The image pipeline uses `comfy.sample.sample`, matching the backend's model
load device, noise-mask preparation and intermediate output device/dtype.
Image, Wan, Hunyuan and LTX sampling forward spatial/temporal latent scale hints
and remove consumed hints from sampler outputs. Batch indices, masks and other
latent metadata remain intact; nested audiovisual latents use backend nodes.
Hunyuan schedules use the same shifted model as its guider. Progress-bar flags
are respected by both ordinary and custom sampling paths.

VAE loading preserves checkpoint metadata and validates the constructed VAE.
Changing encoder/VAE settings invalidates component caches, and changing LoRAs
invalidates cached text conditioning. Per-image ControlNet conditioning stays
separate from cached text. Missing/broken components, pipeline selection and
audio decoding errors reach the task recovery UI instead of silently producing
a logo, empty result or video with missing audio. Models without RGB latent
factors skip optional latent previews.

The worker starts after pipeline modules finish importing, preventing startup
from selecting partially initialized classes. A new request can restart a dead
worker; Python-level recovery does not repair a crashed native process/driver.

### Sampling previews

Auto preview mode uses an installed approximate VAE decoder when available,
then falls back to the model's latent RGB projection. Anima uses Wan 2.1
latents; its RGB projection is a coarse color estimate, not a decoded image.
For detailed Anima previews, place
[lighttaew2_1.safetensors](https://huggingface.co/lightx2v/Autoencoders/resolve/main/lighttaew2_1.safetensors)
in `models/vae_approx/`. The same decoder can preview a still image as a
single video frame. Explicit disabled, latent RGB and TAESD preview choices
are respected. Decoder loading failures are logged and use the RGB fallback.

RuinedFooocus defaults `--preview-method` to `auto`. Other choices are `none`, `latent2rgb` and `taesd`. Preview decoding is approximate and does not change sampling or final VAE decoding.

### Explicit style insertion

The style dropdown starts empty. Selecting a style only stages it for the
**Send Style to prompt** button; sampling does not apply dropdown selections
or saved default styles. The button inserts style text into the visible
positive and negative prompts and clears the selection. Explicit `<style:...>`
prompt tags remain supported. API clients can explicitly supply styles, but
API/chat generation no longer inherits the UI's saved style defaults.

### Dependency maintenance at startup

Successful bootstrap and pip-maintenance checks are cached locally in
`cache/dependency-checks.json`. App commits, dependency-file contents,
Python/environment identity and installed package metadata invalidate the
cache. A `reinstall` request forces maintenance. Unchanged launches skip the
recurring pip upgrade/bootstrap/uninstall commands; package compatibility and
GPU selection checks still run. Required pip operations use quiet output,
while errors remain visible. After an app update, the updater restarts Python
before checking its new dependency files to avoid replacing loaded native DLLs.
Delete the cache file to request fresh maintenance without a forced reinstall.

Anima sampling preserves the backend's native shift 3 unless `anima_image_shift` is explicitly configured. Reproducing an external image also requires matching the complete workflow and component versions.

PixArt sampling rejects latent sizes that would produce a zero positional-embedding base, with guidance to use 512x512 or a larger supported resolution.

### GGUF dependency

The launcher pins molbal/ComfyUI-GGUF to
`5a0a3ffa0e3eae5c6af8b0b981b660a24d5fbc04`. Online startup installs
that revision; the repositories directory supplies its Python package to the
image and video pipelines. Diffusion GGUFs use its quantization-aware UNet
loader; GGUF text encoders use its CLIP loader and model patcher.

The application's pinned gguf, sentencepiece, protobuf and comfy-kitchen
packages satisfy the dependency's requirements. Triton is optional: available
native K-quant kernels are selected automatically, with dequantized execution
when an accelerator is unavailable or fails. Do not replace a working vendor
Torch installation to obtain optional acceleration. Qwen vision projectors
must accompany their encoders in the same directory; the updated loader handles
the Qwen3-VL/Qwen3VL filename alias and rejects ambiguous alias matches.

## Qwen Image attention on Windows ROCm

Qwen Image automatically uses ComfyUI's sub-quadratic attention on Windows ROCm when no attention implementation is explicitly selected. Its memory estimate matches that attention path, allowing the backend to offload weights before temporary tensors exhaust physical VRAM. Other model families, platforms, memory flags and explicit attention choices keep their existing behavior. This does not alter sampling steps, CFG, resolution or checkpoint weights.

For an optional distilled workflow, select a compatible [Qwen Image Lightning adapter](https://huggingface.co/lightx2v/Qwen-Image-Lightning) at strength 1 and use its published steps, CFG and sampling shift. A lower step count without the matching adapter is not an equivalent quality setting. Adapters are never automatically loaded.
