# Download catalog

Downloads report bytes received, percentage when the size is known, and average MB/s.
Files become available only after a complete transfer. Failed partial files are removed.

The JSON files in [modules/pathdb](../modules/pathdb) are the authoritative
source for download URLs, filenames and storage folders. The UI reads these
files directly; this guide describes their use rather than duplicating the catalog.
Catalog availability does not imply that every model has been generation-tested.

## Use and compatibility

| Catalog | Application route | Settings |
| --- | --- | --- |
| checkpoint | Image loader; SDXL base and Top5 | SDXL Performance preset |
| clip / vae | Matching image/video loader | Settings encoder/VAE overrides; blank uses the model default |
| clip_vision | Wan 2.1 image-to-video conditioning | CLIP Vision; not needed for Wan text-to-video |
| latent_upscalers | LTX 2.5 distilled second stage | Automatic for the two-stage workflow |
| llm | Chat and One Button local chat enhancement | Chat model and quantization; llama.cpp or xllamacpp |
| llm_vision | Optional chat image review and matching projectors | Enable Model Vision; stored separately in models/vision |
| controlnet | SDXL control LoRAs | PowerUp control type; not interchangeable with other model families |
| lora / lcm | Image LoRA loader | SDXL LoRA selection and matching checkpoint recipe |
| upscalers | Spandrel plus extra architecture adapters | PowerUp upscaler selection |
| faceswap | GFPGAN through Spandrel | PowerUp: Restore cropped face (restoration, not identity swapping) |

Use components from the same architecture and release. A list entry is not a
universal replacement for every encoder or VAE. GGUF, FP8 and NVFP4 variants also
depend on the installed backend and device. Auxiliary files such as Gemma's
`mmproj`, Ernie's prompt enhancer and LTX embedding connectors are not standalone
text encoders; only the matching workflow can consume them. Hunyuan 1.5's VAE
entry alone does not implement a Hunyuan 1.5 generation workflow.

Catalogued checkpoints and LoRAs are also offered in Settings and downloaded on
first load when online. Select the LCM LoRA and its matching sampler together; DMD2
weights require their own four-step recipe and must not be applied to unrelated
model families.

LTX 2 selects its own VAEs and Dev/Distilled connector; LTX 2.3 selects its text
projection and VAEs; LTX 2.5 uses its combined Gemma encoder and separate VAEs.
Encoder/VAE changes take effect on the next model load. Other video pipelines
may require restarting after changing advanced component settings.

Performance recipes and native resolutions are documented in [model presets](model-presets.md)
and [video](video.md). Use Custom for variants without a verified recipe; a base
recipe must not silently stand in for a distilled, editing or dual-expert workflow.

## Removed download choices

- Rogue Q3_k_l was deleted upstream; its existing Q3_k_m option remains.
- Six Q4_0_4_4 / Q4_0_4_8 / Q4_0_8_8 chat downloads use formats removed by
  [llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/10847). Use a standard Q4 or K quantization.
- SPSR is no longer implemented by the current Spandrel packages. README.md is
  documentation, not an upscaler. Neither belongs in the downloadable model menu.
- Existing local files are not removed by these catalog changes.
