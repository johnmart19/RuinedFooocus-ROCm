# Source FIXME audit

Reviewed the tracked source notes in September 2026. Tokenizer vocabulary entries
containing `FIXME` are model data and are unchanged. Downloaded repositories are
maintained through their pins, not edited locally.

| Area | Resolution |
| --- | --- |
| Settings | Keep missing saved checkpoints and LoRAs selectable so unrelated settings can be saved. Write each JSON file through an atomic replacement; serialization or replacement failure preserves the previous file. |
| Component paths | Include local CLIP/VAE/component files alongside catalog downloads; search all configured folders before downloading. |
| Inbox | Resolve files relative to their actual Inbox root. Preserve nested paths and move cached artwork with the model. Leave collisions in Inbox instead of overwriting weights or metadata. |
| Image browser | Reuse the shared path manager, parameterize metadata searches, fix exact-page boundaries, and retain GIF comments and dimensions. Searches still cover all metadata. |
| Chat context | Validate and batch complete text/URL documents into one index for both llama.cpp and xllamacpp. Publish the index only after successful construction. |
| Image preview | Display the first synchronous preview; job initialization already resets the preview grid. |
| Image routing | Detect Kontext by local model metadata or filename, while preserving explicit selection. Ordinary Flux image-to-image no longer automatically enters Kontext mode. |
| Diffusion loading | Let ComfyUI choose dtype from the model and hardware instead of forcing FP8/BF16. GGUF keeps its quantization. Recognize uppercase GGUF extensions in image and generic video loading. Use the catalog's Wan 2.1 VAE default. |
| Video | Use family-specific frame/spatial alignment, MiniMax's own image-conditioning node, and explicit LTX audiovisual latent handling. Initialize video-only decoding output and offload the diffusion patcher before VAE decoding. GIF previews are capped at 512 pixels per side; MP4 resolution is unchanged. |

Obsolete notes removed: chatbot greetings already come from `info.json`;
`known_model_info.keys()` already supplies the supported model names; Flux
guidance already uses conditioning correctly. Commented image-model patches
in the video pipeline were inactive. Legacy LTXV remains routed to its dedicated
pipeline, rather than the incomplete duplicate in the audiovisual pipeline.
No arbitrary default video download was added: the selected checkpoint determines
the required components.

## Validation

- Windows and WSL: temporary-data checks for atomic saves, missing-model Gradio
  preprocessing, local paths, Inbox collisions, metadata searches and chat indexing.
- Windows full UI: with an empty checkpoint folder, changed the backend to Vulkan
  and saved successfully using an isolated settings profile.
- Windows: consecutive-job first-frame checks and pinned ComfyUI node checks for
  LTX/MiniMax text/image latents and LTX audiovisual splitting, using CPU tensors.
- The initial audit used CPU contracts; the Windows GPU follow-up below exercises
  actual model weights. NVIDIA hardware and MiniMax generation remain untested.

### Windows GPU follow-up

RX 7900 XTX (24 GB), 128 GB system RAM, installed ROCm PyTorch. Jobs ran serially
through RuinedFooocus/Gradio with isolated settings and outputs; no installer changes.

| Checkpoint | Test | Result |
| --- | --- | --- |
| Wan 2.2 TI2V 5B FP16 | 1280x704, 25 frames, 20 steps | Completed in 95 seconds; coherent fox/meadow footage. Peak allocation 16.8 GiB; diffusion offloaded before VAE decoding. |
| Wan 2.1 T2V 1.3B Q8_0 | 832x480, 49 frames, 30 steps | Completed in 169 seconds; recognizable fox walking through green grass. The shorter 17-frame test had much less natural colors. |
| Flux1 Dev FP8 | 1024x1024, 20 steps | Completed in 54 seconds; car/cafe scene and requested HELLO sign rendered correctly. |
| LTX 2.5 distilled INT8 convrot | 768x512, 25 frames, 8 steps, CFG 1 | Pipeline completed in 90 seconds; video and audio saved. Peak allocation 20.3 GiB; diffusion offloaded before VAE decoding. The animal looked somewhat cat-like. |

These are short functional checks, not quality benchmarks or guarantees for longer
clips. MP4s were checked separately from GIF previews, which have limited colors.
The subsequent LTX two-stage implementation was tested through Gradio on the same
GPU: 768x512 initial generation, learned latent upscaling, and three-step refinement
produced a 1536x1024, 25-frame MP4 with audio in 147 seconds including the first
upscaler download (20.7 GiB peak allocation). The output was coherent but cartoon-like
despite the realistic prompt. A 768x512 image-to-video run preserved the supplied
first frame and animated the subject; it completed in 90 seconds immediately after
cancelling a preceding job during refinement. Both sampling stages, upscaler download
progress, and diffusion offload before upscaling/decoding were observed. Longer
clips and additional checkpoint variants remain outside these short tests.

Ad hoc checks and logs stay in ignored `tmp/local-tests/`. User settings, model
files, the local launcher edit, and GPU installation code are outside these commits.
