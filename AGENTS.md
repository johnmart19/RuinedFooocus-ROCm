# RuinedFooocus contributor guide

RuinedFooocus is a Gradio application for local image/video generation and chat.
It embeds pinned ComfyUI backends and offers optional OpenAI-compatible APIs.
Keep changes small, readable, and scoped to the feature being fixed.

## Start here

- Check `git status`, recent commits, and the relevant `docs/` page before editing.
- `development` is the working branch; verify the current branch and remote.
- Do not overwrite local settings, model files, cached artwork, or unrelated edits.
- Put feature documentation in `docs/`; keep the main README focused.
- Read current code and pins rather than treating this guide as a version lock.

## Startup and dependencies

`RuinedFooocus.bat` / `RuinedFooocus.sh` → `entry_with_update.py` → `launch.py`
→ `webui.py`. The shell launcher activates `./venv`; the batch launcher uses
the selected Python on PATH. Both forward arguments.

- `entry_with_update.py` installs bootstrap requirements and can update the
  checkout from its remote branch. Avoid this update path while testing edits.
- `launch.py` prepares dependencies, checks out backend pins from `git_repos`,
  configures import paths, and starts the UI.
- `requirements_versions.txt`: bootstrap dependencies.
- `pip/modules.txt`: application dependencies.
- `requirements_api_versions.txt`: checked when `--api` is enabled.
- `repositories/`: downloaded ComfyUI and GGUF backends, not tracked application
  source. Make lasting compatibility fixes in `modules/` or update backend pins.
- `argparser.py` defines launch arguments. For an already prepared environment,
  use `python launch.py --offline --nobrowser --port 7863` to test without startup
  updates. Offline mode is not a general network sandbox; missing model components
  can still require downloads. API dependency checks can also run with `--api`.

See [launchers](docs/launchers.md), [Python compatibility](docs/python-compatibility.md),
and [ComfyUI compatibility](docs/comfy-compatibility.md).

## Code map

| Area | Entry points and responsibilities |
| --- | --- |
| UI | `webui.py`: Gradio components, callbacks, control registration, generation submission |
| Shared state | `shared.py`: settings, path/model managers, current pipeline, caches and UI state |
| Generation | `modules/async_worker.py`: queued jobs, progress/results, cancellation; `modules/pipelines.py`: pipeline selection |
| Images | `modules/image_pipeline.py`, `modules/upscale_pipeline.py`, `modules/controlnet.py` |
| Video | `modules/video_settings.py`: family detection/routing/timing; `modules/video_preview.py`: previews and sampling ETA; `*_video_pipeline.py` and `video_pipeline.py`: execution |
| Backend adapters | `modules/comfy_compat.py`, `modules/gguf_loader.py` |
| Chat | `modules/llama_pipeline.py`: conversation and image tool; `llama_models.py`: catalog/local models; `llama_server.py` and `llama_installer.py`: runtime lifecycle/install |
| API | `modules/openai_api.py`, `modules/api_runtime.py`; see `docs/api.md` |
| Model metadata | `modules/model_handler.py`, `modules/model_sources.py`: local detection, metadata, artwork and refresh |
| Downloads/paths | `modules/path.py`, `modules/pathdb/*.json`: configured folders and downloadable components |
| Configuration | `modules/settings.py`, `modules/performance.py`, `modules/resolutions.py`, `settings/*.default` |
| Checkpoint tools | `modules/checkpoint_tools.py`; UI under ⋮ → PowerUp → Cheat Code |

## Storage and UI conventions

- Resolve paths through `PathManager`; users can override the default folders.
- Checkpoint weights default to `models/checkpoints`. Metadata and artwork belong
  in `cache/checkpoints`, not alongside newly downloaded checkpoint weights.
- Other components use their own folders: `models/clip`, `models/clip_vision`,
  `models/vae`, `models/upscale_models`, and `models/llm`.
- Preserve `shared_cache` and refresh behavior. Refresh All Files should refresh
  model entries and artwork, respecting offline/local-only settings.
- Tensor headers identify architecture, not exact fine-tune identity. Conversion
  changes hashes; retain source metadata and previews. Do not send local model
  fingerprints to external services during testing without user authorization.
- Work with RuinedFooocus's own model folders; do not scan or import LM Studio
  libraries automatically. Explicit user imports are a separate action.
- Removing a local chat model must distinguish removing its entry from deleting
  its file, with the user's choice in the confirmation dialog.
- Keep labels short. Show quantization only when multiple choices exist; keep
  reasoning and browser Python execution optional. Show download progress until
  completion, and preserve Load/Unload behavior and runtime-switch cleanup.
- Performance presets describe model families with documented settings and native
  resolutions. Avoid preset proliferation; expose manual adjustments in Custom.
  Wallpaper output sizes are not automatically valid model recommendations.
- Preserve saved configuration overrides when introducing new defaults.

See [metadata](docs/model-metadata.md), [chat models](docs/chat-models.md),
[chat runtime](docs/chat-runtime.md), [presets](docs/model-presets.md),
[upscalers](docs/upscalers.md), and [video](docs/video.md).

## GPU and video boundaries

- Keep GPU installation changes separate from general application work. Preserve
  working installed PyTorch wheels; never hard-code a developer's GPU/backend.
- GPU-specific entry points are `runtime_support.py`, `gpu_installer.py`,
  `rocm_installer.py`, and their launch integration. Read `docs/gpu-setup.md`.
- NVIDIA compatibility includes older Pascal hardware needing CUDA 12.4; verify
  Python/wheel compatibility instead of assuming the newest CUDA works everywhere.
- Model family recognition does not prove every variant has a working pipeline.
  In particular, Wan 2.2 14B dual-expert workflows need additional support.
- Wan text-to-video does not require CLIP Vision. Use the component contract for
  the exact model/workflow rather than adding every possible encoder.
- Wan offloads its diffusion patcher before tiled VAE decoding. Use ComfyUI's
  memory manager for phase transitions; keep patchers reusable for the next job.
- Full VRAM usage alone does not establish a leak or failed offload. Separate
  sampling, prompt encoding, decoding and saving when investigating performance.
- Cancellation uses ComfyUI's interrupt mechanism. Preserve worker recovery so
  stopping one job does not break subsequent generations.

## Validation and delivery

- Use focused syntax/config checks first, then a small relevant runtime/UI test.
  Avoid repeated long generations or concurrent GPU-heavy test processes.
- A running user's UI may contain an active job. Check before stopping/restarting
  it; target the verified RuinedFooocus process, never all Python processes.
- Put ad hoc scripts, logs, browser captures and test artifacts in ignored
  `tmp/local-tests/`. Do not commit them, downloaded models, credentials or outputs.
- For UI changes, verify rendered behavior, not just callback return values.
  For generation changes, inspect the output; successful completion alone does
  not establish image/video quality or prompt adherence.
- Report what was actually tested. Syntax checks, successful component loading,
  and completed GPU generations are different levels of evidence. Do not claim
  cross-platform or NVIDIA hardware validation from AMD-only tests.
- Before committing, run `git diff --check` and inspect the staged file list.
  Use focused commits with concise descriptions and the established trailer:
  `Co-authored-by: Codex <codex@openai.com>`.
- Keep general changes independent of the optional final commits titled
  **Add Windows and Linux launchers**, **Add Windows and Linux GPU runtime detection**,
  and **Integrate AMD ROCm and NVIDIA CUDA installation**. When explicitly asked
  to reorder history, back it up, verify identical final trees and test reverting
  the optional commits. Do not rewrite published history merely for routine edits.
- Push only when requested. For an authorized history rewrite, use an explicit
  force-with-lease and verify the remote still matches the expected old tip.
