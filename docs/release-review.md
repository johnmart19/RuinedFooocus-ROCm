# Release review — 2026-09-30

Reviewed the consolidated application and GPU setup changes on `development`.
Experimental changes on `optional` were not merged as part of this review.
Automatic conversation summaries and context growth remain experimental.

## Final corrections

- Normal conversation is the default chatbot. UI and API share image-tool guidance.
  Actual local image calls are replayed as paired tool messages, retaining their
  complete prompts for revisions. Force image generation applies to one message.
- Old reasoning is not replayed as conversation facts. Vision observations and
  image payloads expire after the next follow-up without changing the transcript.
- Native chat probes its own GPU support; CPU PyTorch does not disable Vulkan.
  Explicit CPU selection remains respected. Reference indexing loads only when used.
- Explicit API img2img dimensions fit/crop the input instead of silently retaining
  a different aspect ratio. The size schema uses portable ASCII digits.
- Worker failures finish their requests and leave the queue usable. Unrelated
  queued outputs no longer cause a busy wait. Nested image calls do not leak events.
- Wan failures are errors, not successful outputs containing an error picture.
  Removed unused preview decoders and commented loader experiments. Bundled video
  loading passes its diffusion model into component loading instead of discarding it.
- Preset writes use the existing atomic JSON writer. Fixed undefined path-manager
  references in checkpoint and LoRA lookup helpers.

## Verification

Windows, Python 3.12.10, RX 7900 XTX, PyTorch ROCm; chat used Vulkan.
The review used a separate settings profile and ignored test outputs.

| Area | Evidence from this review |
| --- | --- |
| Consolidated source/configuration | Python 3.10 grammar and JSON parsing across the changed files; static name checks and diff checks |
| Chat routing | Qwen 2.5 7B on xllamacpp and GPT-OSS 20B on native llama.cpp: 13 cases each, including Akame revision, topic changes, prompt-only replies and forced calls; native Qwen also tested through API |
| Chat UI | Normal text reply; force option generated Akame and reset; beach/sunglasses revision generated again; a subsequent story request returned text without generation |
| Image/API | Actual Janku img2img at 768×512, RealESRGAN 2× upscale, invalid-request recovery and subsequent chat; output inspected |
| Video/API | Wan 2.1 1.3B GGUF, 832×480, nine frames, 30 steps; completed output and representative frames inspected |
| Failure handling | Injected pre-sampling failure followed by successful dispatch; bundled-video patcher handoff checked with mocked components |
| Regression checks | Chat context/tool history, schemas, atomic-write failure handling and Canny bounds passed before removing test files from the release tree |
| API contracts | Auth, streaming, inline vision input, image edits, tools, video/upscale/review schemas and invalid-input handling |

This is not certification of every downloadable model or hardware combination.
NVIDIA, DirectML, CPU-only and WSL generation were not rerun on this Windows AMD
machine. Full LTX, Hunyuan and MiniMax generations were not rerun. Catalog/schema
validity and family detection do not prove arbitrary checkpoints work; existing
workflow limitations remain documented in [workflows](workflows.md).

Tool routing depends on model behavior. The tested examples passed; that does not
promise flawless interpretation of every ambiguous request or character likeness.
Use explicit prompt-only wording or the one-message force option when needed.
The Akame revision retained the requested details in its prompt, but the inspected
Janku output did not render the sunglasses cleanly. Routing success is distinct
from perfect visual adherence.

## Subsequent NVIDIA validation

The 2026-09-30 clean Windows RTX 4060 setup verified JANKU v7.77 with the
Illustrious XL profile, API image-to-chat recovery, native Vulkan/CUDA chat and
xllamacpp CUDA chat. It also caught a newer `nvidia-smi` driver-header format and
added Windows launcher environment selection/creation checks. Full details and
limits are recorded in [GPU setup](gpu-setup.md#verified-windows-nvidia-setup-2026-09-30).

Final cleanup reran all 20 focused regression/launcher checks and the installed
package dependency check. Another 42 simulated vendor, override and installed
runtime cases passed without installing packages. These include Windows/Linux
paths, NVIDIA under WSL without PCI discovery, and an AMD system with a stale
NVIDIA driver header. Tests and their logs are retained locally under ignored
`tmp/local-tests/`, rather than shipped in the release tree.

The final history folds follow-up corrections into their feature commits. The
three GPU setup commits remain last so the application changes can be reviewed
independently. Occupied-port recovery is retained; automatic context growth and
conversation summarization remain outside this release.

## Optional branch follow-up

`optional` is based on the finalized `development` release and adds two beta
changes: cached-chat isolation/cleanup and automatic context growth/compaction.
Port recovery and the portable resolution schema are already in `development`
and are not duplicated here. Conflict resolution preserves the release's
reasoning filtering, image-tool history and one-follow-up vision lifetime.
Replaying these commits and passing focused checks does not qualify either beta
feature for release; cleanup/migration and long-conversation behavior still need
runtime review before promotion.
