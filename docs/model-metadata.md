# Local metadata and previews

Tensor headers identify SDXL/FLUX architecture, not individual fine-tunes.
Metadata and previews belong in `cache/checkpoints` as `model.json` and
`model.jpg` (PNG/GIF also work). Existing `.civitai.info` sidecars and embedded
ModelSpec thumbnails can be read. Re-saving a checkpoint changes its lookup hash.
**Refresh All Files** updates the gallery; `--offline` prevents remote lookups.

Online refresh tries Civitai, then its `.red` endpoint if the primary lookup fails.
If artwork cannot be downloaded, it searches Hugging Face and uses repository images
only after matching the checkpoint SHA-256. The repository's model-card thumbnail is
the fallback when no image files are public. A converted file may not match;
keep its original metadata and preview when converting it.
Under Settings, **Local model metadata only** disables these requests. Save the setting
before refreshing; launch-time offline mode always takes precedence.

## Checkpoint tools

Under **⋮ → PowerUp → Cheat Code**, choose **Checkpoint tools** to inspect the
selected model, check for NaN/Inf weights, or create a separate FP16 copy.
Conversion only reduces FP32 tensors, preserves metadata/previews in the cache, and keeps
the original. It is lossy; test the copy before choosing it for generation.
Checks are local and do not establish a model's training history or quality.
