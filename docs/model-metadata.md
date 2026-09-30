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
