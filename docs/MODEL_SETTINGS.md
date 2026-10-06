# Checkpoint recommendations

Public cards can supply labelled sampling fields or literal `num_inference_steps`
and `guidance_scale` arguments. Conflicting examples are ignored. For repackaged
weights, refresh can follow up to three declared or linked Hugging Face source
repositories, accepting a creator's settings only when its manifest contains the
same checkpoint SHA-256. Repository rename redirects are supported. Cards without
sampling values continue to use fallback defaults; a successful metadata fetch
does not mean a sampling recommendation was published.

The local checkpoint hash is cached separately from a release's file list, so
another precision or format in the same release cannot supply its identity.

Choose **Model Defaults** under **Performance** to resolve sampling settings for the selected checkpoint. The recipe combines an available family preset, supported settings from the checkpoint's version metadata, and your saved checkpoint override, in that order. The selector contains only Model Defaults and Own settings. Family presets remain internal defaults.

Open **Model Defaults** to refresh the metadata or manage an override. Automatic refresh runs on page load or checkpoint/performance changes while Model Defaults is selected, at most once per day for successful lookups. Failed lookups retain cached settings and retry after five minutes. You can turn automatic refresh off for the current UI session; explicit Refresh still works. Offline and local-metadata modes prohibit both kinds of network lookup.

To save your preferred settings, select **Own settings**, edit steps, CFG, sampler, scheduler and Clip Skip, then choose **Save model settings**. It selects Model Defaults and saves those values for that checkpoint. The source table labels these values **Your override**, only when settings have actually been saved; they take priority over network recommendations. Refreshing recommendations preserves them. **Remove saved settings** restores fetched/family settings. Saved settings survive restarting the application and discovering a public version ID; a different known version does not inherit an old version's settings.

Recommendations fetch Civitai version metadata using discovered public IDs. If no supported version settings are available, a discovered Hugging Face repository can supply its model card and generation-settings JSON. HF lookup requires the cached checkpoint SHA-256 to match a file in the public repository manifest; comparison happens locally, and configuration reads are pinned to that repository revision. Repository-wide recommendations are labelled as such. Identity discovery also works when local artwork already exists.

These lookups use the existing metadata scanner's discovered IDs, repository matches and cached hashes. Recommendation fetching does not perform additional model hashing, upload weights, download weights, write into model folders, execute embedded workflows, or import example-image prompts/settings. Missing verified identity or unsupported card fields retain available local metadata and defaults. Lookup is independent of checkpoint names and the preset catalogue. Recognizing a public model identity and finding settings does not add support for a new inference architecture: generation still requires a compatible backend/pipeline. Unsupported video recipes continue through the existing video controls.

Only supported samplers/schedulers and bounded numeric settings are accepted. Structured generation settings take precedence over explicit labelled creator-card fields. Conflicting alternatives are left unspecified; ranges start at their lower recommended value. Unspecified fields inherit a family preset when one exists, otherwise the existing generic sampling defaults. These are starting points, not a guarantee of image quality. Verify your own settings for families without a specific preset. Resolution, prompts, LoRAs, VAE selection and model sampling shifts remain under their existing controls.

Fetched recommendations and overrides live in the ignored `settings/model_settings.json`, separately from read-only checkpoint directories. Family-level defaults are shared application configuration; individual checkpoint settings and public IDs are not shipped in code or the changelog. The UI and worker/API use the same local recipe resolver; API-only calls consume cached metadata and do not trigger remote refresh.

Checkpoint family detection uses the installed backend’s tensor-shape architecture registry for safetensors and GGUF headers. It allocates shape-only meta tensors, never checkpoint weights. Unrecognized or incomplete headers remain unidentified; backend recognition does not guarantee an application pipeline supports the architecture.

Gallery artwork preserves animated frames and refreshes legacy low-resolution caches from published sources. Repository social cards are not model example images. Missing published artwork keeps the existing placeholder; generations are not automatically promoted to gallery artwork. You can provide a local checkpoint preview using the existing `.preview` image files.
