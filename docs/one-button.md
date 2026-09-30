# One Button prompts

With an empty main prompt, **Generate** builds a scene from the selected One Button
preset before applying styles. Typed prompts are retained unless **Replace typed
prompt** is enabled. **Random Prompt** fills the editor without generating an image;
**Instant OBP** builds the prompt and starts generation.

Choose a prompt format for the checkpoint: **SDXL**, **Illustrious / NoobAI /
Pony**, or **FLUX / SD3.5**. The latter uses the generator's unweighted mode
without random quality filler. This changes prompt construction, not model loading.
It does not convert every legacy template into natural-language prose.

For FLUX, describe the subject, action, setting, lighting and composition clearly;
see the [official prompting guide](https://docs.bfl.ai/guides/prompting_unified_basics).
For anime fine-tunes, use clear character/scene tags and follow the checkpoint
author's instructions for any required quality or trigger tags.

Bundled anime presets use plain illustration wording rather than `key visual`,
which can introduce literal keys, or triple-weighted quality tags. Unchanged old
preset fields migrate on loading; custom text is retained. Already generated
prompts must be regenerated or edited manually.

When an anime prompt uses the cinematic style, its wrapper describes an anime
illustration and no longer adds contradictory medium exclusions. Your explicit
negative prompt is retained. For the least style interference, select **None**.

Prompt changes cannot guarantee better images. Compare the same checkpoint,
seed and sampling settings when evaluating them.

## Prompt enhancement

- **None** uses the procedural generator without another model.
- **SuperPrompt** uses `roborovski/superprompt-v1`, a dedicated T5 prompt expander.
  Sampling probabilities stay in range, and input numbers are preserved.
- **Local chat model** uses the GGUF selected in Chatbot settings (or the default
  Qwen model). It no longer picks a random poet or novelty persona from `llamas/`.
  Its instruction preserves the scene and requests visual tags for anime mode
  or concise descriptive sentences otherwise. Empty, reasoning-only or truncated
  replies fall back to the original scene instead of contaminating the prompt.

The enhancer does not benchmark or automatically choose among installed LLMs.
The local enhancer follows the selected `llama.cpp` or `xllamacpp` runtime.
Both receive the same system/user messages and `max_tokens` request, without
native-server-only options. Prompt cleanup and fallback behavior are shared.
Reasoning models can exhaust the enhancement token budget before answering;
the original scene is retained in that case.

Each bundled preset includes an `enhancement_focus` brief: portraits prioritize
the face, fashion prioritizes garments, landscapes prioritize depth and terrain,
and architecture prioritizes structure and perspective. The local chat enhancer
receives the complete generated scene, including prefix and suffix, and rewrites
it once without appending those fragments again. Explicit scene details take
precedence over the brief. These briefs are not added directly to image prompts
and do not change the SuperPrompt model's input format.

Missing briefs are added to saved bundled presets; existing briefs and other
custom fields are preserved. Custom presets without a brief use the general
scene-preserving instruction.
