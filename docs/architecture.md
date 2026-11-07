# Application map

Start with `launch.py` (direct launch) or `entry_with_update.py` (dependency/update
bootstrap). The Windows and Linux launchers select Python and call the latter.
Application code lives under `ruinedfooocus`; downloaded backends live under
`repositories` and are not application source.

| Feature                                      | Source under `ruinedfooocus/`                           | UI entry                                         | Backend                                      |
| -------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------ | -------------------------------------------- |
| Generation, batches, automatic settings      | `features/generation`, `features/models/performance.py` | Create → Generation                              | ComfyUI sampling and model loading           |
| Image editing, masks, ControlNet, geometry   | `features/image_editing`                                | Create → Image tools                             | Native ComfyUI edit/ControlNet nodes         |
| Upscale, background removal, captioning      | `features/image_processing`                             | Create → Generation / Image tools / image upload | ComfyUI upscale nodes; ONNX / caption models |
| Models, LoRAs, components, checkpoint copies | `features/models`                                       | Models                                           | ComfyUI; local safetensors inspection        |
| Prompt transforms, styles, rewrite presets   | `features/prompting`                                    | Create → Prompt tools / Style                    | Shared local chat; ComfyUI conditioning      |
| Random prompts and datasets                  | `features/prompting/one_button`                         | Prompt tools                                     | One Button Prompt                            |
| Optional SuperPrompter                       | `features/prompting/superprompter`                      | One Button prompt enhancer                       | Local Transformers T5                        |
| Local chat, assistants, image tools, vision  | `features/chat`                                         | Chat                                             | Shared llama runtime and image worker        |
| Browser Python runner                        | `features/chat/browser_python.py`, `chat_code.py`       | Chat → Python                                    | Isolated browser sandbox                     |
| Gallery, search, metadata replay             | `features/gallery`                                      | Library / Create → Details                       | SQLite, local images                         |
| Local commands and search routing            | `features/automation`                                   | Prompt commands                                  | Existing feature services                    |
| Retained video code                          | `features/video`                                        | Conditional controls                             | Pinned ComfyUI; outside image acceptance     |
| HTTP/API access                              | `integrations`, `app/api*.py`                           | Startup options                                  | FastAPI, shared serial worker                |
| Runtime selection and installation           | `packaging`, `backends`                                 | Launchers / Settings                             | CUDA, ROCm, DirectML, CPU, llama, ONNX       |

`app` composes services and owns the serial job worker. `core` owns paths,
configuration, process handling and common utilities. `ui` owns the shell, shared
controls and assets. Feature-specific UI stays beside its feature. `backends/comfy`
contains adaptations to the pinned ComfyUI revision in `app/bootstrap.py`; do not
copy native model/sampler implementations into feature folders.

## Resources and user data

`core/paths.py` owns root resolution. Resource discovery does not depend on the
working directory. `resources` and `ui/assets` are bundled inputs. Writable data
defaults to the clone root; `RF_DATA_DIR` selects another data directory for an
extracted or read-only application. Existing absolute model paths remain valid.
Legacy `../models/...` path values retain their former `modules/` base semantics.

| Bundled source                      | User override                                        |
| ----------------------------------- | ---------------------------------------------------- |
| `resources/prompt_presets/*.txt`    | `prompt_presets/`; legacy `llamas/` remains readable |
| `resources/assistants/`             | `assistants/`; legacy `chatbots/` remains readable   |
| `resources/defaults/`               | `settings/` and named settings profiles              |
| One Button bundled datasets/presets | Existing `random_prompt/userfiles/`                  |
| Model download catalogue            | `settings/model_settings.json` for recommendations   |

New user folders override legacy folders, which override bundled entries with the
same relative identity. Discovery never renames, copies or overwrites custom
presets/assistants. Rewrite presets are instructions, assistants are personas,
and GGUF files are model weights; all LLM features share the selected chat runtime.

Keep weights, generated images, caches and environments out of source commits.
Existing `models`, `outputs`, custom wildcards, generation presets and model caches
must not be moved by a source upgrade. Imported settings and PNG metadata retain
their public keys; folder names are internal implementation details.

## Validation and distribution

Developer formatting tools are pinned separately from application dependencies:

```sh
python -m pip install -r tools/format/requirements.txt
npm ci --prefix tools/format
python tools/format_code.py --check
```

Omit `--check` to format. Only tracked files are considered; vendor sources,
tokenizers, prompt text, datasets and user files are excluded. Formatting verifies
that executable Python ASTs remain identical. Ruff also checks syntax and undefined
names. Markdown, JSON, CSS, HTML and standalone JavaScript use Prettier.

Use this checkout's Python environment and pinned `repositories/ComfyUI`.
`python -m unittest discover -s tests` runs the CPU contracts; real generation,
visual quality and GPU stability require separate evidence. Register feature
commits in `tools/feature_commits.json` and validate their declared prerequisites
with `tools/validate_feature_cherry_picks.py`.

Portable releases must build separately for Windows/Linux and CUDA/ROCm. Bundle
an appropriate Python/runtime rather than copying a development venv. Drivers
remain system prerequisites. Preserve writable data on upgrade, keep model
downloads explicit, and bypass Git update behavior for archive builds. A source
or CPU smoke check does not certify a portable GPU distribution.

Task controls are saved with **Create → Save current defaults**; Settings contains persistent app preferences and advanced defaults. Prompt rewrite previews before applying. Model preloading is optional; browsing alone does not require weights to be loaded.
