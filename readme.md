# RuinedFooocus

Local image generation, editing and chat, using a pinned ComfyUI backend and a
shared local llama.cpp runtime. Supports AMD ROCm, NVIDIA CUDA, DirectML and CPU
where the selected model and runtime are compatible.

## Install

RuinedFooocus needs Python 3.11–3.14 (3.10 still works but is deprecated) and Git. The launchers create a
`venv` folder, install the right PyTorch build for your GPU and start the web UI. The first start downloads
several GB of packages and the default model
[sd_xl_base_1.0_0.9vae.safetensors](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors)
into `models/checkpoints`. If you already have it, copy it there first.

### Windows

    git clone https://github.com/johnmart19/RuinedFooocus-ROCm.git RuinedFooocus
    cd RuinedFooocus
    RuinedFooocus.bat

Set `PYTHON` to a python.exe, or activate a virtual environment, to use it instead of `venv`.

### Linux and WSL

    git clone https://github.com/johnmart19/RuinedFooocus-ROCm.git RuinedFooocus
    cd RuinedFooocus
    ./RuinedFooocus.sh

On Ubuntu install `python3-venv` and `git` first; set `PYTHON` to choose the interpreter that creates `venv`. Pass `--listen` (with `--auth user/password`) to open a LAN port.

### GPU backends

The launcher detects the GPU and installs a validated PyTorch bundle:

| GPU                                                                                  | PyTorch                 | Notes                                                                          |
| ------------------------------------------------------------------------------------ | ----------------------- | ------------------------------------------------------------------------------ |
| NVIDIA Maxwell–Pascal                                                                | CUDA 12.6 / 12.4        |                                                                                |
| NVIDIA Turing and newer                                                              | CUDA 13.2 / 13.0 / 12.8 | Newest one your driver supports. `--cuda-nightly` installs CUDA 13.x nightlies |
| AMD RX 5000/6000/7000/9000, Strix Halo (Windows, Linux, WSL), Instinct MI300 (Linux) | ROCm 10.1               | `--rocm10` reinstalls/repairs it                                               |
| Other AMD/Intel GPUs on Windows                                                      | DirectML                | Python 3.10–3.12 only; or `--directml`                                         |
| No supported GPU                                                                     | CPU                     | or `--cpu`                                                                     |

Set `TORCH_PLATFORM` (for example `cu128`, `rocm10.1`, `cpu`) to override detection. Chat models run on the
native llama.cpp server, which picks Vulkan, ROCm or CUDA itself; change it under Settings > llama.cpp backend.
To switch PyTorch later, start with `TORCH_PLATFORM` set or with `--rocm10`, `--cuda-nightly`, `--cpu` or
`--directml`. Create an empty `reinstalltorch` file in the RuinedFooocus folder to reinstall it on the next start,
or an empty `freezetorch` file to stop the launcher from changing PyTorch (for example after installing it
yourself). Deleting the `venv` folder gives a completely fresh install. The launchers replace the
old portable setup helpers and Conda environment file; reinstall/freeze marker files are created locally.

If you see **"MetadataIncompleteBuffer"**, a model file is incomplete or damaged; download it again. **Models > Model > Checkpoint tools** can check a checkpoint for damage.

## Ruined Edition Features

The main destinations are **Create**, **Chat**, **Library**, **Models** and **Settings**. Create puts the image
model and prompt first, with Generation, Prompt tools, Image tools and Details beside the preview.
Settings groups Appearance, Storage, Runtime, Defaults and Integrations. Text size applies immediately
and is remembered by your browser; browser zoom and keyboard navigation remain available.

### Image models

- SD 1.5, SDXL (including Pony, Illustrious and NoobAI), SD3, Flux, Flux 2 and Flux 3, Qwen-Image, HiDream and
  HiDream O1, Lumina 2, Z-Image, Anima, Krea 2, MOSS Image, Fibo 2, Mage Flow, Ernie Image, NewBie Image,
  PixArt, PixelDiT, AuraFlow, Cosmos Predict 2, Boogu and Ideogram 4 (limited). See the [update log](update_log.md).
- Safetensors and GGUF checkpoints, including UNet-only files. **Models > Model > Image model components** lists
  and downloads the text encoders and VAE a checkpoint needs.
  ComfyUI detects checkpoint configuration directly; legacy YAML configs and merge-recipe files are not used.
- **Models > Model > Checkpoint tools** reports a checkpoint's size and tensor types, checks its weights for
  damage (NaN/Inf), and can create a smaller FP16 copy while keeping the original.
  Existing destination files are never overwritten, including on drives without hard-link support.
- If a checkpoint cannot load, the error names the cause: a missing file, a missing text encoder or VAE, a
  damaged or incomplete download, or a file that is not an image checkpoint.
- Model gallery with animated thumbnails, Civitai metadata and trigger words, and subfolders. Files dropped into
  `models/inbox` are identified on Civitai and moved to the right folder.
- Custom model and output folders in `settings/paths.json` (on Windows use `/` or `\\` instead of `\`).

### Video models

- Wan 2.1/2.2, Hunyuan Video, LTX Video (including LTX 2.3 and 2.5) and MiniMax H3. Image to video is in
  **Image tools** when a video model is selected.

### Prompts and styles

- Styles from `settings/styles.csv`: combine several, send a style to the prompt, or add `<style:name>`.
- Wildcards: type `__name__` to insert a random line from `wildcards/name.txt` (see `wildcards/colors.txt`).
- LoRAs from **Models > LoRAs** or with `<lora:name:0.9>` in the prompt. Trigger words are read from a `.txt`
  file next to the LoRA or downloaded from Civitai; the `LoRA keywords` style adds them for you.
- Put `---` on its own line to queue several prompts. Automatic negative prompt is available.
- **Prompt tools** has [One Button Prompt](https://github.com/AIrjen/OneButtonPrompt) for random prompts. Its
  Prompt format follows the checkpoint: tags for SD/SDXL, anime tags for Illustrious/NoobAI/Pony, tags with a
  description for Anima, and natural language for Flux, Qwen and similar models.
  The bundled **Weird Hats** preset keeps existing saved selections and custom edits under its old spelling.
  If both spellings already contain custom presets, each saved selection keeps its own values.
- **Rewrite prompt** and prompt enhancement use the selected local chat model. Rewrite instructions are bundled
  under `ruinedfooocus/resources/prompt_presets`; add your own in `prompt_presets`. Legacy `llamas` entries still work.
- Evolve creates variations of the current prompt (in **Image tools**).

### Image tools

- Image to image with a Denoise slider, and **Native image edit** on checkpoints that can edit images
  (Qwen-Image, Mage Flow, Fibo 2, HiDream O1, MOSS with an edit format, Flux Kontext). Adding an image picks the
  right one automatically; on other checkpoints it falls back to image to image.
- ControlNet (Canny, Depth, Sketch, Recolour) for SDXL, inpainting and background removal (RemBG).
- Upscaling is in **Generation > Size and upscale**.
- Drop an image on the main image to interrogate it (CLIP or Florence) and fill in the prompt.

### Generations and metadata

- Full generation settings are embedded in each PNG. Drag an earlier RuinedFooocus image onto the main window,
  or paste its JSON into the prompt box, to reuse them. **Details** shows the metadata.
- `Ctrl+Enter` generates, `Esc` toggles edit mode, **Cancel** stops, and Image number `0` generates forever.
  A seed of `-1` is random.
- **Image browser** searches earlier generations (or type `search:` in the prompt) and opens a slideshow. The
  **Last image** page in Details always shows the newest image.
- If `notification.mp3` exists in the root folder, it plays when generations finish.

### Chat bots

- Local GGUF chat models on llama.cpp, using Vulkan, ROCm, CUDA or CPU (Settings > llama.cpp backend).
- Assistants are folders with an `info.json` and `avatar.png`, or SillyTavern character cards. Add them in
  `assistants`; legacy `chatbots` entries still work. Bundled assistants live under `ruinedfooocus/resources/assistants`.
  Select **General chat** for a conversation without a character persona.
- Chat can generate images, with an optional reference image that is edited natively when the checkpoint can.
- Optional vision review and a browser Python runner for code from the chat. **Allow image tools** controls
  image generation/editing for the conversation; Force image generation applies to one message.

### Settings and access

- Defaults for the UI in `settings/settings.json`, custom resolutions in `settings/resolutions.json`, custom
  performance presets, Gradio themes and translations (`--language`).
- **Create > Save current defaults** stores the selected model, prompt, styles, LoRAs, size, seed and sampling
  choices for the next launch. It never stores uploads or a previous generated image. Clear an active generation
  preset before saving. Prompt rewrite offers Preview, Apply and Undo, protecting later edits to your prompt.
- Model weights load on demand. **Settings > Runtime > Image compute > Preload selected image model** optionally
  loads the selected image model during startup.
- `--listen` opens the UI on your network and requires `--auth user/password` (or `RF_UI_AUTH`). `--guest`
  starts a restricted, generation-only page for other people. `--nobrowser` skips opening a browser.
- `--api` enables an OpenAI-compatible API; set `--api-key` (or `RF_API_KEY`), which is required when listening
  beyond localhost. `--mcp` starts an MCP server.
- Create an empty file named `reinstall` (all packages) or `reinstalltorch` (PyTorch only) to reinstall on the
  next start, or `freezetorch` to keep the installed PyTorch.

## Thanks

For source locations, UI entry points, backend ownership and user-data compatibility, see the
[application map](docs/architecture.md). `RF_DATA_DIR` selects a separate writable data root; the default
continues to use this clone's data. Source upgrades preserve existing models, outputs and customization.

This codebase is a fork from the original amazing work by [lllyasviel](https://github.com/lllyasviel/Fooocus/discussions/117)
The wonderful [MoonRide](https://github.com/MoonRide303/) who also maintains an amazing fork!
The codebase starts from an odd mixture of [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-ruinedfooocus.ui.app) and [ComfyUI](https://github.com/comfyanonymous/ComfyUI). (And they both use GPL license.)

## Discord Support Server

You can join our discord support server [Here](https://discord.gg/CvpAFya9Rr)

## Update Log

The log is [here](update_log.md).
