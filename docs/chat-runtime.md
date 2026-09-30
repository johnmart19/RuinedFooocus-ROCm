# Chat bots

## Chat models

Under **Chat bots**, choose a character, model and quantization, then click **Load**.
Qwen 2.5 7B is the default. Models are discovered only in RuinedFooocus's configured model folders;
missing files download before loading.
**Import GGUF** copies a file into RuinedFooocus's model folder and keeps the original.
Select the model and click **Load** to load it. WSL uses
the Windows picker; native Linux requires a desktop with Zenity, KDialog or Tk.
**Local** lists installed files; **Cloud** lists files that **Load** can download for local use. After loading, **Unload** releases the model from memory.
The **×** beside a local model opens a confirmation for the selected quantization:
keep its file and remove it from the list, or permanently delete the file too.
Import a hidden file again to restore it to the list.
RF Linux simulates shell output; enter a command rather than a coding question.

## Images from chat

Enable image generation in Settings > Chatbot settings. Ask the chatbot to draw an image; it can pass a visual prompt to the generator using the current Main settings. The chat model is unloaded before image generation to release memory.

Chat bots opens with **Normal** selected. Image generation is a capability of the
conversation, not its default purpose. Greetings, questions, stories and descriptions
receive text responses. Character bots remain available in the selector.

After an image, ask for a change to revise its complete prompt and generate again:
"Update the prompt for Akame to be on the beach in sunglasses in her black fighting
suit." The previous generation prompt stays in chat history; unchanged details
should carry forward. For text without generation, say "Rewrite the prompt only;
do not generate." Change topic normally, for example "Now write a story about her."

**Force image generation** applies to the next message only and resets when sent.
It requires an image-tool call even for a short subject such as "a blue teapot".
This explicit action works even when automatic image-tool selection is disabled
in Settings. The selected model must support function calling. Failed or truncated
tool calls are reported rather than presented as successful images.

Tool selection and prompt quality still depend on the model and its context.
The force option controls the next tool call; it does not guarantee visual identity
or likeness. Keep chat history enabled when revising earlier images.

## Chat reasoning

Show reasoning is off by default. Enable it to display collapsible thoughts when the model returns them. Full conversation state is retained independently of the display.

## Browser Python runner

Turn on **Enable Python runner** below **Show reasoning** to review and run code; it is off by default.
**Load code from chat** refreshes the snippets. Select one by its code preview, then click **Run Python**.
Script input appears for code using `input()`. Empty input supplies a blank answer; **Stop** cancels a run.
The first run downloads [Pyodide](https://pyodide.org/); supported imported packages load automatically.
Runs use a temporary browser filesystem, with a 60-second execution limit and capped output.
Desktop apps, GPU packages and some network requests require a separate local Python environment.

## Native chat runtime

Chat bots use official [llama.cpp b10917](https://github.com/ggml-org/llama.cpp/releases/tag/b10917),
downloaded on first use into `cache/llama.cpp` and verified against its release checksum.
Auto preserves the existing selection: Vulkan on Windows, ROCm on supported Linux
AMD installations, otherwise Vulkan, CPU or the macOS build.
PyTorch and its CUDA/ROCm packages stay unchanged.

**Settings → Chatbot settings → llama.cpp backend** selects an official build:

- Windows x64: CPU, Vulkan, ROCm 10, CUDA 12.4 or CUDA 13.3.
- Linux x64: CPU, Vulkan or ROCm 10; ARM64: CPU or Vulkan.
- macOS: the native build (CPU selection disables GPU layers).

Save settings, then load a model or send a message to unload the previous runtime
and load the selected build. Missing builds download once into separate cache
folders. CUDA downloads its matching DLLs. Windows ROCm reuses a compatible
installed SDK or installs isolated libraries; Linux ROCm uses isolated libraries.
GPU-layer settings and CPU launch mode still apply. CUDA 12.4 remains available
for older NVIDIA GPUs; newer CUDA builds require compatible hardware and drivers.
Linux CUDA can also be built from Settings as described below. Other custom builds
can use `LLAMA_SERVER`, which overrides the selector. xllamacpp continues to use the backend supplied by its installed wheel.
These are llama.cpp release versions, not LM Studio runtime version numbers.

For WSL, keep frequently used GGUF files in the Linux filesystem. A symlink into
`/mnt/c` still reads Windows storage and can make loading much slower even when
CUDA inference works correctly. Copy an existing file into the WSL `models/llm`
folder to avoid another download. See the measured comparison in
[NVIDIA support](nvidia-support.md#linux-and-wsl).

### Build a local CUDA runtime

On Linux/WSL x86_64 with Turing or newer NVIDIA GPUs and a driver supporting CUDA
13.2, open **Settings → Chatbot settings → Build local CUDA runtime**. Click
**Install dependencies and build**. The status box shows dependency installation
and compiler progress. Install host prerequisites first if requested; on Ubuntu:

```sh
sudo apt install build-essential git python3-venv
```

The action installs NVIDIA's CUDA 13.2 toolkit wheels and CMake in a private
environment under `cache/llama.cpp/<version>/local-cuda/build-work`, then builds
the pinned llama.cpp source, including its ggml CUDA backend, for the detected
GPU architecture. It does not replace the app's Torch packages or install a GPU
driver. Compilation needs several GB of disk space and may take several minutes.

After the build reports a usable CUDA device, **CUDA (local build)** becomes
available in **llama.cpp backend**. Select **llama.cpp** as the chat runtime,
choose this backend, save Settings, and load a chat model. The choice is hidden
until the build succeeds and when its executable or CUDA libraries are missing.
Clearing this cache or changing the pinned llama.cpp release requires rebuilding.
Keep the private toolkit alongside the runtime; moving the checkout also requires
rebuilding because the binaries reference its local library path.

This optional builder does not cover older NVIDIA GPUs, Windows, AMD or ARM64.
Their existing runtime choices remain unchanged. In particular, Pascal users
should retain a compatible CUDA 12.x or Vulkan runtime. WSL may expose CUDA
without a usable Vulkan GPU; inspect the runtime's reported devices instead of
assuming that Auto/Vulkan can use the Windows GPU.

Build failures leave the backend unavailable and show the build log location.
The full log is `cache/llama.cpp/<version>/local-cuda/build.log`. Build again after
correcting the reported prerequisite or compiler issue. Repeated clicks are
disabled while a build is running.

### Runtime status and lifecycle

The Chat runtime selector shows the loaded backend, for example `llama.cpp [Vulkan]`.
It updates after loading or unloading; changing an unsaved selection does not change
the active runtime label. `CPU weights` means `n_gpu_layers` is zero. In native
llama.cpp, `-1` uses automatic VRAM fitting; a positive number sets an explicit
layer limit. xllamacpp retains its own `-1` full-offload behavior. Saved values are preserved.
The native runtime keeps upstream automatic Flash Attention, batch sizes and
VRAM margin, with one parallel slot and the configured context length. Extra
arguments can override tuning; CPU mode still prevents GPU offloading. A GPU
launch fails clearly if the chosen binary reports no usable GPU device.
Compare backends using equal token counts and separate loading, prompt processing
and token generation times; a short reply's total time alone is not a speed benchmark.
Setting `chat_history` to zero sends the current message without older conversation turns.

Under **Settings → Chatbot settings**, you can add a custom GGUF path.
Extra llama.cpp arguments can configure
features such as reasoning or KV cache types; quote paths containing spaces.
Use `LLAMA_SERVER` to select your own native build, or choose `xllamacpp` for the
previous Python runtime. Changes apply on the next chat request.
New engine capabilities do not automatically add audio/video controls to this UI.

The native server belongs to the RuinedFooocus process. Unload and normal shutdown
stop it explicitly. Windows uses a kill-on-close Job Object; Linux/WSL uses a
parent-death signal so an abrupt session exit also releases the server's memory.

Under **Installed chat runtimes**, click **Refresh** to query cached llama.cpp
builds (or `LLAMA_SERVER`) and the installed xllamacpp package. The readout shows
versions and devices reported by each runtime, independently of PyTorch's backend.
Probes run in separate processes without model loading or downloads. Missing
libraries or an unavailable driver are reported as unavailable device information;
a successful probe does not replace a generation test.

If a Llama template fails to build its tool parser because it requires a first
user message, the native runtime retries once with tool definitions in the system
message. Tools remain enabled; explicit `chat_template_kwargs` placement is respected.

### Model vision in chat

Enable **Chat bots > Enable Model Vision**, below **Show reasoning**.
The optional **Include system prompt** checkbox appears when vision is enabled and is off by default. Enable it to include the chatbot's character/system instructions as review context. The user's request and actual generation prompt are always reviewed, including character details already written into that generation prompt.
For a catalogued vision-capable chat model, its matching projector is resolved
and downloaded automatically. Otherwise **Select Model** appears for choosing
a separate vision model to inspect images. The main chat selection is unchanged.

The separate `modules/pathdb/llm_vision.json` catalog links SmolVLM 256M,
Qwen2.5-VL 3B and Gemma 3 4B to their matching projectors from ggml-org on
Hugging Face. Downloads use PathManager and its normal progress callbacks.
Vision models and projectors use `models/vision` (configurable as `path_vision`).
They appear only in the vision selector by default. To use one as a regular chat
model, explicitly import its GGUF through Import GGUF. Existing files are reused. Unknown custom model
vision capability is not inferred from its filename; use a catalogued reviewer.

With image generation enabled, the reviewer inspects the generated image and
returns observations under **Vision model feedback:**. The selected chat model
is restored after review, including when the reviewer fails. No automatic reply
or revised image follows. Your next message goes to the main chat model with the
feedback as temporary context, so you can request a corrected prompt.
No automatic revision loop runs.

To revise the prompt and generate another image, send **"Generate a new image
based on feedback"**. This explicitly authorizes generation without another
confirmation. The exact wording is not required; an equivalent direct request
such as "Regenerate it with those corrections" has the same intent.
To review the prompt first, ask **"Rewrite the prompt only; do not generate"**.
Prompt-only requests and vision feedback alone do not authorize another image.

Review uses a local JPEG preview up to 1024 pixels per side; originals are kept.
Diffusion models are offloaded before the vision model loads. Review errors
preserve the generated image. References to the latest 64 generated images are
retained in this app session, not across restarts.

With Model Vision enabled, the latest generated prompt and reviewer feedback are
available for the next user turn, even when `llm_chat_history` is zero. Old review
text and image payloads are omitted from later model input; the visible transcript
is unchanged. A new generated image replaces the previous review context. Clearing
the chat clears its context. Other turns follow the configured history limit.

Development does not summarize conversations or grow the context automatically.
Experimental context compaction remains on the `optional` branch. API clients
manage their own message history; the provider does not silently rewrite it.
