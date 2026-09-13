# Native chat runtime

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
Linux CUDA and other custom builds can use `LLAMA_SERVER`, which overrides the
selector. xllamacpp continues to use the backend supplied by its installed wheel.
These are llama.cpp release versions, not LM Studio runtime version numbers.

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
feedback in its conversation history, so you can request a corrected prompt.
No automatic revision loop runs.

To revise the prompt and generate another image, send **"Generate a new image
based on feedback"**. This explicitly authorizes generation without another
confirmation. The exact wording is not required; an equivalent direct request
such as "Regenerate it with those corrections" has the same intent.
To review the prompt first, ask **"Rewrite the prompt based on the feedback"**.
Prompt-only requests and vision feedback alone do not authorize another image.

Review uses a local JPEG preview up to 1024 pixels per side; originals are kept.
Diffusion models are offloaded before the vision model loads. Review errors
preserve the generated image. References to the latest 64 generated images are
retained in this app session, not across restarts.

With Model Vision enabled, the latest generated prompt and reviewer feedback are
retained as image-task context even when `chat_history` is zero. Clearing the chat
clears this context. Other older conversation turns still follow the history limit.
