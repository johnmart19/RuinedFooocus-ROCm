# API

Start with `RuinedFooocus.bat --api` on Windows or `./RuinedFooocus.sh --api`
on Linux. The API uses the same port as the UI (7860 by default; use `--port`
to change it). Examples below use port 7863.

`--api` checks `requirements_api_versions.txt` and installs missing dependencies.
Open WebUI is a separate client and is not installed into RuinedFooocus.

Set `RF_API_KEY` or pass `--api-key` to require a bearer key. A key is required
when using `--listen` beyond localhost or `--share`. The key protects the new
API endpoints; it does not add authentication to the existing Gradio UI.
`GRADIO_SERVER_NAME` is also checked when it supplies the listening address.

## Open WebUI

See the [Open WebUI setup guide](open-webui.md) for connections,
image editing, tools and troubleshooting.

## Endpoints

Interactive documentation: `/v1/docs` and `/tools/docs`.

| Route | Purpose |
| --- | --- |
| `GET /v1/models` | Local chat/image/video models and catalogued vision helpers/upscalers, identified by `type`. |
| `GET /v1/capabilities` | Model-specific performance presets, resolutions, fixed settings, local LoRAs, samplers and schedulers. |
| `POST /v1/chat/completions` | OpenAI chat completions, including SSE streaming. |
| `POST /v1/images/generations` | Generate images, returning URLs or `b64_json`. |
| `POST /v1/images/edits` | Multipart single-image img2img, compatible with Open WebUI editing. |
| `POST /v1/images/upscale` | Upscale an input image using an `upscaler/…` ID. |
| `POST /v1/images/review` | Vision feedback using a `vision/…` ID; no automatic revision. |
| `POST /v1/videos/generations` | RuinedFooocus video extension, using a `video/…` ID. |
| `POST /tools/image_generation` | The same image operation for OpenAPI tool clients. |
| `POST /tools/video_generation` | Video generation tool. |
| `POST /tools/image_upscale` | Upscaling tool. |
| `POST /tools/image_review` | Vision review tool. |
| `GET /tools/models`, `GET /tools/capabilities` | Model and settings discovery for tool clients. |

Download or import models in RuinedFooocus first. The API only loads known local
models; it does not accept arbitrary file paths. Known vision helpers, projectors,
upscalers and required checkpoint components can download on first use through
PathManager. Gated downloads require the same Hugging Face authentication as the UI.
`chat/default` uses the configured default chat model. `image/default` uses the
checkpoint selected in Main. Explicit IDs from `/v1/models` select another model.

Example chat body:

```json
{
  "model": "chat/default",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": true
}
```

Example image body:

```json
{
  "model": "image/default",
  "prompt": "A lighthouse above a calm sea at sunrise",
  "size": "1024x1024",
  "n": 1,
  "response_format": "b64_json"
}
```

Image requests inherit Main's performance, resolution, styles and LoRAs at
submission. Optional `performance`, `steps`, `cfg`, `sampler`, `scheduler`,
`clip_skip`, `negative_prompt`, `seed` and `size` override those defaults.
`styles: []` and `loras: []` explicitly disable inherited styles and LoRAs.
Select LoRAs by local name, for example
`"loras": [{"name": "3dsrx_1250.safetensors", "strength": 0.4}]`.
Use only adapters compatible with the chosen diffusion model; these are not chat-model LoRAs.
Dimensions must be multiples of eight, between 64 and 2048.
Pass an explicitly requested resolution as `size: "WIDTHxHEIGHT"`; otherwise
omit `size` to keep Main's defaults. Up to four images
can be requested together. Returned image URLs expire after 24 hours or when
evicted from the 256-image URL cache; files remain in the output folder.

Chat messages, reasoning and tool calls pass through the configured llama.cpp or xllamacpp
runtime. Clients supply tool definitions and execute returned tool calls;
RuinedFooocus does not execute arbitrary client tools. Register the OpenAPI image
tool to let a chat client request images. User messages also accept OpenAI
`image_url` content blocks containing base64 image data URLs. Select a vision-capable
chat model; its matching projector is loaded automatically. Remote URLs and arbitrary
local paths are rejected. `max_completion_tokens` maps to the runtime's `max_tokens`.
The API is stateless: clients must resend conversation history, tool results and
vision feedback in subsequent turns. Audio and the Responses API are not implemented.

API work shares the UI's GPU queue. Image generation unloads the chat model when
necessary; the next chat request reloads it. Eight API requests can be pending
or running at once. Requests time out after 30 minutes. Existing Gradio and MCP
interfaces remain available.
# ControlNet and image-to-image

`POST /v1/images/generations` and `/tools/image_generation` also accept
`input_image` (base64 PNG/JPEG, an image data URL, or this API's image URL) and `controlnet`:

```json
{
  "model": "image/default",
  "prompt": "a stone cottage in a forest",
  "input_image": "<base64 image>",
  "controlnet": {"type": "canny", "strength": 0.8, "start": 0, "stop": 0.8}
}
```

Supported types are `canny`, `depth`, `recolour`, `sketch` (bundled SDXL weights)
and `img2img`. For `img2img`, set `denoise` in `(0, 1]` (default 0.64).
Canny computes edges using `edge_low` and `edge_high` (defaults 0.2/0.8).
Depth and sketch require a prepared control image; there is no automatic depth
estimator. Input images are limited to 16 megapixels. Arbitrary remote URLs and
local file paths are not accepted. These are RuinedFooocus extensions to the image endpoint.

An input image without `controlnet` uses conservative img2img with denoise 0.25.
Empty prompts are accepted with an input image and do not request random prompting.
Multipart edits accept one `image` or `image[]` upload plus scalar generation fields.
Masks and multiple input images are rejected rather than silently ignored.

## Video, upscaling and review

These are RuinedFooocus extensions, not OpenAI's asynchronous video-job protocol.
Video requests use the image request fields plus `duration` (0.5–10 seconds,
default 3) and optional `fps` (8–60); `n` must be 1. Use a `video/…` ID.
`input_image` supplies the first frame for models that support image-to-video.
The selected family's presets, resolutions and fixed sampling settings apply.
ControlNet is not accepted for video. Output format follows the existing video
pipeline (MP4/GIF); media URLs return the correct content type.

Upscaling accepts `model`, `input_image` and `response_format`. Choose an
`upscaler/…` ID; the model determines the scale factor.

Review accepts `model` (a `vision/…` ID), `input_image`, and `prompt` describing
what to inspect. Optional `system_prompt` is included only when
`include_system_prompt` is true. It returns `{"feedback": "…"}`, releases the
reviewer and restores the previously loaded chat model, if any.

Generation, review and upscale tools accept base64 input or an unexpired image
URL returned by this API. URLs from other servers are not fetched. A video URL
cannot be used as a still-image input. Clients should render returned images
with Markdown and videos with a link/player, then pass review feedback back to
the chat model before requesting a corrected prompt or another generation.

## Runtime compatibility

The API uses the same serial worker, pipelines and configured runtimes as the UI;
it does not install a separate GPU stack. Windows and WSL use the same routes.
CUDA/ROCm/CPU/DirectML apply to diffusion through the installed ComfyUI/PyTorch
environment. Chat uses the selected llama.cpp/xllamacpp backend, such as Vulkan,
CUDA, ROCm or CPU. DirectML is not a llama.cpp backend. Hardware and model support
remain subject to the installed runtime; an API endpoint cannot make an unsupported
model or GPU compatible. Keep large GPU requests sequential.

Validation on Windows covered Open WebUI chat/vision, streamed tool calls,
image generation and multipart editing, plus the upscaling/review tools and a
short Wan video request. Inline vision messages were also checked on CPU with
both native llama.cpp and xllamacpp. This does not establish NVIDIA, WSL, DirectML,
every model family, or long-video quality validation.
