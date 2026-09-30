# Open WebUI setup

RuinedFooocus supplies local models and generation tools. Open WebUI supplies the
chat interface and executes the tools selected for each conversation.

## Start both applications

1. Start RuinedFooocus with `RuinedFooocus.bat --api --port 7863` on Windows,
   or `./RuinedFooocus.sh --api --port 7863` on Linux/WSL.
2. Start your separate Open WebUI installation with `open-webui serve` and open
   `http://localhost:8080`. Do not install it into RuinedFooocus's environment.
3. In RuinedFooocus, download/select a chat model and an image checkpoint. Test
   them in the UI first. Check `http://localhost:7863/v1/models` for API IDs.

For two native Windows services, use the URLs below. Docker clients normally use
`host.docker.internal` instead of `localhost`. Between Windows and WSL, use the
reachable host address if localhost forwarding is unavailable. When exposing
RuinedFooocus beyond loopback, use `--listen` and `--api-key YOUR_KEY`; allow only
the intended network through the firewall. The API key does not protect Gradio UI.

## Connect chat

Open **Admin Settings → Connections → OpenAI** and add:

* URL: `http://localhost:7863/v1`
* API key: your configured key; leave empty for local use without a key

Refresh models and choose a `chat/…` model. Use `chat/` IDs for conversations,
`image/` IDs for images, and `video/` IDs for the video tool. Other IDs are helpers.

To attach images in chat, choose a vision-capable chat model and enable its
**Vision** capability in Open WebUI's model settings if it is not detected.
RuinedFooocus resolves its catalogued projector automatically. Text-only chat
models can instead use the separate `image_review` tool.

## Configure images and editing

Under **Admin Settings → Images** (called **Experience → Images** in some versions):

1. Enable image generation; select the **OpenAI** engine.
2. Set its base URL to `http://localhost:7863/v1`, with the same API key.
3. Select an `image/…` model ID and a resolution supported by that model.
4. For editing, enable **Image Edit**, choose **OpenAI**, and set its separate
   base URL, key and image model to the same local provider.

Editing currently supports one input image through img2img, not masked inpainting
or multi-image composition. Use `/v1/capabilities` to find model presets, resolutions
and available LoRAs. Generation inherits Main settings unless overridden.

## Enable tools in chat

Add an **OpenAPI tool server** with:

* Server URL: `http://localhost:7863/tools`
* Schema URL: `http://localhost:7863/tools/openapi.json`
* Authentication: the same API key, if configured

Select the tools in the chat's **Integrations**, or save them as defaults under
**Workspace → Models → edit model → Tools**. Set function calling to **Native**.
Reload the model/start a new chat after changing defaults. Refresh the tool schema
after upgrading RuinedFooocus so new functions appear.

Available tools:

| Tool | Use |
| --- | --- |
| `image_generation` | Generate with an `image/…` model, optional LoRAs/styles or input image. |
| `video_generation` | Generate with a `video/…` model and duration; optionally supply a first frame. |
| `image_upscale` | Enlarge an image with an `upscaler/…` model. |
| `image_review` | Inspect an image with a `vision/…` helper and return feedback. |
| `available_models` | List usable model IDs before choosing a generation/review model. |
| `generation_settings` | Discover model presets/resolutions, local LoRAs, samplers and schedulers. |

Suggested chat instructions:

> Use the enabled generation tools when I ask for images or videos. Show image
> URLs in Markdown and video URLs as links. Use image_review when I request visual
> feedback, then use its feedback when proposing a corrected prompt. Generate a
> revision when I explicitly ask, for example “Generate a new image based on feedback.”
> Pass any requested resolution in the tool's size field as WIDTHxHEIGHT.
> Otherwise keep the existing defaults.

RuinedFooocus also supplies this guidance to API chat when an image-generation
tool is present. Refresh the OpenAPI tool schema after upgrading. Models follow
these instructions at inference time; they do not learn or train on the tool.

Tools accept inline image data or image URLs returned by RuinedFooocus. The chat
model must have access to the returned URL/tool result. Open WebUI attachments
are converted to inline data for vision chat and image editing; arbitrary external
URLs are not fetched by RuinedFooocus tools.

## Troubleshooting

* **No connection:** verify `/v1/models` from the Open WebUI host; check port,
  Windows/WSL/Docker address, firewall and key.
* **Model not found:** refresh model IDs and choose a downloaded checkpoint/chat
  model. `image/default` requires a valid Main selection.
* **Chat describes a picture but does not generate:** enable a tool for that chat
  and use Native function calling. An Images engine connection alone does not
  enable an OpenAPI tool.
* **Vision unavailable:** select a vision model with a matching projector. First
  use can download components; gated files require Hugging Face authentication.
* **Video not shown inline:** use its returned link/player; Open WebUI's Images
  engine is for still images. Call the video tool instead.
* **Unexpected styles/LoRAs:** clear them in Main, or pass `styles: []` and
  `loras: []` in API image requests.

See [API reference](api.md) for request fields, limits and runtime support.
