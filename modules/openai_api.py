"""Opt-in provider endpoints and an OpenAPI image-generation tool."""

import asyncio
import base64
import json
import mimetypes
import os
import queue
import secrets
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationError


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = "chat/default"
    messages: list[dict] = Field(min_length=1)
    stream: bool = False
    n: Literal[1] = 1

    @field_validator("messages")
    @classmethod
    def text_messages(cls, messages):
        for message in messages:
            if message.get("role") not in ("system", "developer", "user", "assistant", "tool"):
                raise ValueError("Unsupported message role.")
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict) or item.get("type") not in ("text", "image_url"):
                        raise ValueError("Supported content types are text and image_url.")
                    if item["type"] == "text" and not isinstance(item.get("text"), str):
                        raise ValueError("Text content requires a text string.")
                    if item["type"] == "image_url":
                        url = item.get("image_url", {}).get("url") if isinstance(item.get("image_url"), dict) else None
                        if message["role"] != "user" or not isinstance(url, str) or not url.startswith("data:image/"):
                            raise ValueError("Vision inputs must be user messages with base64 image data URLs.")
                        from modules.api_runtime import decode_image
                        decode_image(url)
            elif content is not None and not isinstance(content, str):
                raise ValueError("Message content must be text.")
        return messages


class ControlNetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["canny", "depth", "recolour", "sketch", "img2img"]
    strength: float = Field(default=1, ge=0, le=2)
    start: float = Field(default=0, ge=0, le=1)
    stop: float = Field(default=1, ge=0, le=1)
    edge_low: float = Field(default=0.2, ge=0, le=1)
    edge_high: float = Field(default=0.8, ge=0, le=1)
    denoise: float = Field(default=0.64, gt=0, le=1)


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(default="", max_length=16000,
                        description="Complete visual description of the requested image. Put output dimensions in size, not only in this text.")
    model: str = Field(default="image/default", description="Use image/default for the selected checkpoint, or an image/ ID from available_models. Do not use chat model IDs.")
    n: int = Field(default=1, ge=1, le=4)
    size: str | None = Field(default=None, pattern=r"^(auto|\d+x\d+)$",
        description="Output WIDTHxHEIGHT. Set when the user requests a resolution; otherwise omit to inherit Main's size. Dimensions must be multiples of 8, 64–2048.")
    response_format: Literal["url", "b64_json"] = "url"
    negative_prompt: str | None = None
    performance: str | None = None
    steps: int | None = Field(default=None, ge=1, le=200)
    cfg: float | None = Field(default=None, ge=0, le=30)
    sampler: str | None = None
    scheduler: str | None = None
    clip_skip: int | None = Field(default=None, ge=1, le=12)
    seed: int | None = Field(default=None, ge=-1, le=4294967295)
    user: str | None = None
    input_image: str | None = Field(default=None, max_length=28000000,
                                    description="Base64 image, image data URL, or unexpired image URL returned by this API. No arbitrary URLs or file paths.")
    controlnet: ControlNetRequest | None = None
    styles: list[str] | None = None
    loras: list["LoraRequest"] | None = None


class LoraRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    strength: float = Field(default=1, ge=-2, le=2)


ImageRequest.model_rebuild()


class VideoRequest(ImageRequest):
    model: str = "video/default"
    n: Literal[1] = 1
    duration: float = Field(default=3, ge=0.5, le=10)
    fps: int | None = Field(default=None, ge=8, le=60)


class UpscaleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    input_image: str = Field(max_length=28000000)
    response_format: Literal["url", "b64_json"] = "url"


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    input_image: str = Field(max_length=28000000)
    prompt: str = Field(default="Describe the image and any visible issues.", max_length=16000)
    system_prompt: str = Field(default="", max_length=16000)
    include_system_prompt: bool = False


def create_apps(backend, key=None):
    bearer = HTTPBearer(auto_error=False)
    def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if key and (credentials is None or not secrets.compare_digest(credentials.credentials, key)):
            raise HTTPException(401, "Invalid API key.", headers={"WWW-Authenticate": "Bearer"})
    api = FastAPI(title="RuinedFooocus API", version="1", dependencies=[Depends(authorize)])
    tools = FastAPI(title="RuinedFooocus tools", version="1", dependencies=[Depends(authorize)])
    assets = {}

    def inline_image(value):
        if value and value.startswith(("http://", "https://")):
            path = urlparse(value).path
            token = path.removeprefix("/v1/images/")
            entry = assets.get(token) if path.startswith("/v1/images/") else None
            if entry is None or time.time() - entry[1] > 86400 or not entry[0].is_file():
                raise HTTPException(400, "Use an image data URL or an unexpired image URL returned by this API.")
            if entry[0].stat().st_size > 21000000:
                raise HTTPException(400, "Input image exceeds the size limit.")
            return base64.b64encode(entry[0].read_bytes()).decode()
        return value

    async def http_error(request, error):
        return JSONResponse({"error": {"message": str(error.detail), "type": "invalid_request_error",
                                       "code": error.status_code}}, status_code=error.status_code, headers=error.headers)
    async def validation_error(request, error):
        return JSONResponse({"error": {"message": str(error), "type": "invalid_request_error", "code": 400}}, status_code=400)
    for app in (api, tools):
        app.add_exception_handler(HTTPException, http_error)
        app.add_exception_handler(RequestValidationError, validation_error)

    def select(model, kind):
        try:
            return backend.resolve_model(model, kind)
        except ValueError as error:
            raise HTTPException(404, str(error)) from error

    def submit(operation, *args):
        try:
            return operation(*args)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(429, str(error)) from error

    async def events(job, request):
        deadline = time.monotonic() + 1800
        try:
            while True:
                if await request.is_disconnected():
                    return
                if time.monotonic() >= deadline:
                    raise HTTPException(504, "Generation timed out.")
                try:
                    kind, value = await asyncio.to_thread(job.events.get, True, 0.25)
                except queue.Empty:
                    continue
                if kind == "error":
                    raise HTTPException(500, value)
                if kind == "invalid_request":
                    raise HTTPException(400, value)
                yield kind, value
                if kind == "result":
                    return
        finally:
            job.cancelled.set()

    @api.get("/models")
    @tools.get("/models", operation_id="available_models")
    def list_models():
        return {"object": "list", "data": [dict(id=name, object="model", created=0,
                owned_by="ruinedfooocus", type=info["kind"]) for name, info in backend.models().items()]}

    @api.get("/capabilities")
    @tools.get("/capabilities", operation_id="generation_settings")
    def capabilities():
        return backend.capabilities()

    @api.post("/chat/completions")
    async def chat_completions(body: ChatRequest, request: Request):
        model, selected = select(body.model, "chat")
        payload = body.model_dump(exclude_none=True)
        payload["model"] = model
        job = submit(backend.chat, payload, selected)
        if body.stream:
            async def stream():
                try:
                    async for kind, value in events(job, request):
                        if kind == "chunk":
                            yield "data: " + json.dumps(value) + "\n\n"
                except HTTPException as error:
                    yield "data: " + json.dumps({"error": {"message": error.detail, "code": error.status_code}}) + "\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
        async for kind, value in events(job, request):
            if kind == "result":
                return value
        raise HTTPException(499, "Client disconnected.")

    async def generate(body, request, kind="image"):
        _, selected = select(body.model, kind)
        operation = backend.upscale if kind == "upscaler" else backend.images
        payload = body.model_dump(exclude_none=True)
        if payload.get("input_image"):
            payload["input_image"] = inline_image(payload["input_image"])
        job = submit(operation, payload, selected)
        async for kind, value in events(job, request):
            if kind != "result":
                continue
            data = []
            for filename in value:
                path = Path(filename)
                if body.response_format == "b64_json":
                    data.append({"b64_json": base64.b64encode(path.read_bytes()).decode()})
                else:
                    now = time.time()
                    for token in list(assets):
                        if now - assets[token][1] > 86400:
                            assets.pop(token)
                    while len(assets) >= 256:
                        assets.pop(next(iter(assets)))
                    token = secrets.token_urlsafe(24)
                    assets[token] = (path, now)
                    # Both apps live on the same host; media is served by /v1.
                    url = str(request.url.replace(path="/v1/images/" + token, query=""))
                    data.append({"url": url})
            return {"created": int(time.time()), "data": data}
        raise HTTPException(499, "Client disconnected.")

    @api.post("/images/generations")
    async def image_generations(body: ImageRequest, request: Request):
        return await generate(body, request)

    @api.post("/images/edits")
    async def image_edits(request: Request):
        """Edit one uploaded image using the local image-to-image workflow."""
        async with request.form(max_files=2, max_fields=20, max_part_size=28000000) as form:
            uploads = form.getlist("image") + form.getlist("image[]")
            if len(uploads) != 1 or not hasattr(uploads[0], "read"):
                raise HTTPException(400, "Upload exactly one image using image or image[].")
            if "mask" in form:
                raise HTTPException(400, "Mask editing is not supported by this endpoint; use image-to-image.")
            raw = await uploads[0].read(21000001)
            if len(raw) > 21000000:
                raise HTTPException(400, "Input image exceeds the size limit.")
            values = {name: value for name, value in form.items() if name not in ("image", "image[]")}
            for name in ("styles", "loras", "controlnet"):
                if name in values:
                    try:
                        values[name] = json.loads(values[name])
                    except (TypeError, ValueError) as error:
                        raise HTTPException(400, f"{name} must be JSON.") from error
            values["input_image"] = base64.b64encode(raw).decode()
            try:
                body = ImageRequest.model_validate(values)
            except ValidationError as error:
                raise HTTPException(400, str(error)) from error
            return await generate(body, request)

    @tools.post("/image_generation", operation_id="image_generation")
    async def image_generation(body: ImageRequest, request: Request):
        """Create an image now when the user asks to draw, generate or regenerate one.

        Supply a visual prompt and pass any requested resolution in size as WIDTHxHEIGHT.
        Otherwise keep defaults. No extra confirmation is needed for a generation request.
        Do not generate for prompt-only requests. Display the returned URL as a Markdown image.
        """
        return await generate(body, request)

    @api.post("/videos/generations")
    @tools.post("/video_generation", operation_id="video_generation")
    async def video_generation(body: VideoRequest, request: Request):
        """Generate a video using a video model ID; optional input_image is the first frame."""
        return await generate(body, request, "video")

    @api.post("/images/upscale")
    @tools.post("/image_upscale", operation_id="image_upscale")
    async def image_upscale(body: UpscaleRequest, request: Request):
        """Upscale an inline image using a known upscaler model ID."""
        return await generate(body, request, "upscaler")

    @api.post("/images/review")
    @tools.post("/image_review", operation_id="image_review")
    async def image_review(body: ReviewRequest, request: Request):
        """Inspect an inline image with a vision model and return feedback, without generating a revision."""
        _, selected = select(body.model, "vision")
        payload = body.model_dump()
        payload["input_image"] = inline_image(payload["input_image"])
        job = submit(backend.review, payload, selected)
        async for kind, value in events(job, request):
            if kind == "result":
                return {"feedback": value}
        raise HTTPException(499, "Client disconnected.")

    # Opaque expiring URLs can be rendered by clients that cannot attach API headers.
    # The mapping contains only files generated by this API, never caller-supplied paths.
    async def image_file(token: str):
        entry = assets.get(token)
        if entry is None or time.time() - entry[1] > 86400 or not entry[0].is_file():
            raise HTTPException(404, "Image URL expired or was not found.")
        return FileResponse(entry[0], media_type=mimetypes.guess_type(entry[0].name)[0] or "application/octet-stream")
    # A separate router avoids inheriting the bearer dependency for capability URLs.
    from fastapi import APIRouter
    media = APIRouter()
    media.add_api_route("/images/{token}", image_file, methods=["GET"], include_in_schema=False)
    api.router.routes.extend(media.routes)
    return api, tools


def validate_options(args):
    key = args.api_key or os.environ.get("RF_API_KEY")
    listen = args.listen or os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    if args.api and (args.share or listen not in ("localhost", "127.0.0.1", "::1")) and not key:
        raise ValueError("--api with --listen or --share requires --api-key or RF_API_KEY.")
    return key


def install(app, key=None):
    from modules import api_runtime
    api, tools = create_apps(api_runtime, key)
    app.mount("/v1", api)
    app.mount("/tools", tools)
