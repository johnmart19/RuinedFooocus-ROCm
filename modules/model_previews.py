"""Decode model artwork before publishing a bounded, animation-safe cache file."""

import io
import json
import os
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

SUFFIXES = (".gif", ".jpeg", ".jpg", ".png", ".webp")
VERSION = 2


def valid_preview(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError, Image.DecompressionBombError):
        return False


def is_warning(path):
    path = Path(path)
    for warning in (Path("html/warning.jpeg"), Path("html/warning.png")):
        if (
            warning.is_file()
            and path.is_file()
            and path.stat().st_size == warning.stat().st_size
            and path.read_bytes() == warning.read_bytes()
        ):
            return True
    return False


def write_preview(data, cache_path, caption="", video=False):
    cache_path = Path(cache_path)
    frames, durations = [], []

    def append(image, duration):
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        if caption:
            ImageDraw.Draw(image).text(
                (3, 3), caption, fill="white", stroke_width=1, stroke_fill="black"
            )
        frames.append(image.copy())
        durations.append(max(20, min(1000, int(duration))))

    try:
        with Image.open(io.BytesIO(data)) as image:
            for index in range(min(getattr(image, "n_frames", 1), 120)):
                image.seek(index)
                append(image, image.info.get("duration", 100))
    except UnidentifiedImageError:
        if not video:
            raise
        import av

        with av.open(io.BytesIO(data)) as container:
            rate = float(container.streams.video[0].average_rate or 12)
            for index, frame in enumerate(container.decode(video=0)):
                if index >= 120:
                    break
                append(frame.to_image(), 1000 / rate)
    if not frames:
        raise ValueError("Preview contains no decodable frames")
    destination = cache_path.with_suffix(".gif" if len(frames) > 1 else ".jpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent, suffix=destination.suffix
    )
    os.close(descriptor)
    try:
        if len(frames) > 1:
            frames[0].save(
                temporary,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=0,
                disposal=2,
                optimize=False,
            )
        else:
            frames[0].save(temporary, format="JPEG", quality=95, subsampling=0)
        os.replace(temporary, destination)
        # Publish first, then remove obsolete formats so the gallery cannot select
        # an old static frame instead of the regenerated animation.
        for suffix in SUFFIXES:
            old = cache_path.with_suffix(suffix)
            if old != destination:
                old.unlink(missing_ok=True)
        cache_path.with_suffix(".preview.json").write_text(
            json.dumps({"version": VERSION}), encoding="utf-8"
        )
        return destination
    finally:
        Path(temporary).unlink(missing_ok=True)
