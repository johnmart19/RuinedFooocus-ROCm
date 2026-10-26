"""Image-only preparation shared by the inpaint editor and pipeline."""
from PIL import Image, ImageOps
import numpy as np


def load_source(value):
    if value is None:
        raise ValueError('Select a source image before enabling inpainting.')
    try:
        if isinstance(value, Image.Image):
            return ImageOps.exif_transpose(value).convert('RGB')
        with Image.open(value) as source:
            return ImageOps.exif_transpose(source).convert('RGB')
    except (OSError, TypeError, ValueError) as error:
        raise ValueError('Select a readable source image for inpainting.') from error


def editor_payload(value):
    image = load_source(value)
    if image.height > 600:
        image = image.resize((max(1, round(image.width * 600 / image.height)), 600),
                             Image.Resampling.LANCZOS)
    return {'background': image, 'layers': [Image.new('RGBA', image.size)], 'composite': None}


def source_and_mask(value, editor):
    source = load_source(value)
    layers = editor.get('layers') if isinstance(editor, dict) else None
    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError('Paint a mask before running inpainting.')
    layer = layers[0]
    if isinstance(layer, Image.Image):
        if layer.mode != 'RGBA':
            raise ValueError('The inpaint mask must have an alpha channel.')
    else:
        pixels = np.asarray(layer)
        if pixels.ndim != 3 or pixels.shape[2] != 4 or pixels.dtype != np.uint8:
            raise ValueError('The inpaint mask must be an RGBA image.')
        layer = Image.fromarray(pixels)
    mask = layer.getchannel('A')
    if mask.getbbox() is None:
        raise ValueError('Paint a mask before running inpainting.')
    return source, mask.resize(source.size, Image.Resampling.LANCZOS)
