"""Preserve image proportions while fitting the model's working pixel budget."""
import math

FRAMING_CHOICES = [('Preserve proportions', 'preserve'), ('Crop to selected size', 'crop')]


def canvas_plan(size, multiple=16):
    width, height = map(int, size)
    if min(width, height, multiple) < 1:
        raise ValueError('Image dimensions must be positive.')
    canvas_w = math.ceil(width / multiple) * multiple
    canvas_h = math.ceil(height / multiple) * multiple
    return dict(width=width, height=height, canvas_width=canvas_w, canvas_height=canvas_h,
                left=(canvas_w-width)//2, top=(canvas_h-height)//2)


def reference_plan(source_size, target_size, mode='preserve', multiple=16):
    if mode not in ('preserve', 'crop'):
        raise ValueError('Choose Preserve proportions or Crop to selected size.')
    source_w, source_h = map(int, source_size)
    target_w, target_h = map(int, target_size)
    if min(source_w, source_h, target_w, target_h, multiple) < 1:
        raise ValueError('Image dimensions must be positive.')
    if mode == 'crop':
        width, height = target_w, target_h
    else:
        scale = min(1.0, math.sqrt(target_w * target_h / (source_w * source_h)))
        width, height = max(1, round(source_w * scale)), max(1, round(source_h * scale))
    return dict(source_width=source_w, source_height=source_h,
                **canvas_plan((width,height),multiple), mode=mode)


def prepare_reference(image, target_size, mode='preserve', *, multiple=16):
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from PIL import Image, ImageOps
    image = ImageOps.exif_transpose(image).convert('RGB')
    plan = reference_plan(image.size, target_size, mode, multiple)
    size = plan['width'], plan['height']
    image = (ImageOps.fit(image, size, Image.Resampling.LANCZOS) if mode == 'crop'
             else image.resize(size, Image.Resampling.LANCZOS))
    pixels = torch.from_numpy(np.asarray(image).copy()).float().unsqueeze(0) / 255.0
    left, top = plan['left'], plan['top']
    right, bottom = plan['canvas_width'] - size[0] - left, plan['canvas_height'] - size[1] - top
    pixels = functional.pad(pixels.movedim(-1, 1), (left, right, top, bottom), mode='replicate').movedim(1, -1)
    return pixels, plan


def restore_reference_frame(pixels, plan):
    if plan is None:
        return pixels
    if tuple(pixels.shape[1:3]) != (plan['canvas_height'], plan['canvas_width']):
        raise ValueError('Decoded image does not match its reference canvas; refusing to stretch it.')
    top, left = plan['top'], plan['left']
    return pixels[:, top:top + plan['height'], left:left + plan['width'], :]
