"""Saved native compute options, applied before backend initialization."""
import json
import math
from pathlib import Path


CHOICES = {
    'image_dynamic_vram': ('Auto', 'Enabled', 'Disabled'),
    'image_async_offload': ('Auto', 'Disabled', '1', '2', '4'),
    'image_fast_disk': ('Auto', 'Enabled', 'Disabled'),
    'image_vae_device': ('Auto', 'CPU'),
    'image_preview_method': ('auto', 'none', 'latent2rgb', 'taesd'),
}
MEMORY_FIELDS = ('image_reserve_vram', 'image_vram_headroom')


def validate_compute(values):
    result = {key: value for key, value in values.items() if key in CHOICES or key in MEMORY_FIELDS}
    for key, value in result.items():
        if key in CHOICES:
            if value not in CHOICES[key]:
                raise ValueError(f'{key}: choose a listed option.')
        elif value in (None, ''):
            result[key] = None
        else:
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f'{key}: enter a non-negative number or leave blank.') from None
            if isinstance(value, bool) or not math.isfinite(number) or number < 0:
                raise ValueError(f'{key}: enter a finite non-negative number.')
            result[key] = number
    return result


def explicit_options(parser, argv):
    """Recognize argparse's long-option abbreviations and --option=value syntax."""
    known = {option for action in parser._actions for option in action.option_strings}
    result = set()
    for token in argv:
        if token == '--':
            break
        option = token.split('=', 1)[0]
        matches = [name for name in known if name.startswith(option)] if option.startswith('--') else []
        if option in known:
            result.add(option)
        elif len(matches) == 1:
            result.add(matches[0])
    return result


def apply_compute(args, values, options=()):
    """CLI choices win, including opposing flags and explicit zero values."""
    values = validate_compute(values)
    options = set(options)
    for key in MEMORY_FIELDS:
        flag = '--' + key.removeprefix('image_').replace('_', '-')
        if flag not in options and values.get(key) is not None:
            setattr(args, key.removeprefix('image_'), values[key])
    for key, enabled, disabled, conflicts in (
        ('image_dynamic_vram', 'enable_dynamic_vram', 'disable_dynamic_vram',
         ('--cpu', '--directml', '--gpu-only', '--highvram', '--lowvram', '--novram', '--normalvram')),
        ('image_fast_disk', 'fast_disk', 'disable_fast_disk', ()),
    ):
        choice = values.get(key, 'Auto')
        flags = {'--' + name.replace('_', '-') for name in (enabled, disabled)}
        if choice != 'Auto' and not options.intersection(flags | set(conflicts)):
            setattr(args, enabled, choice == 'Enabled')
            setattr(args, disabled, choice == 'Disabled')
    offload = values.get('image_async_offload', 'Auto')
    if offload != 'Auto' and not options.intersection(('--async-offload', '--disable-async-offload')):
        args.disable_async_offload = offload == 'Disabled'
        args.async_offload = None if offload == 'Disabled' else int(offload)
    if values.get('image_vae_device') == 'CPU' and '--gpu-only' not in options:
        args.cpu_vae = True
    if 'image_preview_method' in values and '--preview-method' not in options:
        args.preview_method = values['image_preview_method']


def load_compute(args, parser, argv):
    folder = Path('settings') / args.settings if args.settings not in (None, '', 'default') else Path('settings')
    path = folder / 'settings.json'
    if path.is_file():
        apply_compute(args, json.loads(path.read_text(encoding='utf-8')), explicit_options(parser, argv))
