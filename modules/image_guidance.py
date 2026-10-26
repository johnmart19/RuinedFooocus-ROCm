"""GPU-free image guidance schema and request-scoped native model patches."""
import math
from modules.image_components import IMAGE_COMPONENTS, ALIASES

MOSS_EDIT_FORMATS = ('Disabled', 'ai-toolkit', 'ai-toolkit (kv_cache off)', 'Moss-Image-Edit')
FIELD_SCHEMA = {
    'moss_image_edit_format': {'choices': MOSS_EDIT_FORMATS, 'default': 'Disabled'},
    'moss_image_edit_strength': {'minimum': 0.0, 'maximum': 100.0, 'step': 0.05, 'default': 1.0},
    'moss_image_slg': {'default': False},
    'sampling_shift': {'minimum': 0.01, 'maximum': 100.0, 'step': 0.01, 'default': None},
    'cfg_norm_strength': {'minimum': 0.0, 'maximum': 100.0, 'step': 0.01, 'default': None},
}
SHIFT_FAMILIES = {
    'Anima': 'AuraFlow', 'AuraFlow': 'AuraFlow', 'Lumina2': 'AuraFlow',
    'ZImage': 'AuraFlow', 'HiDream': 'SD3', 'SD3': 'SD3',
    'NewBieImage': 'AuraFlow', 'QwenImage': 'AuraFlow',
    'MOSSImage': 'AuraFlow', 'MageFlow': 'AuraFlow',
}



def capabilities(family):
    family = ALIASES.get(family, family)
    return {'moss_edit': family == 'MOSSImage', 'moss_slg': family == 'MOSSImage',
            'sampling_shift': family in SHIFT_FAMILIES, 'cfg_norm': family in IMAGE_COMPONENTS,
            'multiple_refs': family in ('MOSSImage', 'MageFlow', 'Fibo2')}


def normalize_reference_images(references, primary, family, edit_format='Disabled'):
    """Accept PIL images/Gallery entries, never paths or remote URLs."""
    from PIL import Image, ImageOps
    if references is None:
        return []
    if not isinstance(references, (list, tuple)) or len(references) > 15:
        raise ValueError('Use at most 15 additional reference images.')
    if not references:
        return []
    family = ALIASES.get(family, family)
    if primary is None:
        raise ValueError('Additional references require a primary input image.')
    if not isinstance(primary, Image.Image):
        raise ValueError('The primary reference must be an image.')
    if family not in ('MageFlow', 'Fibo2', 'MOSSImage') or (family == 'MOSSImage' and edit_format == 'Disabled'):
        raise ValueError('Additional references require native Mage/Fibo2 or enabled MOSS image editing.')
    result = []
    for entry in references:
        image = entry[0] if isinstance(entry, (tuple, list)) and len(entry) == 2 else entry
        if (not isinstance(image, Image.Image) or min(image.size) <= 0
                or image.width * image.height > 16777216):
            raise ValueError('Each additional reference must be an image of at most 16 megapixels.')
        try:
            result.append(ImageOps.exif_transpose(image).convert('RGB'))
        except (OSError, ValueError) as error:
            raise ValueError('Invalid additional reference image.') from error
    return result


def reference_image_map(primary, references, transform):
    return {'image_1': primary, **{f'image_{index}': transform(image)
                                  for index, image in enumerate(references, start=2)}}


def validate_guidance(values):
    result = {key: value for key, value in values.items() if key in FIELD_SCHEMA}
    for key, value in result.items():
        if key == 'moss_image_edit_format':
            if value not in MOSS_EDIT_FORMATS:
                raise ValueError('Choose a supported MOSS Image edit format.')
        elif key == 'moss_image_slg':
            if type(value) is not bool:
                raise ValueError('MOSS skip-layer guidance must be a boolean.')
        else:
            if key in ('sampling_shift', 'cfg_norm_strength') and value in (None, ''):
                result[key] = None
                continue
            if isinstance(value, bool):
                raise ValueError(f'{key}: enter a finite number.')
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f'{key}: enter a finite number.') from None
            spec = FIELD_SCHEMA[key]
            if not math.isfinite(number) or not spec['minimum'] <= number <= spec['maximum']:
                raise ValueError(f'{key}: value must be between {spec["minimum"]} and {spec["maximum"]}.')
            result[key] = number
    return result


def resolve_guidance(request, defaults, family):
    """Missing fields inherit defaults; explicit None/blank means Auto or Off."""
    family = ALIASES.get(family, family)
    values = {key: request[key] if key in request else defaults.get(key, spec['default'])
              for key, spec in FIELD_SCHEMA.items()}
    if family != 'MOSSImage':
        values.update(moss_image_edit_format='Disabled', moss_image_edit_strength=1.0, moss_image_slg=False)
    values = validate_guidance(values)
    if values['sampling_shift'] is not None and family not in SHIFT_FAMILIES:
        raise ValueError('Sampling shift is not supported by this image model family.')
    return values


def request_model(base, values, family, nodes=None):
    """Always start from a clone of the cached, unmodified LoRA-patched model."""
    family = ALIASES.get(family, family)
    model = base.clone()
    if nodes is None:
        from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow, ModelSamplingSD3
        from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple
        from comfy_extras.nodes_cfg import CFGNorm
        nodes = {'AuraFlow': ModelSamplingAuraFlow().patch_aura, 'SD3': ModelSamplingSD3().patch,
                 'slg': SkipLayerGuidanceDiTSimple.execute, 'cfg_norm': CFGNorm.execute}
    if values['sampling_shift'] is not None:
        model = nodes[SHIFT_FAMILIES[family]](model=model, shift=values['sampling_shift'])[0]
    if values['moss_image_slg']:
        model = nodes['slg'](model=model, double_layers='', single_layers='22',
                             start_percent=0.0, end_percent=0.5)[0]
    if values['cfg_norm_strength'] is not None:
        model = nodes['cfg_norm'](model=model, strength=values['cfg_norm_strength'], pre_cfg=False)[0]
    return model


def effective_guidance(values, model, family):
    result = dict(values)
    result['sampling_shift_supported'] = capabilities(family)['sampling_shift']
    sampling = model.get_model_object('model_sampling')
    shift = getattr(sampling, 'shift', None)
    result['effective_sampling_shift'] = (
        float(shift) if isinstance(shift, (int, float)) and math.isfinite(shift) else None)
    return result


def restore_guidance_metadata(metadata, family=None):
    """Restore only validated scalar guidance, never reference pixels or paths.

    Flat fields take precedence over grouped metadata. Recorded finite effective
    shift makes an Auto request reproducible after model defaults change.
    Legacy metadata without guidance contributes no overrides.
    """
    if not isinstance(metadata, dict):
        return {}
    group = metadata.get('image_guidance', {})
    group = group if isinstance(group, dict) else {}
    recorded = {key: metadata[key] if key in metadata else group[key]
                for key in FIELD_SCHEMA if key in metadata or key in group}
    restored = {}
    for key, value in recorded.items():
        try:
            restored.update(validate_guidance({key: value}))
        except ValueError:
            continue  # Invalid metadata is not a trusted generation override.
    supported = metadata.get('sampling_shift_supported', group.get('sampling_shift_supported'))
    can_promote = supported is not False and (family is None or capabilities(family)['sampling_shift'])
    if recorded.get('sampling_shift') in (None, '') and can_promote:
        actual = metadata.get('effective_sampling_shift', group.get('effective_sampling_shift'))
        if actual not in (None, ''):
            try:
                restored.update(validate_guidance({'sampling_shift': actual}))
            except ValueError:
                pass
    return restored


def read_preset_metadata(filename):
    if not filename:
        return {}
    import json
    from PIL import Image
    try:
        with Image.open(filename) as image:
            metadata = json.loads(image.info.get('parameters', ''))
        if not isinstance(metadata, dict) or metadata.get('software') != 'RuinedFooocus':
            return None
        return metadata
    except (OSError, ValueError, TypeError):
        return None


def effective_preset_checkpoint(checkpoint, preset, mode):
    metadata = read_preset_metadata(preset)
    if metadata is None:
        return None
    selected = metadata.get('base_model_name', checkpoint) if preset and mode & 4 else checkpoint
    return selected if isinstance(selected, str) and selected else None
