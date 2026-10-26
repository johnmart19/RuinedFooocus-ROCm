"""Image component inventory and explicit catalog downloads; no tensor runtime."""
from pathlib import Path
import shutil
import sys

from modules.resource_defaults import resource_name
from modules.model_identity import read_safetensors_header
from modules.video_settings import VIDEO_FPS, detect_video_model

# Resource keys mirror the image backend's architecture configurations.
_COMPONENT_KEYS = {
    'Anima': (('clip_qwen3_06b',), 'vae_qwen_image'),
    'AuraFlow': (('clip_aura',), 'vae_auraflow'),
    'BaseModel': (('clip_l',), 'vae_sd'),
    'Boogu': (('clip_qwen3vl_8b_scaled',), 'vae_flux'),
    'CosmosPredict2': (('clip_oldt5',), 'vae_wan'),
    'ErnieImage': (('clip_ministral3',), 'vae_flux2'),
    'Fibo2': (('clip_qwen3vl_4b',), 'vae_flux2'),
    'Flux': (('clip_l', 'clip_t5'), 'vae_flux'),
    'Flux2': (('clip_mistral3',), 'vae_flux2'),
    'Flux2Klein4B': (('clip_qwen3_4b',), 'vae_flux2'),
    'Flux2Klein9B': (('clip_qwen3_8b',), 'vae_flux2'),
    'Flux3': (('clip_qwen3vl_4b',), 'vae_flux2'),
    'HiDream': (('clip_g', 'clip_l', 'clip_llama', 'clip_t5'), 'vae_flux'),
    'HiDreamO1': ((), 'pixel_space'),
    'Ideogram4': (('clip_qwen3vl_8b',), 'vae_flux2'),
    'Krea2': (('clip_qwen3vl_4b_scaled',), 'vae_qwen_image'),
    'Lumina2': (('clip_gemma',), 'vae_lumina2'),
    'MageFlow': (('clip_qwen3vl_4b',), 'vae_mage_flow'),
    'MOSSImage': (('clip_qwen35_9b',), 'vae_flux2'),
    'NewBieImage': (('clip_gemma3', 'clip_jina'), 'vae_flux'),
    'PixArt': (('clip_t5',), 'vae_pixart'),
    'PixelDiTT2I': (('clip_gemma2_it_elm',), 'pixel_space'),
    'QwenImage': (('clip_qwen25',), 'vae_qwen_image'),
    'SD3': (('clip_l', 'clip_g', 'clip_t5'), 'vae_sd3'),
    'SDXL': (('clip_l', 'clip_g'), 'vae_sdxl'),
    'ZImage': (('clip_qwen3_4b',), 'vae_lumina2'),
}
IMAGE_COMPONENTS = {name: {'clip_names': list(clips), 'vae_name': vae}
                    for name, (clips, vae) in _COMPONENT_KEYS.items()}
ALIASES = {'SD 1.5': 'BaseModel', 'SD 2.0': 'BaseModel', 'SDXL 1.0': 'SDXL',
           'Pony': 'SDXL', 'Illustrious': 'SDXL', 'NoobAI': 'SDXL',
           'Pony XL': 'SDXL', 'Illustrious XL': 'SDXL', 'NoobAI XL': 'SDXL',
           'Animagine XL': 'SDXL', 'Animagine XL 3.1': 'SDXL',
           'Flux.1 D': 'Flux', 'Flux.1 S': 'Flux', 'SD 3': 'SD3', 'SD3.5': 'SD3',
           'SD 3.5': 'SD3', 'SD 3.5 Large': 'SD3', 'Qwen': 'QwenImage', 'Qwen-Image': 'QwenImage',
           'MOSS Image': 'MOSSImage', 'MOSS-Image': 'MOSSImage', 'Qwen Image': 'QwenImage',
           'Krea 2': 'Krea2',
           'Mage-Flow': 'MageFlow', 'Mage Flow': 'MageFlow', 'Lumina 2': 'Lumina2', 'Z-Image': 'ZImage',
           'Z-Image Turbo': 'ZImage', 'HiDream O1': 'HiDreamO1', 'FLUX.1 dev': 'Flux',
           'FLUX.1 schnell': 'Flux', 'Flux.2': 'Flux2'}
# Only unambiguous GGUF tags: e.g. lumina2 and qwen_image cover several
# architectures and need the initialized backend's tensor-shape classifier.
GGUF_ARCHITECTURES = {'sdxl': 'SDXL', 'sd3': 'SD3', 'aura': 'AuraFlow',
    'hidream': 'HiDream', 'cosmos': 'CosmosPredict2', 'moss_image': 'MOSSImage',
    'fibo2': 'Fibo2', 'ideogram': 'Ideogram4', 'krea2': 'Krea2',
    'flux3': 'Flux3', 'qwen_image21': 'QwenImage'}


def inventory(checkpoint, family, settings, paths):
    """Resolve local image dependencies. This function never downloads or hashes."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise ValueError('Select an installed image checkpoint')
    tensors = read_safetensors_header(checkpoint) if checkpoint.suffix.lower() == '.safetensors' else None
    detected_video = detect_video_model(tensors) if tensors is not None else None
    if family in VIDEO_FPS or detected_video:
        raise ValueError('This panel supports image checkpoints only')
    gguf_architecture = None
    is_gguf = checkpoint.suffix.lower() == '.gguf'
    if is_gguf:
        # Read shape metadata before choosing dependencies; cached family can be stale.
        import gguf
        reader = gguf.GGUFReader(str(checkpoint))
        try:
            field = reader.get_field('general.architecture')
            arch = field.contents() if field else None
            tensors = {tensor.name: {'shape': list(reversed(tensor.shape.tolist()))} for tensor in reader.tensors}
            if arch in ('wan', 'hyvid', 'ltxv', 'minimax_h3') or detect_video_model(tensors):
                raise ValueError('This panel supports image checkpoints only')
            gguf_architecture = GGUF_ARCHITECTURES.get(arch)
        finally:
            reader.data._mmap.close()
    detected = None
    # The detector creates meta tensors, but imports backend/device modules.
    # Reuse it only in an initialized app; standalone inventory stays Torch-free.
    if tensors is not None and 'comfy.model_detection' in sys.modules:
        from modules.model_architecture import detect_tensor_architecture
        detected = detect_tensor_architecture(tensors)
    if detected in VIDEO_FPS:
        raise ValueError('This panel supports image checkpoints only')
    canonical = (detected or gguf_architecture) if is_gguf else (detected or family)
    architecture = ALIASES.get(canonical, canonical)
    if architecture not in IMAGE_COMPONENTS:
        if is_gguf:
            raise ValueError('GGUF architecture cannot be identified safely; component downloads are disabled')
        raise ValueError('Image architecture is unknown; refresh local model metadata first')
    if tensors is None:
        raise ValueError('Component inspection supports safetensors and image GGUF checkpoints')
    names = list(tensors or {})
    embedded_clip = any(name.startswith(('cond_stage_model.', 'conditioner.embedders.', 'text_encoders.')) for name in names)
    embedded_vae = any(name.startswith('first_stage_model.') for name in names)
    clips, vae = IMAGE_COMPONENTS[architecture]['clip_names'], IMAGE_COMPONENTS[architecture]['vae_name']
    rows = []
    for key in (*clips, *(() if vae == 'pixel_space' else (vae,))):
        category = 'clip' if key.startswith('clip_') else 'vae'
        filename = resource_name(key, settings)
        override = bool(settings.get(key))
        embedded = embedded_clip if category == 'clip' else embedded_vae
        roots = paths.paths.get('path_' + category, [])
        roots = roots if isinstance(roots, list) else [roots]
        roots = [paths.get_abspath(root).resolve() for root in roots]
        found = next((root / filename for root in roots if (root / filename).is_file()), None)
        catalog = next(((name, item) for name, item in paths.DOWNLOADABLE_FILES.items()
                        if item.get('path') == 'path_' + category and item.get('filename') == filename), None)
        download_key = None
        if catalog and roots and not Path(filename).is_absolute():
            target = (roots[0] / filename).resolve()
            if target.is_relative_to(roots[0]) and not embedded and not override:
                download_key = catalog[0]
        rows.append(dict(key=key, filename=filename, category=category, override=override,
            status='Embedded in checkpoint' if embedded else 'Installed' if found else 'Missing',
            location=str(checkpoint if embedded else found or (roots[0] / filename if roots else filename)),
            source=catalog[1].get('url', '') if catalog and not embedded else '',
            bytes=found.stat().st_size if found else None, download_key=download_key,
            download_available=bool(download_key and not found and not embedded)))
    return dict(family=family, architecture=architecture, rows=rows,
        note=('Pixel-space model: no VAE required. ' if vae == 'pixel_space' else '') +
             'Local file presence does not certify model compatibility or verify checksums. '
             'Missing custom overrides require a local file, or reset the component to Automatic in Settings.')


def download_missing(checkpoint, family, settings, paths, *, offline=False, progress=None):
    """Explicit user action. Recompute selected image requirements; accept no client file list."""
    if offline:
        raise ValueError('Offline mode: component downloads are disabled')
    state = inventory(checkpoint, family, settings, paths)
    downloaded = []
    for row in state['rows']:
        if row['download_available']:
            if Path(row['location']).exists():
                continue
            if progress:
                progress(0, desc='Downloading ' + row['filename'])
            def update(received, total, speed):
                if progress:
                    progress(received / total if total else 0, desc='Downloading ' + row['filename'])
            paths.download_file(row['download_key'], progress=update)
            downloaded.append(row['filename'])
    return downloaded


def disk_free(paths):
    root = paths.paths.get('path_clip', [])
    root = root[0] if isinstance(root, list) and root else root
    path = paths.get_abspath(root)
    while not path.exists() and path.parent != path:
        path = path.parent
    return shutil.disk_usage(path).free
