import gc
import numpy as np
import os
import torch
import traceback

import ruinedfooocus.features.image_editing.controlnet as control_tools
import ruinedfooocus.app.async_worker as worker
import ruinedfooocus.features.prompting.prompt_processing as pp

from PIL import Image
from ruinedfooocus.features.models.model_identity import file_signature, files_unchanged

# Known models (from ComfyUI/comfy/model_base.py)
from comfy.model_base import Flux, SDXL
from ruinedfooocus.app.state import path_manager, settings
import ruinedfooocus.app.state as shared
from pathlib import Path
import random

import comfy.utils
import comfy.model_management
from comfy.sd import load_checkpoint_guess_config

from nodes import (
    CLIPTextEncode,
    CLIPSetLastLayer,
    ControlNetApplyAdvanced,
    EmptyLatentImage,
    VAEEncode,
    VAEEncodeForInpaint,
)

from comfy_extras.nodes_sd3 import EmptySD3LatentImage
from comfy_extras.nodes_flux import FluxKontextImageScale, EmptyFlux2LatentImage
from comfy_extras.nodes_hunyuan import EmptyHunyuanImageLatent
from comfy_extras.nodes_hidream_o1 import EmptyHiDreamO1LatentImage, HiDreamO1ReferenceImages
from comfy_extras.nodes_chroma_radiance import EmptyChromaRadianceLatentImage
from comfy_extras.nodes_edit_model import ReferenceLatent
from comfy_extras.nodes_qwen import TextEncodeQwenImageEdit
from comfy_extras.nodes_mage import TextEncodeMageFlowEdit
from comfy_extras.nodes_fibo2 import TextEncodeFibo2
from comfy_extras.nodes_moss_image import EDIT_FORMATS, TextEncodeMOSSImageEdit
from ruinedfooocus.features.image_editing.image_guidance import (resolve_guidance, request_model, effective_guidance,
                                    normalize_reference_images, reference_image_map)
from node_helpers import conditioning_set_values

from comfy.sample import fix_empty_latent_channels
from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels
from comfy_extras.nodes_canny import Canny
from comfy.sd import CLIP, VAE
from ruinedfooocus.backends.comfy.sequence_prefill import install_qwen35_prefill

install_qwen35_prefill()

from ruinedfooocus.features.generation.pipeline_utils import (
    get_previewer,
    clean_prompt_cond_caches,
    load_diffusion_model,
)
from ruinedfooocus.features.image_editing.canny_utils import sanitize_canny_thresholds
from ruinedfooocus.features.image_editing.image_geometry import prepare_reference, restore_reference_frame

from molbal_comfyui_gguf.nodes import gguf_sd_loader as load_gguf_sd, DualCLIPLoaderGGUF


def reference_tensor(image, width=None, height=None):
    pixels = torch.from_numpy(np.array(image.convert('RGB')).astype(np.float32) / 255.0)[None,]
    if width is not None:
        pixels = ImageScaleToTotalPixels().execute(image=pixels, upscale_method='bicubic',
            megapixels=float(width) * float(height) / (1024 * 1024), resolution_steps=1)[0]
    return pixels


class pipeline:
    pipeline_type = ["sdxl", "ssd", "sd3", "flux", "flux2", "lumina2"]


    class StableDiffusionModel:
        def __init__(self, unet, vae, clip, clip_vision):
            self.unet = unet
            self.vae = vae
            self.clip = clip
            self.clip_vision = clip_vision

    # Add new model setups here.
    # Simple workflows with "Load models"->"Sample"->"VAE" might work right away.
    # Add new clip/vae models to settings and pathdb-files.
    known_model_info = {
        "Anima": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.COSMOS,
            "shift_setting": "anima_image_shift",
        },
        "AuraFlow": {
            "clip_type": comfy.sd.CLIPType.STABLE_DIFFUSION,
            "shift_setting": "auraflow_shift",
        },
        "BaseModel": {
            "clip_type": comfy.sd.CLIPType.STABLE_DIFFUSION,
        },
        "Boogu": {
            "clip_type": comfy.sd.CLIPType.BOOGU,
        },
        "CosmosPredict2": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.COSMOS,
        },
        "ErnieImage": {
            "latent": "FLUX2",
            "clip_type": comfy.sd.CLIPType.FLUX2,
        },
        "Fibo2": {
            "latent": "FLUX2",
            "clip_type": comfy.sd.CLIPType.FIBO2,
            "options": {"has_image_edit": True}
        },
        "Flux": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.FLUX,
        },
        "Flux2": {
            "latent": "FLUX2",
            "clip_type": comfy.sd.CLIPType.FLUX2,
        },
        "Flux2Klein4B": {
            "latent": "FLUX2",
            "clip_type": comfy.sd.CLIPType.FLUX2,
        },
        "Flux2Klein9B": {
            "latent": "FLUX2",
            "clip_type": comfy.sd.CLIPType.FLUX2,
        },
        "Flux3": {
            "latent": "FLUX2",
            "clip_type": comfy.sd.CLIPType.FLUX3,
        },
        "HiDream": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.HIDREAM,
            "shift_setting": "hidream_shift",
        },
        "HiDreamO1": {
            "latent": "HiDreamO1",
            "clip_type": comfy.sd.CLIPType.HIDREAM,
        },
        "Ideogram4": {
            "latent": "FLUX2",
            "clip_type": comfy.sd.CLIPType.IDEOGRAM4,
        },
        "Krea2": {
            "clip_type": comfy.sd.CLIPType.KREA2,
        },
        "Lumina2": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.LUMINA2,
            "shift_setting": "lumina2_shift",
        },
        "MageFlow": {
            "latent": "MAGE",
            "clip_type": comfy.sd.CLIPType.MAGE,
            "options": {"has_image_edit": True}
        },
        "MOSSImage": {
            "latent": "FLUX2",
            "clip_type": comfy.sd.CLIPType.MOSS_IMAGE,
        },
        "NewBieImage": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.NEWBIE,
            "shift_setting": "newbieimage_shift",
        },
        "PixArt": {
            "clip_type": comfy.sd.CLIPType.PIXART,
        },
        "PixelDiTT2I": {
            "latent": "ChromaRadience",
            "clip_type": comfy.sd.CLIPType.PIXELDIT,
        },
        "QwenImage": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.QWEN_IMAGE,
            "shift_setting": "qwen_image_shift",
            "options": {"has_image_edit": True}
        },
        "SD3": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.SD3,
            "shift_setting": "sd3_shift",
        },
        "SDXL": {
            "clip_type": comfy.sd.CLIPType.STABLE_DIFFUSION,
        },
        "ZImage": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.LUMINA2,
            "shift_setting": "lumina2_shift",
        },
    }
    known_models = known_model_info.keys()

    xl_base: StableDiffusionModel = None
    xl_base_hash = None

    xl_base_patched: StableDiffusionModel = None
    xl_base_patched_hash = None

    xl_controlnet: StableDiffusionModel = None
    xl_controlnet_hash = None

    model_info = None
    # Why the last checkpoint load failed, shown to the user instead of a generic error.
    load_error = None

    def get_clip_and_vae(self, unet):
        from ruinedfooocus.features.models.resource_defaults import resource_name
        unet_type = unet.model.__class__.__name__
        config = unet.model.model_config.unet_config
        # ComfyUI already classified these variants before constructing the model.
        if unet_type == "Lumina2":
            if config.get('dim') == 3840:
                unet_type = "ZImage"
            elif 'clip_text_dim' in config:
                unet_type = "NewBieImage"
        elif unet_type == "Flux2":
            unet_type = {20: 'Flux2Klein4B', 24: 'Flux2Klein9B'}.get(config.get('depth_single_blocks'), 'Flux2')

        if unet_type not in self.known_model_info.keys():
            raise RuntimeError(
                f"No default text encoder/VAE configuration for backend model {unet_type}. "
                "This architecture needs an explicit supported component configuration."
            )

        info = self.known_model_info[unet_type].copy()
        shift = settings.default_settings.get(info.pop('shift_setting', None))
        info['sampling_shift'] = float(shift) if shift not in (None, '') else None
        from ruinedfooocus.features.models.image_components import IMAGE_COMPONENTS
        info.update(IMAGE_COMPONENTS[unet_type])
        info["clip_names"] = [resource_name(key, settings.default_settings) for key in info["clip_names"]]
        if info["vae_name"] != "pixel_space":
            info["vae_name"] = resource_name(info["vae_name"], settings.default_settings)
        return info

    def load_base_model(self, name, unet_only=False, input_unet=None, hash=None, input_components=None):
        from ruinedfooocus.features.models.resource_defaults import RESOURCE_DEFAULTS
        embedded = input_components or {}
        components = {key: value for key, value in settings.default_settings.items()
                      if key in RESOURCE_DEFAULTS}
        filename = shared.models.get_model_path("checkpoints", name, hash=hash, default=None)
        signature = file_signature(filename) if filename is not None else None
        if (self.xl_base_hash is not None
                and self.xl_base_hash in (name, hash)
                and signature is not None and signature == getattr(self, 'base_file_signature', None)
                and components == getattr(self, "component_settings", None)
                and files_unchanged(getattr(self, 'component_file_signatures', None))):
            return
        self.component_settings = components
        component_files = []

        self.xl_base = None
        self.xl_base_hash = None
        self.xl_base_patched = None
        self.xl_base_patched_hash = None
        self.conditions = None
        self.model_info = None
        self.load_error = None

        if filename is None:
            print(f"ERROR: Could not load checkpoint {name}")
            self.load_error = (f"Checkpoint not found: {name}. It may have been moved or renamed; "
                               "choose another checkpoint in the Models tab.")
            return

        if input_unet is None: # Be quiet if we already loaded a unet
            print(f"Loading base {'unet' if unet_only else 'model'}: {name}")

        gc.collect(generation=2)

        comfy.model_management.cleanup_models()
        comfy.model_management.soft_empty_cache()

        unet = None

        filename = str(filename)
        if Path(filename).suffix.lower() == ".gguf" or unet_only:
            with torch.inference_mode():
                try:
                    unet = load_diffusion_model(filename, input_unet)

                    # Get text-encoders (clip) and vae to match the unet
                    model_info = self.get_clip_and_vae(unet)
                    self.model_info = model_info

                    # Load everything...
                    clip_paths = []
                    for clip_name in ([] if "clip" in embedded else model_info['clip_names']):
                        clip_paths.append(
                            str(
                                path_manager.get_folder_file_path(
                                    "clip",
                                    clip_name,
                                    default = os.path.join(path_manager.model_paths["clip_path"], clip_name)
                                )
                            )
                        )

                    missing = [Path(path).name for path in clip_paths if not Path(path).is_file()]
                    if missing:
                        raise ValueError(f"{name} needs text encoder file(s) that are not installed: "
                                         f"{', '.join(missing)}. Download them in Models > Model > Image model components.")
                    print(f"Loading CLIP: {model_info['clip_names']}")
                    component_files.extend(file_signature(path) for path in clip_paths)
                    if "clip" in embedded:
                        clip = embedded["clip"]
                    elif all(name.endswith(".safetensors") for name in clip_paths):
                        model_options = {}
                        device = comfy.model_management.get_torch_device()
                        if device.type == "cpu":
                            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
                        clip = comfy.sd.load_clip(ckpt_paths=clip_paths, clip_type=model_info['clip_type'], model_options=model_options)
                    else:
                        clip_loader = DualCLIPLoaderGGUF()
                        clip = clip_loader.load_patcher(
                            clip_paths,
                            model_info['clip_type'],
                            clip_loader.load_data(clip_paths)
                        )

                    if "vae" in embedded:
                        vae = embedded["vae"]
                    else:
                        metadata = None
                        if model_info['vae_name'] == "pixel_space":
                            sd = {"pixel_space_vae": torch.tensor(1.0)}
                        else:
                            vae_path = path_manager.get_folder_file_path(
                                "vae", model_info['vae_name'],
                                default=os.path.join(path_manager.model_paths["vae_path"], model_info['vae_name'])
                            )
                            if not Path(vae_path).is_file():
                                raise ValueError(f"{name} needs the VAE file {Path(vae_path).name}, which is not "
                                                 "installed. Download it in Models > Model > Image model components.")
                            print(f"Loading VAE: {model_info['vae_name']}")
                            component_files.append(file_signature(vae_path))
                            if str(vae_path).endswith(".gguf"):
                                sd, extra = load_gguf_sd(str(vae_path), handle_prefix=None)
                                metadata = extra.get("metadata", {})
                            else:
                                sd, metadata = comfy.utils.load_torch_file(str(vae_path), return_metadata=True)
                        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
                        if model_info['vae_name'] != "pixel_space" and not str(vae_path).endswith(".gguf"):
                            vae.patcher.cached_patcher_init = (comfy.sd.load_vae_patcher, (str(vae_path), metadata, None))
                    vae.throw_exception_if_invalid()

                    clip_vision = None
                except Exception as e:
                    unet = None
                    traceback.print_exc()
                    from ruinedfooocus.core.task_errors import describe_error
                    self.load_error = describe_error(e) if isinstance(e, ValueError) else \
                        f"Could not load {name}. {describe_error(e)}"

        else:
            # The file loader preserves quantization metadata and registers the
            # reload factories required by dynamic patchers and their delegates.
            unet, clip, vae, clip_vision = load_checkpoint_guess_config(filename)
            valid_vae = isinstance(vae, VAE) and vae.first_stage_model is not None
            if not isinstance(clip, CLIP) or not valid_vae:
                embedded = {}
                if isinstance(clip, CLIP):
                    embedded["clip"] = clip
                if valid_vae:
                    embedded["vae"] = vae
                self.load_base_model(name, hash=hash, unet_only=True, input_unet=unet, input_components=embedded)
                return
            vae.throw_exception_if_invalid()

        if unet == None:
            print(f"Failed to load {name}")
            self.load_error = self.load_error or (
                f"Could not load {name}: its model type was not recognized. "
                "Check that it is an image checkpoint or diffusion model file.")
            self.xl_base = None
            self.xl_base_hash = None
            self.xl_base_patched = None
            self.xl_base_patched_hash = None
        else:
            from ruinedfooocus.backends.comfy.model_attention import configure_qwen_attention
            configure_qwen_attention(unet)
            self.xl_base = self.StableDiffusionModel(
                unet=unet, clip=clip, vae=vae, clip_vision=clip_vision
            )
            if not (self.xl_base.unet.model.__class__.__name__ in self.known_models):
                print(
                    f"Model {self.xl_base.unet.model.__class__.__name__} not supported. RuinedFooocus supports {list(self.known_models)} models as the base model."
                )
                self.load_error = (f"{name} is a {self.xl_base.unet.model.__class__.__name__} model, "
                                   "which is not supported as an image model.")
                self.xl_base = None

            if self.xl_base is not None:
                self.xl_base_patched = self.xl_base
                self.xl_base_patched_hash = None

        if self.xl_base is None:
            return
        if self.model_info is None:
            self.model_info = self.get_clip_and_vae(self.xl_base.unet)
        self.xl_base_hash = hash if hash is not None else name
        self.base_file_signature = signature
        self.component_file_signatures = component_files
        return

    def load_loras(self, loras):
        def patch_count(unet, clip):
            return sum(len(rows) for patcher in (unet, getattr(clip, 'patcher', None))
                       for rows in getattr(patcher, 'patches', {}).values())

        loaded_loras = []
        loras = sorted(loras, key=lambda x: x['name'].lower())
        resolved = []
        for lora in loras:
            name = lora.get("name", "None")
            weight = lora.get("weight", 0)
            hash = lora.get("hash", None)
            if name == "None" or weight == 0:
                continue

            filename = shared.models.get_model_path(
                "loras",
                name,
                hash=hash,
            )

            if filename is None:
                raise FileNotFoundError(f"Could not find image LoRA: {name}")
            resolved.append((name, weight, filename))
        signature = [(file_signature(filename), weight) for name, weight, filename in resolved]
        if self.xl_base_patched_hash == signature:
            return

        model = self.xl_base
        for name, weight, filename in resolved:
            print(f"Loading LoRA: {name}")
            try:
                lora = comfy.utils.load_torch_file(str(filename), safe_load=True)
                unet, clip = comfy.sd.load_lora_for_models(
                    model.unet, model.clip, lora, weight, weight
                )
                if patch_count(unet, clip) <= patch_count(model.unet, model.clip):
                    raise ValueError('No compatible weights were applied. Choose a LoRA for this checkpoint family.')
                model = self.StableDiffusionModel(
                    unet=unet,
                    clip=clip,
                    vae=model.vae,
                    clip_vision=model.clip_vision,
                )
                loaded_loras += [(name, weight)]
            except Exception as e:
                raise RuntimeError(f"Could not load image LoRA {name}: {e}") from e
        self.xl_base_patched = model
        self.xl_base_patched_hash = signature
        self.conditions = None

        print(f"LoRAs loaded: {loaded_loras}")
        return

    def refresh_controlnet(self, name=None):
        filename = control_tools.get_model(name)
        signature = file_signature(filename) if filename is not None else None
        if (signature is not None and self.xl_controlnet_hash == name and self.xl_controlnet is not None
                and signature == getattr(self, 'controlnet_file_signature', None)):
            return

        if filename is None and name in ("canny", "depth", "recolour", "sketch"):
            raise ValueError(f"ControlNet weights are unavailable: {name}")
        if filename is not None:
            self.xl_controlnet = comfy.controlnet.load_controlnet(str(filename))
            if self.xl_controlnet is None:
                raise ValueError(f"ComfyUI could not load ControlNet: {filename}")
            self.xl_controlnet_hash = name
            self.controlnet_file_signature = signature
            print(f"ControlNet model loaded: {self.xl_controlnet_hash}")
        if self.xl_controlnet_hash != name:
            self.xl_controlnet = None
            self.xl_controlnet_hash = None

    conditions = None

    def textencode(self, id, text, clip_skip):
        update = False
        hash = f"{text} {clip_skip}"
        if hash != self.conditions[id]["text"]:
            # Keep the model's native CLIP layer intact when switching presets.
            clip = self.xl_base_patched.clip
            if clip_skip > 1:
                clip = CLIPSetLastLayer().set_last_layer(clip, -clip_skip)[0]
            self.conditions[id]["cache"] = CLIPTextEncode().encode(
                clip=clip, text=text
            )[0]
        self.conditions[id]["text"] = hash
        update = True
        return update

    def decode_latent(self, samples, **options):
        from ruinedfooocus.backends.comfy.vae_decode import decode_image_vae
        return decode_image_vae(self.xl_base_patched, samples, **options)

    @torch.inference_mode()
    def process(
        self,
        gen_data=None,
        callback=None,
    ):
        if self.xl_base_patched is None:
            if self.load_error:
                raise ValueError(self.load_error)
            raise RuntimeError("Image model is not loaded. Check its required components.")
        architecture = self.xl_base_patched.unet.model.__class__.__name__
        if architecture not in self.known_models:
            raise ValueError(f"Unsupported image architecture: {architecture}")
        if gen_data.get('sampling_shift', settings.default_settings.get('sampling_shift')) in (None, ''):
            from ruinedfooocus.features.models.performance import MODEL_DEFAULT_SELECTIONS
            gen_data['sampling_shift'] = self.model_info.get('sampling_shift')
            if gen_data['sampling_shift'] in (None, '') and gen_data.get('performance_selection') in MODEL_DEFAULT_SELECTIONS:
                gen_data['sampling_shift'] = gen_data.get('_model_default_sampling_shift')
        if gen_data.get('inpaint_toggle'):
            if not isinstance(self.xl_base_patched.unet.model, SDXL):
                raise ValueError('Inpainting requires an SDXL checkpoint.')
            from ruinedfooocus.features.image_editing.inpaint_utils import source_and_mask
            inpaint_source, inpaint_mask = source_and_mask(gen_data.get('main_view'), gen_data.get('inpaint_view'))
        checkpoint = gen_data.get('base_model_name')
        guidance = resolve_guidance(gen_data, settings.default_settings, architecture, checkpoint)
        sampling_model = request_model(self.xl_base_patched.unet, guidance, architecture, checkpoint=checkpoint)
        effective = effective_guidance(guidance, sampling_model, architecture)
        gen_data.update(effective)
        gen_data['image_guidance'] = effective

        cfg = gen_data["cfg"]
        sampler_name = gen_data["sampler_name"]
        scheduler = gen_data["scheduler"]
        clip_skip = gen_data["clip_skip"]
        seed = gen_data["seed"] if isinstance(gen_data["seed"], int) else random.randint(1, 2**32)
        if "<facerestore>" in gen_data.get("positive_prompt", ""):
            gen_data["facerestore"] = True
        positive_prompt = gen_data["positive_prompt"].replace("<facerestore>", "")
        negative_prompt = gen_data["negative_prompt"].replace("<facerestore>", "")
        image_edit = (self.model_info.get('options', {}).get('has_image_edit', False) or
                      (architecture == "MOSSImage" and guidance['moss_image_edit_format'] != "Disabled"))
        hidream_edit = architecture == 'HiDreamO1'
        controlnet = control_tools.get_settings(gen_data)
        control_type = controlnet.get("type", "")
        input_image = gen_data.get('input_image') if control_type not in ('', 'None', None) else None
        additional_references = normalize_reference_images(
            gen_data.get('reference_images') if input_image is not None else None, input_image,
            architecture, guidance['moss_image_edit_format'])
        gen_data['reference_image_count'] = (1 if input_image is not None else 0) + len(additional_references)
        if control_type == 'edit':
            if input_image is None:
                raise ValueError('Add an image in Image tools.')
            native = image_edit or hidream_edit
            if architecture == 'Flux':
                name = gen_data.get('base_model_name', '')
                path = shared.models.get_file('checkpoints', name)
                metadata = shared.models.get_models_by_path('checkpoints', path, fetch=False) if path else {}
                native = 'kontext' in f'{name} {shared.models.get_model_base(metadata)}'.lower()
                if native:
                    controlnet = dict(controlnet, type='kontext')
                    control_type = 'kontext'
            if not native:
                # Saved presets or API requests may ask for native editing on a
                # checkpoint without it; edit by image-to-image instead of failing.
                from ruinedfooocus.features.models.image_components import IMG2IMG_DENOISE
                print("Native image edit is not available for this checkpoint; using image to image.")
                controlnet = {'type': 'img2img', 'denoise': IMG2IMG_DENOISE}
                control_type = 'img2img'
        if control_type in control_tools.controlnet_models and control_type not in ("upscale", "rembg", "faceswap"):
            if gen_data.get('inpaint_toggle') and control_type == 'img2img':
                # Inpainting encodes Main with its mask, not the optional tool upload.
                input_image, controlnet, control_type = None, {}, ''
                gen_data['reference_image_count'] = 0
            elif input_image is None:
                raise ValueError("Add an image in Image tools.")
            if control_type in ("canny", "depth", "recolour", "sketch"):
                if not isinstance(self.xl_base_patched.unet.model, SDXL):
                    raise ValueError("The bundled ControlNet weights require an SDXL checkpoint.")
                start, stop = float(controlnet["start"]), float(controlnet["stop"])
                if not 0 <= start < stop <= 1:
                    raise ValueError("ControlNet requires 0 <= Start < Stop <= 1.")
            elif control_type == "img2img":
                if not 0 < float(controlnet.get("denoise", controlnet.get("strength", 1))) <= 1:
                    raise ValueError("Image-to-image denoise must be greater than 0 and at most 1.")

        device = comfy.model_management.get_torch_device()
        switched_prompt = []
        img2img_mode = control_type == 'img2img'
        edit_latent = None
        reference_frame = None
        gen_data.pop('image_geometry', None)

        # Pre-process input-image
        input_images = 0
        if input_image:
            if not gen_data.get('inpaint_toggle') and (image_edit or hidream_edit or control_type in ('img2img', 'canny', 'depth', 'sketch', 'recolour')):
                requested_size = gen_data.setdefault('_image_requested_size', (gen_data['width'], gen_data['height']))
                input_image, reference_frame = prepare_reference(input_image, requested_size, guidance['image_resize_mode'],
                    multiple=32 if architecture == 'HiDreamO1' else 16)
                gen_data['width'], gen_data['height'] = reference_frame['canvas_width'], reference_frame['canvas_height']
            else:
                input_image = np.array(input_image.convert("RGB")).astype(np.float32) / 255.0
                input_image = torch.from_numpy(input_image)[None,]
                megapixels=float(gen_data["width"])*float(gen_data["height"])/(1024*1024)
                input_image = ImageScaleToTotalPixels().execute(
                    image=input_image, upscale_method="bicubic", megapixels=megapixels, resolution_steps=1
                )[0]
            input_images = 1 # "Counter" for the single input-image we have. (for now)
        elif architecture == 'HiDreamO1' and (gen_data['width'] % 32 or gen_data['height'] % 32):
            from ruinedfooocus.features.image_editing.image_geometry import canvas_plan
            reference_frame = canvas_plan((gen_data['width'], gen_data['height']), 32)
            gen_data['width'], gen_data['height'] = reference_frame['canvas_width'], reference_frame['canvas_height']

        if hidream_edit and input_image is not None and not img2img_mode:
            controlnet = None

        # Text-encoding
        moss_edit_format = guidance['moss_image_edit_format']
        moss_edit_strength = guidance['moss_image_edit_strength']
        if architecture == "MOSSImage" and moss_edit_format != "Disabled":
            if moss_edit_format not in EDIT_FORMATS:
                raise ValueError("Choose a supported MOSS image edit format in Settings.")
            if not np.isfinite(moss_edit_strength) or not 0 <= moss_edit_strength <= 100:
                raise ValueError("MOSS image edit strength must be between 0 and 100.")
        if image_edit and input_images > 0:
            if callback is not None:
                worker.add_result(
                    gen_data["task_id"],
                    "preview",
                    (-1, f"Processing image with text encoding ...", None)
                )
            if not img2img_mode:
                controlnet = None
            self.conditions = clean_prompt_cond_caches()

            match self.xl_base_patched.unet.model.__class__.__name__:
                case 'Fibo2':
                    self.conditions["+"]["cache"], self.conditions["-"]["cache"], edit_latent = TextEncodeFibo2().execute(
                        clip=self.xl_base_patched.clip,
                        prompt=positive_prompt,
                        negative_prompt=negative_prompt,
                        vae=self.xl_base_patched.vae,
                        width=gen_data["width"],
                        height=gen_data["height"],
                        images=reference_image_map(input_image, additional_references, reference_tensor),
                    ).result
                case 'MOSSImage':
                    for sign, prompt in (("+", positive_prompt), ("-", negative_prompt)):
                        self.conditions[sign]["cache"] = TextEncodeMOSSImageEdit().execute(
                            clip=self.xl_base_patched.clip,
                            prompt=prompt,
                            edit_format=moss_edit_format,
                            image_strength=moss_edit_strength,
                            images=reference_image_map(input_image, additional_references,
                                lambda image: reference_tensor(image, gen_data['width'], gen_data['height'])),
                            vae=self.xl_base_patched.vae,
                        )[0]
                case 'QwenImage':
                    self.conditions["+"]["cache"] = TextEncodeQwenImageEdit().execute(
                        clip=self.xl_base_patched.clip,
                        prompt=positive_prompt,
                        vae=self.xl_base.vae,
                        image=input_image
                    )[0]
                    self.conditions["-"]["cache"] = TextEncodeQwenImageEdit().execute(
                        clip=self.xl_base_patched.clip,
                        prompt=negative_prompt,
                        vae=self.xl_base.vae,
                        image=input_image
                    )[0]
                case 'MageFlow':
                    self.conditions["+"]["cache"], self.conditions["-"]["cache"], edit_latent = TextEncodeMageFlowEdit().execute(
                        clip=self.xl_base_patched.clip,
                        vae=self.xl_base_patched.vae,
                        prompt=positive_prompt,
                        negative_prompt=negative_prompt,
                        images=reference_image_map(input_image, additional_references, reference_tensor),
                        width=gen_data["width"],
                        height=gen_data["height"],
                    ).result
                case _:
                    print(f"ERROR: Trying to encode with unknown model {self.xl_base_patched.unet.__class__.__name__}. (This should never happen.")
                    raise

            input_images -= 1
        else:
            if callback is not None:
                worker.add_result(
                    gen_data["task_id"],
                    "preview",
                    (-1, f"Processing text encoding ...", None)
                )
            if self.conditions is None:
                self.conditions = clean_prompt_cond_caches()
            # Scheduled text is request-local; release the last job's extra tensor.
            self.conditions['switch'] = {'text': None, 'cache': None}
            self.textencode("+", positive_prompt, clip_skip)
            self.textencode("-", negative_prompt, clip_skip)

            prompt_per_step = pp.prompt_switch_per_step(positive_prompt, gen_data["steps"])
            if prompt_per_step:
                encoded_switches = {}
                for i in range(len(prompt_per_step)):
                    text = prompt_per_step[i]
                    if text not in encoded_switches:
                        self.textencode("switch", text, clip_skip)
                        encoded_switches[text] = self.conditions["switch"]["cache"]
                    positive_switch = encoded_switches[text]
                    start_perc = i / len(prompt_per_step)
                    end_perc = (i + 1) / len(prompt_per_step)
                    positive_switch = conditioning_set_values(positive_switch,
                        {"start_percent": start_perc, "end_percent": end_perc, "_rf_prompt_step": i})
                    switched_prompt += positive_switch


        # Keep cached text conditioning independent of each request's image/mask.
        positive_conditioning = self.conditions["+"]["cache"]
        negative_conditioning = self.conditions["-"]["cache"]
        if hidream_edit and input_image is not None:
            images = reference_image_map(input_image, additional_references, reference_tensor)
            positive_conditioning, negative_conditioning = HiDreamO1ReferenceImages.execute(
                positive=positive_conditioning, negative=negative_conditioning, images=images)
            if switched_prompt:
                switched_prompt = HiDreamO1ReferenceImages.execute(
                    positive=switched_prompt, negative=self.conditions['-']['cache'], images=images)[0]
            input_images -= 1

        # Controlnet / img2img
        if controlnet is None or not "type" in controlnet:
            controlnet = {}
            controlnet["type"] = "None"

        if controlnet['type'] == 'kontext':
            img2img_mode = True

        if controlnet["type"] != "None" and input_images > 0:
            if callback is not None:
                worker.add_result(
                    gen_data["task_id"],
                    "preview",
                    (-1, f"Powering up ...", None)
                )

            self.refresh_controlnet(name=controlnet["type"])
            match controlnet["type"].lower():
                case "canny":
                    low, high = sanitize_canny_thresholds(
                        float(controlnet.get("edge_low", 0.0)),
                        float(controlnet.get("edge_high", 1.0)),
                    )
                    input_image = Canny.execute(
                        image=input_image,
                        low_threshold=low,
                        high_threshold=high,
                    )[0]
                case 'depth':
                    from ruinedfooocus.features.image_editing.depth_control import prepare_depth, depth_settings
                    options = depth_settings(controlnet)
                    input_image = prepare_depth(self, input_image, options)
                    gen_data['depth_preprocessing'] = dict(options,
                        resolved_model=getattr(self, 'depth_model_name', None) if options['depth_preprocess'] == 'auto' else None)
            if self.xl_controlnet:
                (
                    positive_conditioning,
                    negative_conditioning,
                ) = ControlNetApplyAdvanced().apply_controlnet(
                    positive=switched_prompt if switched_prompt else positive_conditioning,
                    negative=negative_conditioning,
                    control_net=self.xl_controlnet,
                    image=input_image,
                    strength=float(controlnet["strength"]),
                    start_percent=float(controlnet["start"]),
                    end_percent=float(controlnet["stop"]),
                    vae=self.xl_base_patched.vae,
                )
                switched_prompt = []  # The native result retains every interval and its ControlNet.

            if controlnet["type"].lower() == "img2img":
                img2img_mode = True

        if edit_latent is not None and not img2img_mode:
            # Native edit references and target must share the encoder's size.
            latent = edit_latent
            denoise = None
        elif img2img_mode:
            # Ordinary batches always start from the supplied reference. Explicit
            # loopback is request-local and cannot reuse another job's output.
            if controlnet.get("loopback", False) and gen_data.get("_loopback_image") is not None:
                input_image = Image.fromarray(gen_data["_loopback_image"]).convert("RGB")
                input_image = np.array(input_image).astype(np.float32) / 255.0
                input_image = torch.from_numpy(input_image)[None,]
            if controlnet["type"].lower() == "kontext":
                input_image = FluxKontextImageScale().execute(input_image)[0]

            latent = VAEEncode().encode(
                vae=self.xl_base_patched.vae, pixels=input_image
            )[0]
            # The output keeps the encoded source size; record it for metadata/replay.
            scale = self.xl_base_patched.vae.spacial_compression_decode()
            gen_data["height"], gen_data["width"] = (int(side) * scale for side in latent["samples"].shape[-2:])
            denoise = float(controlnet.get("denoise", controlnet.get("strength", 1)))
        else:
            # Get the correct of latent image to start with
            latent_type = self.model_info.get('latent', None)
            match latent_type:
                case 'FLUX2':
                    latent = EmptyFlux2LatentImage().execute(
                        width=gen_data["width"], height=gen_data["height"], batch_size=1
                    )[0]
                case 'HiDreamO1':
                    latent = EmptyHiDreamO1LatentImage().execute(
                        width=gen_data["width"], height=gen_data["height"], batch_size=1
                    )[0]
                case 'HunyuanImage':
                    latent = EmptyHunyuanImageLatent().execute(
                        width=gen_data["width"], height=gen_data["height"], batch_size=1
                    )[0]
                case "ChromaRadience":
                    latent = EmptyChromaRadianceLatentImage().execute(
                        width=gen_data["width"], height=gen_data["height"], batch_size=1
                    )[0]
                case 'SD3':
                    latent = EmptySD3LatentImage().execute(
                        width=gen_data["width"], height=gen_data["height"], batch_size=1
                    )[0]
                case 'MAGE':
                    zero_latent = torch.zeros([1, 128, gen_data['height'] // 16, gen_data['width'] // 16], device=comfy.model_management.intermediate_device())
                    latent = {"samples": zero_latent, "downscale_ratio_spacial": 16}
                case _:
                    latent = EmptyLatentImage().generate(
                        width=gen_data["width"], height=gen_data["height"], batch_size=1
                    )[0]
            denoise = None

        if gen_data.get('inpaint_toggle'):
            denoise = float(gen_data.get("inpaint_denoise", 1.0))
            if not 0 < denoise <= 1:
                raise ValueError("Inpainting denoise must be greater than 0 and at most 1.")
            mask_grow = int(gen_data.get("inpaint_mask_grow", 20))
            if not 0 <= mask_grow <= 128:
                raise ValueError("Mask expansion must be between 0 and 128 pixels.")
            from ruinedfooocus.features.image_editing.inpaint_utils import prepare_inpaint
            requested_size = gen_data.setdefault('_image_requested_size', (gen_data['width'], gen_data['height']))
            image, mask, reference_frame = prepare_inpaint(
                inpaint_source, inpaint_mask, requested_size, guidance['image_resize_mode'])
            gen_data['width'], gen_data['height'] = reference_frame['canvas_width'], reference_frame['canvas_height']
            latent = VAEEncodeForInpaint().encode(
                vae=self.xl_base_patched.vae, pixels=image, mask=mask, grow_mask_by=mask_grow)[0]

        gen_data["sampling_denoise"] = denoise if denoise is not None else 1.0

        # KSampler

        # Flux embeds guidance in the conditioning instead of sampler CFG.
        positive_cond = switched_prompt if switched_prompt else positive_conditioning
        from ruinedfooocus.features.prompting.prompt_scheduling import bind_prompt_steps
        positive_cond = bind_prompt_steps(positive_cond, sampling_model, gen_data['steps'],
                                          sampler_name, scheduler, denoise)
        if isinstance(self.xl_base.unet.model, Flux):
            if controlnet.get("type", "") == "kontext":
                positive_cond = ReferenceLatent().execute(positive_cond, latent=latent)[0]
            positive_cond = conditioning_set_values(positive_cond, {"guidance": cfg})
            cfg = 1.0

        latent_image = latent["samples"]
        latent_image = fix_empty_latent_channels(
            sampling_model, latent_image,
            latent.get("downscale_ratio_spacial"), latent.get("downscale_ratio_temporal"),
        )
        if architecture == "PixArt" and round(((latent_image.shape[-2] + latent_image.shape[-1]) // 2) / 64) == 0:
            raise ValueError("PixArt resolution is too small for its positional embeddings. Use 512x512 or a larger supported resolution.")

        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        previewer = (get_previewer(device, self.xl_base_patched.unet.model.latent_format)
                     if callback is not None and not gen_data.get("silent", False) else None)

        pbar = comfy.utils.ProgressBar(gen_data["steps"])

        def callback_function(step, x0, x, total_steps):
            y = None
            if previewer:
                try:
                    y = previewer.preview(x0, step, total_steps)
                except:
                    y = None
            if callback is not None:
                callback(step, x0, x, total_steps, y)
            pbar.update_absolute(step + 1, total_steps, None)

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Prepare models ...", None)
            )
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Start sampling ...", None)
            )

        if sampler_name == 'lcm':
            from ruinedfooocus.features.generation.image_sampling import sample_lcm, lcm_options, LCM_FIELDS
            noise_options = lcm_options(guidance, architecture, checkpoint)
            gen_data['image_guidance'].update({'effective_' + key: noise_options[native]
                for key, native in LCM_FIELDS.items()})
            samples = sample_lcm(model=sampling_model, noise=noise, steps=gen_data['steps'], cfg=cfg,
                scheduler=scheduler, positive=positive_cond, negative=negative_conditioning,
                latent_image=latent_image, denoise=denoise, noise_options=noise_options,
                noise_mask=noise_mask, callback=callback_function,
                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
        else:
            samples = comfy.sample.sample(
                model=sampling_model, noise=noise,
                steps=gen_data["steps"], cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler,
                positive=positive_cond, negative=negative_conditioning,
                latent_image=latent_image, denoise=denoise,
                start_step=0, last_step=gen_data["steps"],
                noise_mask=noise_mask,
                callback=callback_function,
                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed,
            )

        # VAE
        sampled_latent = latent.copy()
        sampled_latent.pop("downscale_ratio_spacial", None)
        sampled_latent.pop("downscale_ratio_temporal", None)
        sampled_latent["samples"] = samples

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"VAE decoding ...", None)
            )

        decoded_latent = self.decode_latent(sampled_latent, tiled=guidance['vae_tiled'],
            tile_size=guidance['vae_tile_size'], overlap=guidance['vae_overlap'])
        # Save feedback only when explicitly requested, before output upscaling.
        if img2img_mode and controlnet.get("loopback", False):
            gen_data["_loopback_image"] = np.clip(
                255.0 * decoded_latent[0].cpu().numpy(), 0, 255
            ).astype(np.uint8)
        decoded_latent = restore_reference_frame(decoded_latent, reference_frame)
        if reference_frame is not None:
            gen_data['width'], gen_data['height'] = reference_frame['width'], reference_frame['height']
            gen_data['image_geometry'] = reference_frame
        from ruinedfooocus.features.generation.automatic_upscale import upscale_decoded
        decoded_latent = upscale_decoded(self, decoded_latent, gen_data)
        images = [
            np.clip(255.0 * y.cpu().numpy(), 0, 255).astype(np.uint8)
            for y in decoded_latent
        ]

        if callback is not None:
            callback(gen_data["steps"], 0, 0, gen_data["steps"], images[0])

        return images
