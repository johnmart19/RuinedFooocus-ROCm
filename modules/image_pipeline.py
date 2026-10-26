import gc
import numpy as np
import os
import torch
import traceback
import re

import modules.controlnet
import modules.async_worker as worker
import modules.prompt_processing as pp

from PIL import Image

# Known models (from ComfyUI/comfy/model_base.py)
from comfy.model_base import Flux, SDXL
from shared import path_manager, settings
import shared

from pathlib import Path
import random

import comfy.utils
import comfy.model_management
from comfy.sd import load_checkpoint_guess_config

from comfy_extras.nodes_model_advanced import (
    ModelSamplingAuraFlow,
    ModelSamplingSD3,
)
from nodes import (
    CLIPTextEncode,
    CLIPSetLastLayer,
    ControlNetApplyAdvanced,
    EmptyLatentImage,
    VAEDecode,
    VAEDecodeTiled,
    VAEEncode,
    VAEEncodeForInpaint,
)

from comfy_extras.nodes_sd3 import EmptySD3LatentImage
from comfy_extras.nodes_flux import FluxKontextImageScale, EmptyFlux2LatentImage
from comfy_extras.nodes_hunyuan import EmptyHunyuanImageLatent
from comfy_extras.nodes_hidream_o1 import EmptyHiDreamO1LatentImage, HiDreamO1PatchSeamSmoothing
from comfy_extras.nodes_chroma_radiance import EmptyChromaRadianceLatentImage
from comfy_extras.nodes_edit_model import ReferenceLatent
from comfy_extras.nodes_qwen import TextEncodeQwenImageEdit
from comfy_extras.nodes_mage import TextEncodeMageFlowEdit
from comfy_extras.nodes_fibo2 import TextEncodeFibo2
from comfy_extras.nodes_moss_image import EDIT_FORMATS, TextEncodeMOSSImageEdit
from modules.image_guidance import (resolve_guidance, request_model, effective_guidance,
                                    normalize_reference_images, reference_image_map)
from node_helpers import conditioning_set_values

from comfy.sample import fix_empty_latent_channels
from comfy_extras.nodes_model_advanced import ModelNoiseScale
from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels
from comfy_extras.nodes_canny import Canny
from comfy.sd import CLIP, VAE
from modules.sequence_prefill import install_qwen35_prefill

install_qwen35_prefill()

from modules.pipeline_utils import (
    get_previewer,
    clean_prompt_cond_caches,
    load_diffusion_model,
)
from modules.canny_utils import sanitize_canny_thresholds

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
            "model_sampling": (('AuraFlow', settings.default_settings["anima_image_shift"])
                               if "anima_image_shift" in settings.default_settings else None),
        },
        "AuraFlow": {
            "clip_type": comfy.sd.CLIPType.STABLE_DIFFUSION,
            "model_sampling": ('AuraFlow', settings.default_settings.get("auraflow_shift", 1.73))
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
            "model_sampling": ('SD3', settings.default_settings.get("hidream_shift", 3.0))
        },
        "HiDreamO1": {
            "latent": "HiDreamO1",
            "clip_type": comfy.sd.CLIPType.HIDREAM,
            "options": {"ModelNoiseScale": 8.0, "HiDreamO1SeamSmoothing": True}
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
            "model_sampling": ('AuraFlow', settings.default_settings.get("lumina2_shift", 3.0))
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
            "model_sampling": ('AuraFlow', settings.default_settings.get("newbieimage_shift", 6.0))
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
            "model_sampling": ('AuraFlow', settings.default_settings.get("qwen_image_shift", 3.10)),
            "options": {"has_image_edit": True}
        },
        "SD3": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.SD3,
            "model_sampling": ('SD3', settings.default_settings.get("sd3_shift", 3.0))
        },
        "SDXL": {
            "clip_type": comfy.sd.CLIPType.STABLE_DIFFUSION,
        },
        "ZImage": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.LUMINA2,
            "model_sampling": ('AuraFlow', settings.default_settings.get("lumina2_shift", 3.0))
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

    def get_clip_and_vae(self, unet):
        from modules.resource_defaults import resource_name
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
        from modules.image_components import IMAGE_COMPONENTS
        info.update(IMAGE_COMPONENTS[unet_type])
        info["clip_names"] = [resource_name(key, settings.default_settings) for key in info["clip_names"]]
        if info["vae_name"] != "pixel_space":
            info["vae_name"] = resource_name(info["vae_name"], settings.default_settings)
        return info

    def load_base_model(self, name, unet_only=False, input_unet=None, hash=None, input_components=None):
        embedded = input_components or {}
        components = {key: value for key, value in settings.default_settings.items()
                      if key.startswith(("clip_", "vae_"))}
        if (self.xl_base_hash is not None
                and self.xl_base_hash in (name, hash)
                and components == getattr(self, "component_settings", None)):
            return
        self.component_settings = components

        self.xl_base = None
        self.xl_base_hash = None
        self.xl_base_patched = None
        self.xl_base_patched_hash = None
        self.conditions = None
        self.model_info = None

        default = None

        filename = shared.models.get_model_path(
            "checkpoints",
            name,
            hash=hash,
            default=default,
        )

        if filename is None:
            print(f"ERROR: Could not load checkpoint {name}")
            return

        if Path(filename).suffix == '.merge':
            print(f"Error: Model type not supported.")
            return

        if input_unet is None: # Be quiet if we already loaded a unet
            print(f"Loading base {'unet' if unet_only else 'model'}: {name}")

        gc.collect(generation=2)

        comfy.model_management.cleanup_models()
        comfy.model_management.soft_empty_cache()

        unet = None

        filename = str(filename)
        if Path(filename).suffix.lower() == ".gguf" or unet_only:
            with torch.torch.inference_mode():
                try:
                    unet = load_diffusion_model(filename, input_unet)

                    # Get text-encoders (clip) and vae to match the unet
                    model_info = self.get_clip_and_vae(unet)
                    self.model_info = model_info

                    # Special massaging of Lumina2 unet
                    if model_info.get('model_sampling', None):
                        match model_info['model_sampling'][0]:
                            case 'AuraFlow':
                                unet = ModelSamplingAuraFlow().patch_aura(
                                    model=unet,
                                    shift=model_info['model_sampling'][1]
                                )[0]
                            case 'SD3':
                                unet = ModelSamplingSD3().patch(
                                    model=unet,
                                    shift=model_info['model_sampling'][1]
                                )[0]

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

                    print(f"Loading CLIP: {model_info['clip_names']}")
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
                            print(f"Loading VAE: {model_info['vae_name']}")
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
            self.xl_base = None
            self.xl_base_hash = None
            self.xl_base_patched = None
            self.xl_base_patched_hash = None
        else:
            from modules.model_attention import configure_qwen_attention
            configure_qwen_attention(unet)
            self.xl_base = self.StableDiffusionModel(
                unet=unet, clip=clip, vae=vae, clip_vision=clip_vision
            )
            if not (self.xl_base.unet.model.__class__.__name__ in self.known_models):
                print(
                    f"Model {self.xl_base.unet.model.__class__.__name__} not supported. RuinedFooocus supports {list(self.known_models)} models as the base model."
                )
                self.xl_base = None

            if self.xl_base is not None:
                self.xl_base_hash = hash if hash is not None else name
                self.xl_base_patched = self.xl_base
                self.xl_base_patched_hash = None

        # Model Options
        if self.xl_base is None:
            return
        if self.model_info is None:
            self.model_info = self.get_clip_and_vae(self.xl_base.unet)
        options = self.model_info.get("options", {})
        if options.get("ModelNoiseScale", None) is not None:
            self.xl_base.unet = ModelNoiseScale().patch(
                model=self.xl_base.unet,
                noise_scale=options.get("ModelNoiseScale", 0.0),
            )[0]
        if options.get("HiDreamO1SeamSmoothing", False):
            self.xl_base.unet = HiDreamO1PatchSeamSmoothing().execute(
                model=self.xl_base.unet,
                start_percent=0.80,
                end_percent=1.00,
                pattern='single_shift',
                passes='ramp_2_4',
                blend='median',
                strength=1.00,
            )[0]
        return

    def load_loras(self, loras):
        loaded_loras = []
        loras = sorted(loras, key=lambda x: x['name'].lower())
        if self.xl_base_patched_hash == str(loras):
            return

        model = self.xl_base
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
            print(f"Loading LoRA: {name}")
            try:
                lora = comfy.utils.load_torch_file(str(filename), safe_load=True)
                unet, clip = comfy.sd.load_lora_for_models(
                    model.unet, model.clip, lora, weight, weight
                )
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
        self.xl_base_patched_hash = str(loras)
        self.conditions = None

        print(f"LoRAs loaded: {loaded_loras}")
        return

    def refresh_controlnet(self, name=None):
        if self.xl_controlnet_hash == name and self.xl_controlnet is not None:
            return

        filename = modules.controlnet.get_model(name)

        if filename is None and name in ("canny", "depth", "recolour", "sketch"):
            raise ValueError(f"ControlNet weights are unavailable: {name}")
        if filename is not None:
            self.xl_controlnet = comfy.controlnet.load_controlnet(str(filename))
            if self.xl_controlnet is None:
                raise ValueError(f"ComfyUI could not load ControlNet: {filename}")
            self.xl_controlnet_hash = name
            print(f"ControlNet model loaded: {self.xl_controlnet_hash}")
        if self.xl_controlnet_hash != name:
            self.xl_controlnet = None
            self.xl_controlnet_hash = None

    conditions = None

    def textencode(self, id, text, clip_skip):
        update = False
        # Don't encode <lora> and things like that
        for pat in [r"<lora:[^>]*>", r"<facerestore>"]:
            text = re.sub(pat, "", text)
        text = text.strip(", ")
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

    def decode_latent(self, samples):
        # Dynamic residency can leave diffusion weights in VRAM while a VAE's
        # convolution workspace needs a large contiguous allocation (notably HIP).
        # Retry only actual OOMs after the backend's own decode/tiling fallback.
        try:
            return self._decode_latent(samples)
        except Exception as error:
            comfy.model_management.raise_non_oom(error)
        # Leave the exception scope so failed decoder tensors can be collected.
        print("VAE decoding needs more VRAM; releasing diffusion weights and retrying.")
        comfy.model_management.unload_model_and_clones(self.xl_base_patched.unet)
        comfy.model_management.soft_empty_cache()
        return self._decode_latent(samples)

    def _decode_latent(self, samples):
        vae = self.xl_base_patched.vae
        latent = samples["samples"]
        if vae.device.type != "cpu" and latent.ndim == 4 and not latent.is_nested:
            # Avoid a full decode that exceeds VRAM. Windows can spill into
            # shared memory instead of raising OOM and triggering Comfy's fallback.
            memory = comfy.model_management
            budget = (memory.get_total_memory(vae.device) - memory.extra_reserved_memory()
                      - vae.patcher.model_size())
            required = vae.memory_used_decode(latent.shape, vae.vae_dtype)
            if required > budget:
                print("Using tiled VAE decoding to fit available VRAM.")
                return VAEDecodeTiled().decode(vae=vae, samples=samples, tile_size=512)[0]
        return VAEDecode().decode(vae=vae, samples=samples)[0]

    @torch.inference_mode()
    def process(
        self,
        gen_data=None,
        callback=None,
    ):
        if self.xl_base_patched is None:
            raise RuntimeError("Image model is not loaded. Check its required components.")
        architecture = self.xl_base_patched.unet.model.__class__.__name__
        if architecture not in self.known_models:
            raise ValueError(f"Unsupported image architecture: {architecture}")
        if gen_data.get('inpaint_toggle'):
            if not isinstance(self.xl_base_patched.unet.model, SDXL):
                raise ValueError('Inpainting requires an SDXL checkpoint.')
            from modules.inpaint_utils import source_and_mask
            inpaint_source, inpaint_mask = source_and_mask(gen_data.get('main_view'), gen_data.get('inpaint_view'))
        guidance = resolve_guidance(gen_data, settings.default_settings, architecture)
        sampling_model = request_model(self.xl_base_patched.unet, guidance, architecture)
        effective = effective_guidance(guidance, sampling_model, architecture)
        gen_data.update(effective)
        gen_data['image_guidance'] = effective

        cfg = gen_data["cfg"]
        sampler_name = gen_data["sampler_name"]
        scheduler = gen_data["scheduler"]
        clip_skip = gen_data["clip_skip"]
        input_image = gen_data["input_image"]
        additional_references = normalize_reference_images(gen_data.get('reference_images'), input_image,
            architecture, guidance['moss_image_edit_format'])
        gen_data['reference_image_count'] = (1 if input_image is not None else 0) + len(additional_references)
        seed = gen_data["seed"] if isinstance(gen_data["seed"], int) else random.randint(1, 2**32)
        if "<facerestore>" in gen_data.get("positive_prompt", ""):
            gen_data["facerestore"] = True
        positive_prompt = gen_data["positive_prompt"]
        negative_prompt = gen_data["negative_prompt"]
        image_edit = (self.model_info.get('options', {}).get('has_image_edit', False) or
                      (architecture == "MOSSImage" and guidance['moss_image_edit_format'] != "Disabled"))
        controlnet = modules.controlnet.get_settings(gen_data)
        control_type = controlnet.get("type", "")
        if control_type in modules.controlnet.controlnet_models and control_type not in ("upscale", "rembg", "faceswap"):
            if input_image is None:
                raise ValueError("Select an Input image in PowerUp for this workflow.")
            if control_type in ("canny", "depth", "recolour", "sketch"):
                if not isinstance(self.xl_base_patched.unet.model, SDXL):
                    raise ValueError("The bundled ControlNet weights require an SDXL checkpoint.")
                start, stop = float(controlnet["start"]), float(controlnet["stop"])
                if not 0 <= start < stop <= 1:
                    raise ValueError("ControlNet requires 0 <= Start < Stop <= 1.")
            elif control_type == "img2img" and not image_edit:
                if not 0 < float(controlnet.get("denoise", controlnet.get("strength", 1))) <= 1:
                    raise ValueError("Image-to-image denoise must be greater than 0 and at most 1.")

        device = comfy.model_management.get_torch_device()
        switched_prompt = []
        img2img_mode = False
        edit_latent = None

        # Pre-process input-image
        input_images = 0
        if input_image:
            input_image = np.array(input_image.convert("RGB")).astype(np.float32) / 255.0
            input_image = torch.from_numpy(input_image)[None,]
            megapixels=float(gen_data["width"])*float(gen_data["height"])/(1024*1024)
            input_image = ImageScaleToTotalPixels().execute(
                image=input_image, upscale_method="bicubic", megapixels=megapixels, resolution_steps=1
            )[0]
            input_images = 1 # "Counter" for the single input-image we have. (for now)

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
            controlnet = None # Disable any other controlnet
            self.conditions = clean_prompt_cond_caches()

            match self.xl_base_patched.unet.model.__class__.__name__:
                case 'Fibo2':
                    img = gen_data["input_image"].convert("RGB")
                    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]
                    self.conditions["+"]["cache"], self.conditions["-"]["cache"], edit_latent = TextEncodeFibo2().execute(
                        clip=self.xl_base_patched.clip,
                        prompt=positive_prompt,
                        negative_prompt=negative_prompt,
                        vae=self.xl_base_patched.vae,
                        width=gen_data["width"],
                        height=gen_data["height"],
                        images=reference_image_map(img, additional_references, reference_tensor),
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
                    img = gen_data["input_image"].convert("RGB")
                    img = np.array(img).astype(np.float32) / 255.0
                    img = torch.from_numpy(img)[None,]
                    self.conditions["+"]["cache"], self.conditions["-"]["cache"], edit_latent = TextEncodeMageFlowEdit().execute(
                        clip=self.xl_base_patched.clip,
                        vae=self.xl_base_patched.vae,
                        prompt=positive_prompt,
                        negative_prompt=negative_prompt,
                        images=reference_image_map(img, additional_references, reference_tensor),
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
            self.textencode("+", positive_prompt, clip_skip)
            self.textencode("-", negative_prompt, clip_skip)

            if "[" in positive_prompt and "]" in positive_prompt and not positive_prompt.strip().startswith("{"):
                if controlnet is not None and input_image is not None:
                    print("ControlNet and [prompt|switching] do not work well together.")
                    print("ControlNet will only be applied to the first prompt.")

                prompt_per_step = pp.prompt_switch_per_step(positive_prompt, gen_data["steps"])
                perc_per_step = round(100 / gen_data["steps"], 2)
                for i in range(len(prompt_per_step)):
                    self.textencode("switch", prompt_per_step[i], clip_skip)
                    positive_switch = self.conditions["switch"]["cache"]
                    start_perc = round((perc_per_step * i) / 100, 2)
                    end_perc = round((perc_per_step * (i + 1)) / 100, 2)
                    if end_perc >= 0.99:
                        end_perc = 1
                    positive_switch = conditioning_set_values(positive_switch,
                        {"start_percent": start_perc, "end_percent": end_perc})
                    switched_prompt += positive_switch


        # Keep cached text conditioning independent of each request's image/mask.
        positive_conditioning = self.conditions["+"]["cache"]
        negative_conditioning = self.conditions["-"]["cache"]

        # Controlnet / img2img
        if controlnet is None or not "type" in controlnet:
            controlnet = {}
            controlnet["type"] = "None"

        # Flux and Kontext share an architecture; only opt in when identified.
        if (
            controlnet["type"] == "None" and
            isinstance(self.xl_base.unet.model, Flux) and
            input_image is not None
        ):
            name = gen_data.get("base_model_name", "")
            path = shared.models.get_file("checkpoints", name)
            metadata = shared.models.get_models_by_path("checkpoints", path, fetch=False) if path else {}
            family = shared.models.get_model_base(metadata)
            if "kontext" in f"{name} {family}".lower():
                controlnet["type"] = "kontext"
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
            if self.xl_controlnet:
                (
                    positive_conditioning,
                    negative_conditioning,
                ) = ControlNetApplyAdvanced().apply_controlnet(
                    positive=self.conditions["+"]["cache"],
                    negative=self.conditions["-"]["cache"],
                    control_net=self.xl_controlnet,
                    image=input_image,
                    strength=float(controlnet["strength"]),
                    start_percent=float(controlnet["start"]),
                    end_percent=float(controlnet["stop"]),
                    vae=self.xl_base_patched.vae,
                )

            # NOTE: If we are doing img2img, reuse the previous image ("Loopback").
            #       It is not obvious that this is a good idea.
            if controlnet["type"].lower() == "img2img":
                img2img_mode = True

        if edit_latent is not None:
            # Native edit references and target must share the encoder's size.
            latent = edit_latent
            force_full_denoise = False
            denoise = None
        elif img2img_mode:
            # If this isn't the first image, do "Loopback"
            if controlnet.get("loopback", True) and shared.state.get("preview_count", 0) > 0:
                input_image = Image.fromarray(shared.shared_cache["prev_image"]).convert("RGB")
                input_image = np.array(input_image).astype(np.float32) / 255.0
                input_image = torch.from_numpy(input_image)[None,]
            if controlnet["type"].lower() == "kontext":
                input_image = FluxKontextImageScale().execute(input_image)[0]

            latent = VAEEncode().encode(
                vae=self.xl_base_patched.vae, pixels=input_image
            )[0]
            force_full_denoise = False
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
            force_full_denoise = False
            denoise = None

        if "inpaint_toggle" in gen_data and gen_data["inpaint_toggle"]:
            denoise = float(gen_data.get("inpaint_denoise", 1.0))
            if not 0 < denoise <= 1:
                raise ValueError("Inpainting denoise must be greater than 0 and at most 1.")
            mask_grow = int(gen_data.get("inpaint_mask_grow", 20))
            if not 0 <= mask_grow <= 128:
                raise ValueError("Mask expansion must be between 0 and 128 pixels.")
            main_image = inpaint_source
            image = np.array(main_image)
            image = torch.from_numpy(image)[None,] / 255.0

            mask = np.array(inpaint_mask)
            mask = torch.from_numpy(mask)[None,] / 255.0

            latent = VAEEncodeForInpaint().encode(
                vae=self.xl_base_patched.vae,
                pixels=image,
                mask=mask,
                grow_mask_by=mask_grow,
            )[0]

        gen_data["sampling_denoise"] = denoise if denoise is not None else 1.0

        # KSampler

        # Flux embeds guidance in the conditioning instead of sampler CFG.
        positive_cond = switched_prompt if switched_prompt else positive_conditioning
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

        previewer = get_previewer(device, self.xl_base_patched.unet.model.latent_format)

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

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Start sampling ...", None)
            )

        samples = comfy.sample.sample(
            model=sampling_model, noise=noise,
            steps=gen_data["steps"], cfg=cfg,
            sampler_name=sampler_name, scheduler=scheduler,
            positive=positive_cond, negative=negative_conditioning,
            latent_image=latent_image, denoise=denoise,
            start_step=0, last_step=gen_data["steps"],
            force_full_denoise=force_full_denoise, noise_mask=noise_mask,
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

        decoded_latent = self.decode_latent(sampled_latent)
        # Loopback continues diffusion at native size, independent of output upscale.
        shared.shared_cache["prev_image"] = np.clip(
            255.0 * decoded_latent[0].cpu().numpy(), 0, 255
        ).astype(np.uint8)
        from modules.automatic_upscale import upscale_decoded
        decoded_latent = upscale_decoded(self, decoded_latent, gen_data)
        images = [
            np.clip(255.0 * y.cpu().numpy(), 0, 255).astype(np.uint8)
            for y in decoded_latent
        ]

        if callback is not None:
            callback(gen_data["steps"], 0, 0, gen_data["steps"], images[0])

        return images
