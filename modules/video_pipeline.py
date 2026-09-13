from modules.video_preview import video_callback
from safetensors.torch import save_file
import gc
import numpy as np
import os
import torch
from modules.gguf_loader import load_diffusion_model as load_gguf_model
import traceback
import cv2
import logging
import json

import modules.async_worker as worker
from modules.util import generate_temp_filename, TimeIt, get_checkpoint_hashes, get_lora_hashes
from PIL import Image

import os
from comfy.model_base import LTXAV
from shared import path_manager, settings
import shared

from pathlib import Path
import random
from modules.pipeline_utils import (
    clean_prompt_cond_caches,
)

import comfy.utils
from comfy.sample import fix_empty_latent_channels
from comfy.sd import load_checkpoint_guess_config, load_state_dict_guess_config, VAE
from tqdm import tqdm

#from comfyui_gguf.nodes import gguf_sd_loader as load_gguf_sd, DualCLIPLoaderGGUF, GGUFModelPatcher, UnetLoaderGGUF
#from comfyui_gguf.ops import GGMLOps
from molbal_comfyui_gguf.nodes import gguf_sd_loader as load_gguf_sd, DualCLIPLoaderGGUF, GGUFModelPatcher, UnetLoaderGGUF
from molbal_comfyui_gguf.ops import GGMLOps
#from calcuis_gguf.pig import load_gguf_sd, GGMLOps, GGUFModelPatcher, load_gguf_clip
#from calcuis_gguf.pig import DualClipLoaderGGUF as DualCLIPLoaderGGUF


from nodes import (
    CLIPTextEncode,
    DualCLIPLoader,
    VAEDecodeTiled,
    VAEDecode,
)

from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced, Noise_RandomNoise, BasicScheduler, KSamplerSelect, BasicGuider, CFGGuider
from comfy_extras.nodes_lt import EmptyLTXVLatentVideo, LTXVImgToVideo, LTXVConditioning, LTXVScheduler, LTXVConcatAVLatent, LTXVSeparateAVLatent
from comfy_extras.nodes_lt import ModelSamplingLTXV
from comfy_extras.nodes_lt_audio import LTXVEmptyLatentAudio, LTXVAudioVAEDecode
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
from comfy_extras.nodes_custom_sampler import BasicScheduler, BasicGuider
from comfy_extras.nodes_audio import VAEDecodeAudio


from comfy_extras.nodes_video import CreateVideo
from comfy.model_patcher import ModelPatcher
from comfy_api.latest import Types

class pipeline:
    pipeline_type = ["video_pipeline"]

    class StableDiffusionModel:
        def __init__(self, clip, unet, vae, audio_vae=None):
            self.clip = clip
            self.unet = unet
            self.vae = vae
            self.audio_vae = audio_vae

        def to_meta(self):
            if self.unet is not None:
                self.unet.model.to("meta")
            if self.clip is not None:
                self.clip.cond_stage_model.to("meta")
            if self.vae is not None:
                self.vae.first_stage_model.to("meta")
            if self.audio_vae is not None:
                self.audio_vae.first_stage_model.to("meta")

    clip = None
    vae = None
    audio_vae = None
    model_hash = ""
    model_base = None
    model_hash_patched = ""
    model_base_patched = None
    conditions = None

    ggml_ops = GGMLOps()
    logger = logging.getLogger()

    # Optional function
    def parse_gen_data(self, gen_data):
        gen_data["original_image_number"] = gen_data.get("video_duration", 1 + ((int(gen_data["image_number"] / 4.0) + 1) * 4))
        gen_data["image_number"] = 1
        gen_data["show_preview"] = False
        return gen_data

    @staticmethod
    def get_clip_name(shortname):
        # List of short names and default names for different text encoders
        defaults = {
            "clip_t5": "t5-v1_1-xxl-encoder-Q3_K_S.gguf",
            "clip_gemma3_12b": "gemma-3-12b-it-Q4_0.gguf",
            "clip_gemma4_12b": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
            "clip_ltx23_text_proj": "ltx-2.3_text_projection_bf16.safetensors",
            "clip_ltx2_dev": "ltx-2-19b-embeddings_connector_dev_bf16.safetensors",
            "clip_ltx2_distilled": "ltx-2-19b-embeddings_connector_distill_bf16.safetensors",
            "clip_qwen3vl_32b": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        }
        return settings.default_settings.get(shortname, defaults[shortname] if shortname in defaults else None)

    @staticmethod
    def get_vae_name(shortname):
        # List of short names and default names for different VAE's
        defaults = {
            "vae_ltxv23_audio": "LTX23_audio_vae_bf16.safetensors",
            "vae_ltxv2_audio": "LTX2_audio_vae_bf16.safetensors",
            "vae_ltxv2_video": "LTX2_video_vae_bf16.safetensors",
            "vae_ltxv23_video": "LTX23_video_vae_bf16.safetensors",
            "vae_ltxv25_audio": "ltx-2.5-audio-vae-bf16.safetensors",
            "vae_ltxv25_video": "ltx-2.5-video-vae-bf16.safetensors",
            "vae_minimax_h3_audio": "minimax_h3_audio_vae_fp32.safetensors",
            "vae_minimax_h3_video": "minimax_h3_video_vae_fp16.safetensors",
        }
        return settings.default_settings.get(shortname, defaults[shortname] if shortname in defaults else None)

    known_model_info = {
        "LTXAV": {
            "latent": None,
            "clip_type": comfy.sd.CLIPType.LTXV,
            "clip_names": ["clip_gemma3_12b", "clip_ltx23_text_proj"],
            "vae_name": "vae_ltxv23_video",
            "audio_vae_name": "vae_ltxv23_audio",
            "frame_multiple": 8, "frame_offset": 1, "spatial_multiple": 32,
            "flags": ["need_audio_latent"],
            "options": {"guider": "CFGGuider", "scheduler": "LTXVScheduler"}
        },
        "LTXAV2.5": {
            "latent": None,
            "clip_type": comfy.sd.CLIPType.LTXV,
            "clip_names": ["clip_gemma4_12b"],
            "vae_name": "vae_ltxv25_video",
            "audio_vae_name": "vae_ltxv25_audio",
            "frame_multiple": 8, "frame_offset": 1, "spatial_multiple": 32,
            "flags": ["need_audio_latent"],
            "options": {"guider": "CFGGuider", "scheduler": "LTXVScheduler"}
        },
        "MiniMaxH3": {
            "latent": None,
            "clip_type": comfy.sd.CLIPType.MINIMAX,
            "clip_names": ["clip_qwen3vl_32b"],
            "vae_name": "vae_minimax_h3_video",
            "audio_vae_name": "vae_minimax_h3_audio",
            "frame_multiple": 17, "frame_offset": 5, "spatial_multiple": 32,
            "options": {"fps": 24.0, "guider": "BasicGuider", "scheduler": "BasicScheduler"},
        },
    }

    def get_clip_and_vae(self, unet, checkpoint=""):
        unet_type = unet.model.__class__.__name__
        position = None
        ltx23 = False

        # Some detective work...
        if unet_type == "LTXAV":
            weights = unet.model_state_dict()
            position = weights.get('diffusion_model.keyframes_abs_pos_embedding')
            ltx23 = 'diffusion_model.transformer_blocks.0.attn1.to_gate_logits.weight' in weights
            if position is not None and position.shape[1] == 4096:
                unet_type = "LTXAV2.5"

        ret = self.known_model_info.get(unet_type, {}).copy()
        if unet_type == "LTXAV" and not ltx23:
            connector = "clip_ltx2_distilled" if "distill" in Path(checkpoint).name.lower() else "clip_ltx2_dev"
            ret.update(clip_names=["clip_gemma3_12b", connector],
                       vae_name="vae_ltxv2_video", audio_vae_name="vae_ltxv2_audio")
        ret["clip_names"] = [self.get_clip_name(key) for key in ret["clip_names"]]
        ret["vae_name"] = self.get_vae_name(ret["vae_name"])
        ret["audio_vae_name"] = self.get_vae_name(ret["audio_vae_name"])
        ret['unet_type'] = unet_type
        self.model_info = ret
        return ret

    def load_base_model(self, name, unet_only=False, input_unet=None, hash=None):
        components = {key: value for key, value in settings.default_settings.items()
                      if key.startswith(("clip_", "vae_"))}
        if (self.model_hash is not None
                and self.model_hash in (name, hash)
                and components == getattr(self, "component_settings", None)):
            return
        self.component_settings = components

        self.model_base = None
        self.model_hash = None
        self.model_base_patched = None
        self.model_patched_hash = None
        self.conditions = None
        self.model_info = None
        self.load_error = None

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
                    if Path(filename).suffix.lower() == ".gguf":
                        unet = load_gguf_model(filename)
                    elif input_unet is not None:
                        if isinstance(input_unet, ModelPatcher):
                            unet = GGUFModelPatcher.clone(input_unet)
                            unet.patch_on_device = True
                        else:
                            unet = comfy.sd.load_diffusion_model_state_dict(
                                input_unet, model_options={"custom_operations": self.ggml_ops}
                            )
                            #unet = comfy.sd.load_diffusion_model_state_dict(input_unet)
                            try:
                                unet = GGUFModelPatcher.clone(unet)
                                unet.patch_on_device = True
                            except Exception as e:
                                unet = input_unet
                                print(f"ERROR: {e}")
                                traceback.print_exc()
                    else:
                        # ComfyUI selects a dtype supported by this model and device.
                        model_options = {}
                        unet = comfy.sd.load_diffusion_model(filename, model_options=model_options)

                    # Get text-encoders (clip) and vae to match the unet
                    model_info = self.get_clip_and_vae(unet, filename)
                    self.model_info = model_info

                    # Load everything...
                    clip_paths = []
                    for clip_name in model_info['clip_names']:
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
                    if all(name.endswith(".safetensors") for name in clip_paths):
                        model_options = {}
                        device = comfy.model_management.get_torch_device()
                        if device == "cpu":
                            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
                        clip = comfy.sd.load_clip(ckpt_paths=clip_paths, clip_type=model_info['clip_type'], model_options=model_options)
                    else:
                        clip_loader = DualCLIPLoaderGGUF()
                        self.logger.setLevel(logging.ERROR) # Supress error messages
                        clip = clip_loader.load_patcher(
                            clip_paths,
                            model_info['clip_type'],
                            clip_loader.load_data(clip_paths)
                        )
                        self.logger.setLevel(logging.WARNING)


                    if model_info['vae_name'] == "pixel_space":
                        metadata = None
                        sd = {}
                        sd["pixel_space_vae"] = torch.tensor(1.0)
                    else:
                        vae_path = path_manager.get_folder_file_path(
                            "vae",
                            model_info['vae_name'],
                            default = os.path.join(path_manager.model_paths["vae_path"], model_info['vae_name'])
                        )
                        print(f"Loading VAE: {model_info['vae_name']}")
                        if str(vae_path).endswith(".gguf"):
                            sd, extra = load_gguf_sd(str(vae_path), handle_prefix=None)
                            metadata = extra.get("metadata", {})
                        else:
                            sd, metadata = comfy.utils.load_torch_file(str(vae_path), return_metadata=True)
                    vae = comfy.sd.VAE(sd=sd, metadata=metadata)


                    if model_info['audio_vae_name'] in ["pixel_space", None]:
                        audio_vae = None
                    else:
                        audio_vae_path = path_manager.get_folder_file_path(
                            "vae",
                            model_info['audio_vae_name'],
                            default = os.path.join(path_manager.model_paths["vae_path"], model_info['audio_vae_name'])
                        )
                        print(f"Loading Audio VAE: {model_info['audio_vae_name']}")
                        if str(audio_vae_path).endswith(".gguf"):
                            sd, extra = load_gguf_sd(str(audio_vae_path), handle_prefix=None)
                            metadata = extra.get("metadata", {})
                        else:
                            sd, metadata = comfy.utils.load_torch_file(str(audio_vae_path), return_metadata=True)

                        # https://github.com/kijai/ComfyUI-KJNodes/blob/main/nodes/nodes.py#L2453C1-L2476C22
                        modeltype = model_info.get("unet_type", "")
                        match modeltype:
                            case "MiniMaxH3":
                                try:
                                    meta = metadata.get("minimax_h3_audio_vae", {})
                                    if isinstance(meta, str):
                                        meta = json.loads(meta)
                                    kwargs = {"minimax_h3_audio_vae": meta.get("kwargs", {})}
                                except:
                                    print(f"WARNING: unable to parse metadata: {metadata}")
                                    kwargs = {}
                                #audio_vae = VAE(sd=sd, metadata=kwargs)
                                metadata = {}
                                audio_vae = VAE(sd=sd, metadata=metadata)
                            case "LTXVA":
                                from comfy.ldm.lightricks.vae.audio_vae import AudioVAE
                                audio_vae = AudioVAE(sd, metadata)
                            case _:
                                sd_audio = comfy.utils.state_dict_prefix_replace(
                                    dict(sd), {"audio_vae.": "autoencoder.", "vocoder.": "vocoder."}, filter_keys=True
                                )
                                audio_vae = VAE(sd=sd_audio, metadata=metadata)
                                audio_vae.throw_exception_if_invalid()

                    clip_vision = None
                except Exception as e:
                    unet = None
                    self.load_error = str(e)
                    print(f"Could not load video model: {e}")

        else:
            try:
                with torch.torch.inference_mode():
                    unet, clip, vae, clip_vision = load_checkpoint_guess_config(filename)

                if clip is None or vae is None:
                    return self.load_base_model(filename, unet_only=True)
            except:
                print(f"Trying to load as unet.")
                self.load_base_model(
                    filename,
                    unet_only=True
                )
                return

#            print(f"DEBUG: load aio?")
#            sd = None
#            unet = None
#            try:
#                with torch.torch.inference_mode():
#                    sd = comfy.utils.load_torch_file(filename)
#            except Exception as e:
#                # Failed loading
#                print(f"ERROR: Failed loading {filename}: {e}")
#
#            print(f"DEBUG: sd1: {type(sd)}")
#            try:
#                diffusion_model_prefix = comfy.sd.model_detection.unet_prefix_from_state_dict(sd.copy())
#                parameters = comfy.utils.calculate_parameters(sd, diffusion_model_prefix)
#                if parameters == 0:
#                    sd = comfy.sd.load_diffusion_model(filename)
#            except:
#                sd = None
#                pass
#
#            print(f"DEBUG: sd2: {type(sd)}")
#            if sd is not None:
#                # Try to load as All-In-One checkpoint
#                try:
#                    aio = load_state_dict_guess_config(sd.copy())
#                    #aio = load_checkpoint_guess_config(sd.copy())
#                except Exception as e:
#                    print(f"DEBUG: error: {e}")
#                    aio = None
#                print(f"DEBUG: aio: {aio}")
#                if isinstance(aio, tuple):
#                    unet, clip, vae, clip_vision = aio
#
#                    if (
#                        isinstance(unet, ModelPatcher) and
#                        isinstance(clip, CLIP) and
#                        isinstance(vae, VAE)
#                    ):
#                        # If we got here, we have all models. Dump sd since we don't need it
#                        sd = None
#                    else:
#                        if isinstance(unet, ModelPatcher):
#                            sd = unet.clone()
#
#                if sd is not None:
#                    # We got something, assume it was a unet
#                    # Re-run load_base_model to get text-encoders and vae
#                    self.load_base_model(
#                        name,
#                        hash=hash,
#                        unet_only=True,
#                        input_unet=sd,
#                    )
#                    return
#

            else:
                unet = None

        if unet == None:
            print(f"Failed to load {name}")
            self.model_base = None
            self.model_hash = None
            self.model_base_patched = None
            self.model_patched_hash = None
        else:
            self.model_base = self.StableDiffusionModel(
                unet=unet, clip=clip, vae=vae, audio_vae=audio_vae
            )
            if not (self.model_base.unet.model.__class__.__name__ in self.known_model_info.keys()):
                print(
                    f"Model {self.model_base.unet.model.__class__.__name__} not supported. RuinedFooocus supports {list(self.known_model_info.keys())} models as video model."
                )
                self.model_base = None

            if self.model_base is not None:
                self.model_hash = hash if hash is not None else name
                self.model_base_patched = self.model_base
                self.model_patched_hash = None
                self.model_info = self.get_clip_and_vae(self.model_base_patched.unet, filename)

        return

    def load_loras(self, loras):
        loaded_loras = []

        model = self.model_base

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
                raise FileNotFoundError(f"Could not find video LoRA: {name}")

            print(f"Loading LoRAs: {name}")
            try:
                lora = comfy.utils.load_torch_file(str(filename), safe_load=True)
                unet, clip = comfy.sd.load_lora_for_models(
                    model.unet, model.clip, lora, weight, weight
                )
                model = self.StableDiffusionModel(
                    unet=unet,
                    clip=clip,
                    vae=model.vae,
                    audio_vae=model.audio_vae,
                )
                loaded_loras += [(name, weight)]
            except Exception as error:
                raise RuntimeError(f"Could not load video LoRA {name}: {error}") from error
        self.model_base_patched = model
        if self.model_hash_patched != str(loras):
            self.conditions = None
        self.model_hash_patched = str(loras)

        print(f"LoRAs loaded: {loaded_loras}")

        return

    def refresh_controlnet(self, name=None):
        return

    def clean_prompt_cond_caches(self):
        return

    conditions = None

    def textencode(self, id, text, clip_skip):
        update = False
        hash = f"{text} {clip_skip}"
        if hash != self.conditions[id]["text"]:
            self.conditions[id]["cache"] = CLIPTextEncode().encode(
                clip=self.model_base_patched.clip, text=text
            )[0]
        self.conditions[id]["text"] = hash
        update = True
        return update

    @torch.inference_mode()
    def process(
        self,
        gen_data=None,
        callback=None,
    ):
        if self.model_base_patched is None:
            raise RuntimeError(getattr(self, "load_error", None) or "Video model is not loaded. Check its required components.")

        # Setup

        seed = gen_data["seed"] if isinstance(gen_data["seed"], int) else random.randint(1, 2**32)
        gen_data["frame_rate"] = float(gen_data.get("video_fps", self.model_info.get("options", {}).get("fps", settings.default_settings.get("video_fps", 30.0)))) # Get fps from options, or settings
        frames = int(gen_data["original_image_number"] * gen_data["frame_rate"]) # Generate "Frame number" seconds of video
        multiple = self.model_info["frame_multiple"]
        offset = self.model_info["frame_offset"]
        frame_number = max(offset, ((frames - offset + multiple - 1) // multiple) * multiple + offset)
        from modules.video_settings import is_ltx25_distilled
        two_stage = (self.model_info["unet_type"] == "LTXAV2.5" and
                     is_ltx25_distilled("LTXV 2.5", gen_data.get("base_model_name", "")))
        spatial = self.model_info["spatial_multiple"] * (2 if two_stage else 1)
        gen_data["width"] = max(spatial, (gen_data["width"] // spatial) * spatial)
        gen_data["height"] = max(spatial, (gen_data["height"] // spatial) * spatial)
        if self.model_info["unet_type"] == "MiniMaxH3":
            from comfy_extras.nodes_minimax_h3 import adapt_canvas
            gen_data["width"], gen_data["height"] = adapt_canvas(gen_data["width"], gen_data["height"])


        print(f"Using {self.model_info['unet_type']} to generate video.")
        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Processing text encoding ...", "html/logo.png")
            )

        if self.conditions is None:
            self.conditions = clean_prompt_cond_caches()

        positive_prompt = gen_data["positive_prompt"]
        negative_prompt = gen_data["negative_prompt"]
        clip_skip = 1

        if two_stage:
            from modules.ltx_refinement import UPSCALER, refine
            def download_progress(received, total, speed):
                worker.check_interrupt(gen_data)
                percent = int(100 * received / total) if total else 0
                worker.add_result(gen_data["task_id"], "preview",
                                  (percent, f"Downloading LTX upscaler: {percent}% · {speed / 1048576:.1f} MB/s", None))
            upscaler_path = path_manager.get_folder_file_path("latent_upscalers", UPSCALER, progress=download_progress)
        stage_width = gen_data["width"] // 2 if two_stage else gen_data["width"]
        stage_height = gen_data["height"] // 2 if two_stage else gen_data["height"]

        # Get text_encoding

        with TimeIt("Text encoding"):
            print("Encoding prompts.")
            if self.model_info["unet_type"] != "MiniMaxH3":
                self.textencode("+", positive_prompt, clip_skip)
                if float(gen_data["cfg"]) != 1.0:
                    self.textencode("-", negative_prompt, clip_skip)

        negative_conditioning = self.conditions["-"]["cache"] if float(gen_data["cfg"]) != 1.0 else []

        with TimeIt("Setting up latents"):
            print("Setting up latents and getting ready to sample.")
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Get initial latents ...", None)
            )

            # MiniMax builds its own audiovisual latent and image conditioning below.
            if gen_data["input_image"] is not None and self.model_info["unet_type"] != "MiniMaxH3":
                image = np.array(gen_data["input_image"]).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                (positive, negative, video_latent) = LTXVImgToVideo().generate(
                    positive = self.conditions["+"]["cache"],
                    negative = negative_conditioning,
                    image = image,
                    vae = self.model_base_patched.vae,
                    width = stage_width,
                    height = stage_height,
                    length = frame_number,
                    batch_size = 1,
                    strength = 1,
                )
            else:
                positive = self.conditions["+"]["cache"]
                negative = negative_conditioning
                modeltype = self.model_base_patched.unet.model.__class__.__name__
                match modeltype:
                    case 'LTXAV' | 'LTXAV2.5':
                        video_latent = EmptyLTXVLatentVideo().generate(
                            width = stage_width,
                            height = stage_height,
                            length = frame_number,
                            batch_size = 1,
                        )[0]
                    case "MiniMaxH3":
                        video_latent = None

            audio_latent = None
            if self.model_info["unet_type"] in ("LTXAV", "LTXAV2.5") and "need_audio_latent" in self.model_info.get("flags", []):
                if self.model_info["audio_vae_name"] is not None:
                    audio_latent = LTXVEmptyLatentAudio().execute(
                        audio_vae = self.model_base_patched.audio_vae,
                        frames_number = frame_number,
                        frame_rate = gen_data["frame_rate"],
                        batch_size = 1,
                    )[0]

            if audio_latent is None:
                latent = video_latent
            else:
                # Combine audio and video
                latent = LTXVConcatAVLatent().execute(
                    video_latent = video_latent,
                    audio_latent = audio_latent,
                )[0]

        # Conditioning
        modeltype = self.model_base_patched.unet.model.__class__.__name__
        match modeltype:
            case 'LTXAV' | 'LTXAV2.5':
                positive, negative = LTXVConditioning().execute(
                    positive = positive,
                    negative = negative,
                    frame_rate = gen_data["frame_rate"],
                )
            case "MiniMaxH3":
                positive, latent = MiniMaxH3ImageToVideo().execute(
                    clip = self.model_base_patched.clip,
                    vae = self.model_base_patched.vae,
                    prompt = positive_prompt,
                    width = gen_data["width"],
                    height = gen_data["height"],
                    length = frame_number,
                    first_frame=(torch.from_numpy(np.asarray(gen_data["input_image"], dtype=np.float32) / 255.0)[None]
                                 if gen_data["input_image"] is not None else None),
                )
                # outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],

                negative = None
            case _:
                print(f"ERROR: Couldn't find conditioning for: {modeltype}")
                return

        # Sampler
        with TimeIt("Sampling"):
            print("Get sigmas.")

            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Getting simgas ...", None)
            )

            ksampler = KSamplerSelect().get_sampler(
                sampler_name = gen_data["sampler_name"],
            )[0]

            # Sigmas
            scheduler = self.model_info.get("options", {}).get("scheduler", "")
            if two_stage:
                scheduler = "LTXDistilled"
            match scheduler:
                case "LTXDistilled":
                    ksampler = KSamplerSelect().get_sampler("euler_ancestral")[0]
                    # Lightricks' fixed first-stage schedule for the distilled model.
                    sigmas = torch.tensor([1.0, 0.99375, 0.9875, 0.98125, 0.975,
                                           0.909375, 0.725, 0.421875, 0.0])
                case "LTXVScheduler":
                    sigmas = LTXVScheduler().execute(
                        steps = gen_data["steps"],
                        max_shift = 2.05,
                        base_shift = 0.95,
                        stretch = True,
                        terminal = 0.1,
                        latent = latent
                    )[0]
                case _:
                    sigmas = BasicScheduler().execute(
                        model = self.model_base_patched.unet,
                        scheduler = gen_data["scheduler"],
                        steps = gen_data["steps"],
                        denoise = 1.0,
                    )[0]
    
            # Guider
            opt_guider = self.model_info.get("options", {}).get("guider", "None")
            match opt_guider:
                case "CFGGuider":
                    guider = CFGGuider().execute(
                        model = self.model_base_patched.unet,
                        cfg = float(gen_data["cfg"]),
                        positive = positive,
                        negative = negative,
                    )[0]
                case "BasicGuider":
                    guider = BasicGuider().execute(
                        model = self.model_base_patched.unet,
                        conditioning = positive,
                    )[0]
                case _:
                    print(f"ERROR: Couldn't find guider: {opt_guider}")
                    return

            noise = Noise_RandomNoise(seed)
    
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Generating ...", None)
            )

            #
            # Sample
            #

            latent_image = latent["samples"]
            latent = latent.copy()
            latent_image = fix_empty_latent_channels(guider.model_patcher, latent_image, latent.get("downscale_ratio_spacial", None))
            latent["samples"] = latent_image

            noise_mask = None
            if "noise_mask" in latent:
                noise_mask = latent["noise_mask"]
    

            print("Sampling")
            callback_function = video_callback(self.model_base_patched.unet, gen_data,
                                              "Generating video (stage 1/2)" if two_stage else "Generating video")
            samples = guider.sample(
                noise.generate_noise(latent),
                latent_image,
                ksampler,
                sigmas,
                denoise_mask=noise_mask,
                callback=callback_function,
                disable_pbar=False,
                seed=noise.seed,
            )
            samples = samples.to(comfy.model_management.intermediate_device())

        if two_stage:
            latent = refine({**latent, "samples": samples}, guider, self.model_base_patched.vae,
                            upscaler_path, gen_data, seed)
            samples = latent["samples"]

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"VAE Decoding ...", None)
            )

        # Preserve latent metadata when preparing the sampler output for decoding.
        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = samples
        denoised_output = out

        video_samples = denoised_output
        audio_samples = None
        match self.model_info['unet_type']:
            case 'LTXAV' | 'LTXAV2.5':
                if audio_latent is not None:
                    samples = LTXVSeparateAVLatent().execute(
                        av_latent = denoised_output,
                    )
                    video_samples = samples[0]
                    audio_samples = samples[1]
            case "MiniMaxH3":
                samples = denoised_output["samples"]
                video_samples = {"samples": samples.tensors[0]}
                if len(samples.tensors) >= 2:
                    audio_samples = {"samples": samples.tensors[1]}
                else:
                    audio_samples = None

        worker.check_interrupt(gen_data)
        comfy.model_management.unload_model_and_clones(guider.model_patcher)
        worker.check_interrupt(gen_data)

        # Decode video

        print(f"VAE decode video.")
        decoded_latent = VAEDecodeTiled().decode(
            samples=video_samples,
            tile_size=512,
            overlap=64,
            temporal_size=64,
            temporal_overlap=16,
            vae=self.model_base_patched.vae,
        )[0]


        if audio_samples is not None and self.model_info['audio_vae_name'] is not None:
            # Decode audio
            print(f"VAE decode audio.")
            try:
                audio = VAEDecodeAudio().execute(
                    samples = audio_samples,
                    vae = self.model_base_patched.audio_vae,
                )[0]
            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()
                audio = None
        else:
            audio = None

        # Create Video
        video = CreateVideo().execute(
            images = decoded_latent,
            audio = audio,
            fps = gen_data["frame_rate"],
        )[0]

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Saving ...", None)
            )

        filename = generate_temp_filename(
            folder=path_manager.model_paths["temp_outputs_path"], extension="tmp"
        )
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        print("Saving video")
        # Save MP4
        codec = "auto"
        try:
            loras = []
            for lora_data in gen_data["loras"] if gen_data["loras"] is not None else []:
                if len(lora_data[0]) == 64 and all(c in '0123456789abcdefABCDEF' for c in lora_data[0]): # Looks like sha256?
                    hash = lora_data[0]
                else:
                    hash = None
                w, l  = lora_data[1].split(" - ", 1)
                if not l == "None":
                    loras.append({"name": l, "weight": float(w), "hash": hash})
            data = {
                "Prompt": gen_data["positive_prompt"],
                "Negative": gen_data["negative_prompt"],
                "steps": gen_data["steps"],
                "cfg": gen_data["cfg"],
                "width": gen_data["width"],
                "height": gen_data["height"],
                "seed": abs(int(gen_data["seed"])),
                "sampler_name": gen_data["sampler_name"],
                "scheduler": gen_data["scheduler"],
                "base_model_name": gen_data["base_model_name"],
                "base_model_hash": get_checkpoint_hashes(gen_data["base_model_name"])['SHA256'],
                "loras": [[f"{get_lora_hashes(lora['name'])['SHA256']}", f"{lora['weight']} - {lora['name']}"] for lora in loras],
                "software": "RuinedFooocus",
            }
        except:
            data = {"prompt": gen_data["positive_prompt"], "software": "RuinedFooocus"}
        metadata = {"metadata": json.dumps(data)}

        video.save_to(
            filename.with_suffix(".mp4"),
            format = Types.VideoContainer.MP4,
            codec = Types.VideoCodec(codec),
            metadata = metadata
        )

        pil_images = []
        for image in decoded_latent:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            pil_images.append(img)

        # Save GIF
        compress_level=9 # Min = 0, Max = 9
        pil_images[0].save(
            filename.with_suffix(".gif"),
            compress_level=compress_level,
            comment=json.dumps(data).encode("utf-8"),
            save_all=True,
            duration=int(1000.0/gen_data["frame_rate"]),
            append_images=pil_images[1:],
            optimize=True,
            loop=0,
        )

        return [str(filename.with_suffix(".gif"))]
