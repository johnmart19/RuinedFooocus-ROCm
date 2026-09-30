from modules.video_preview import video_callback
import numpy as np
import os
import torch
from modules.gguf_loader import load_diffusion_model as load_gguf_model
import traceback
import cv2

import modules.async_worker as worker
from modules.util import generate_temp_filename
from PIL import Image

import os
from comfy.model_base import LTXV
from shared import path_manager, settings
import shared

from pathlib import Path
import random
from modules.pipeline_utils import (
    clean_prompt_cond_caches,
)

import comfy.utils
from comfy.sd import load_checkpoint_guess_config
from tqdm import tqdm

from molbal_comfyui_gguf.nodes import gguf_sd_loader as load_gguf_sd, DualCLIPLoaderGGUF

from nodes import (
    CLIPTextEncode,
    DualCLIPLoader,
    VAEDecodeTiled,
    VAEDecode,
)

from comfy_extras.nodes_custom_sampler import SamplerCustom, RandomNoise, BasicScheduler, KSamplerSelect, BasicGuider
from comfy_extras.nodes_lt import EmptyLTXVLatentVideo, LTXVImgToVideo, LTXVConditioning, LTXVScheduler
from comfy_extras.nodes_lt import ModelSamplingLTXV
from comfy_extras.nodes_flux import FluxGuidance


class pipeline:
    pipeline_type = ["ltx_video"]

    class StableDiffusionModel:
        def __init__(self, unet, vae, clip, clip_vision):
            self.unet = unet
            self.vae = vae
            self.clip = clip
            self.clip_vision = clip_vision

        def to_meta(self):
            if self.unet is not None:
                self.unet.model.to("meta")
            if self.clip is not None:
                self.clip.cond_stage_model.to("meta")
            if self.vae is not None:
                self.vae.first_stage_model.to("meta")

    model_hash = ""
    model_base = None
    model_hash_patched = ""
    model_base_patched = None
    conditions = None


    # Optional function
    def parse_gen_data(self, gen_data):
        gen_data["original_image_number"] = gen_data.get("video_duration", gen_data["image_number"])
        gen_data["image_number"] = 1
        gen_data["show_preview"] = False
        return gen_data

    def load_base_model(self, name, unet_only=True, hash=None): # LTXV never has the clip and vae models?
        # Check if model is already loaded
        if self.model_hash == name:
            return

        self.model_base = None
        self.model_hash = ""
        self.model_base_patched = None
        self.model_hash_patched = ""
        self.conditions = None

        default = None

        filename = str(
            shared.models.get_model_path(
                "checkpoints",
                name,
                hash=hash,
                default=default,
            )
        )

        print(f"Loading LTX video {'unet' if unet_only else 'model'}: {name}")

        if filename.endswith(".gguf") or unet_only:
            with torch.torch.inference_mode():
                try:
                    if filename.endswith(".gguf"):
                        unet = load_gguf_model(filename)
                    else:
                        # Let ComfyUI choose a supported dtype for this device.
                        model_options = {}
                        unet = comfy.sd.load_diffusion_model(filename, model_options=model_options)

                    clip_paths = []
                    clip_names = []

                    if isinstance(unet.model, LTXV):
                        clip_name = settings.default_settings.get("clip_t5", "t5-v1_1-xxl-encoder-Q3_K_S.gguf")
                        clip_names.append(str(clip_name))
                        clip_path = path_manager.get_folder_file_path(
                            "clip",
                            clip_name,
                            default = os.path.join(path_manager.model_paths["clip_path"], clip_name)
                        )
                        clip_paths.append(str(clip_path))
                        clip_type = comfy.sd.CLIPType.HUNYUAN_VIDEO

                        vae_name = settings.default_settings.get("vae_ltxv", "LTX-Video-0.9.6-VAE-BF16.safetensors")

                    else:
                        print(f"ERROR: Not a LTX Video model?")
                        unet = None
                        return

                    print(f"Loading CLIP: {clip_names}")
                    clip_type = comfy.sd.CLIPType.LTXV
                    if all(name.endswith(".safetensors") for name in clip_paths):
                        model_options = {}
                        device = comfy.model_management.get_torch_device()
                        if device == "cpu":
                            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
                        clip = comfy.sd.load_clip(ckpt_paths=clip_paths, clip_type=clip_type, model_options=model_options)
                    else:
                        clip_loader = DualCLIPLoaderGGUF()
                        clip = clip_loader.load_patcher(
                            clip_paths,
                            clip_type,
                            clip_loader.load_data(clip_paths)
                        )

                    vae_path = path_manager.get_folder_file_path(
                        "vae",
                        vae_name,
                        default = os.path.join(path_manager.model_paths["vae_path"], vae_name)
                    )

                    print(f"Loading VAE: {vae_name}")
                    if str(vae_path).endswith(".gguf"):
                        sd, extra = load_gguf_sd(str(vae_path), handle_prefix=None)
                        metadata = extra.get("metadata", {})
                    else:
                        sd, metadata = comfy.utils.load_torch_file(str(vae_path), return_metadata=True)
                    vae = comfy.sd.VAE(sd=sd, metadata=metadata)

                    clip_vision = None
                except Exception as e:
                    unet = None
                    traceback.print_exc() 

        else:
            try:
                with torch.torch.inference_mode():
                    unet, clip, vae, clip_vision = load_checkpoint_guess_config(filename)

                if clip == None or vae == None:
                    raise
            except:
                print(f"Failed. Trying to load as unet.")
                self.load_base_model(
                    filename,
                    unet_only=True
                )
                return

        if unet == None:
            print(f"Failed to load {name}")
            self.model_base = None
            self.model_hash = ""
        else:
            self.model_base = self.StableDiffusionModel(
                unet=unet, clip=clip, vae=vae, clip_vision=clip_vision
            )
            if not (
                isinstance(self.model_base.unet.model, LTXV)
            ):
                print(
                    f"Model {type(self.model_base.unet.model)} not supported. Expected LTX Video model."
                )
                self.model_base = None

            if self.model_base is not None:
                self.model_hash = name
                print(f"Base model loaded: {self.model_hash}")
        return

    def load_keywords(self, lora):
        filename = lora.replace(".safetensors", ".txt")
        try:
            with open(filename, "r") as file:
                data = file.read()
            return data
        except FileNotFoundError:
            return " "

    def load_loras(self, loras):
        loaded_loras = []

        model = self.model_base
#        for name, weight in loras:
#            if name == "None" or weight == 0:
#                continue
#            filename = str(shared.models.get_file("loras", name))

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
                continue

            print(f"Loading LoRAs: {name}")
            try:
                lora = comfy.utils.load_torch_file(filename, safe_load=True)
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
            except:
                pass
        self.model_base_patched = model
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
        seed = gen_data["seed"] if isinstance(gen_data["seed"], int) else random.randint(1, 2**32)

        fps = float(gen_data.get("video_fps", settings.default_settings.get("fps", 30)))
        gen_data["frames"] = 1 + (int(gen_data["original_image_number"] * fps / 4.0) * 4)

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

        self.textencode("+", positive_prompt, clip_skip)
        self.textencode("-", negative_prompt, clip_skip)

        callback_function = video_callback(self.model_base_patched.unet, gen_data)

        # latent_image
        # t2v or i2v?
        if gen_data["input_image"]:
            image = np.array(gen_data["input_image"]).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            (positive, negative, latent_image) = LTXVImgToVideo().generate(
                positive = self.conditions["+"]["cache"],
                negative = self.conditions["-"]["cache"],
                image = image,
                vae = self.model_base_patched.vae,
                width = gen_data["width"],
                height = gen_data["height"],
                length = gen_data["frames"],
                batch_size = 1,
                strength = 1,
            )
        else:
            # latent_image
            latent_image = EmptyLTXVLatentVideo().generate(
                width = gen_data["width"],
                height = gen_data["height"],
                length = gen_data["frames"],
                batch_size = 1,
            )[0]
            positive = self.conditions["+"]["cache"]

        negative = self.conditions["-"]["cache"]

        # LTXVConditioning
        positive, negative = LTXVConditioning().execute(
            positive = positive,
            negative = negative,
            frame_rate = fps
        )

        # Sampler
        ksampler = KSamplerSelect().get_sampler(
            sampler_name = gen_data["sampler_name"],
        )[0]

        # Sigmas
        sigmas = LTXVScheduler().execute(
            steps = gen_data["steps"],
            max_shift = 2.05,
            base_shift = 0.95,
            stretch = True,
            terminal = 0.1,
            latent = latent_image
        )[0]

        worker.add_result(
            gen_data["task_id"],
            "preview",
            (-1, f"Generating ...", None)
        )

        samples = SamplerCustom().sample(
            model=self.model_base_patched.unet,
            add_noise=True,
            noise_seed=seed,
            cfg=float(gen_data["cfg"]),
            positive=positive,
            negative=negative,
            sampler=ksampler,
            sigmas=sigmas,
            latent_image=latent_image,
        )[0]

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"VAE Decoding ...", None)
            )

        decoded_latent = VAEDecodeTiled().decode(
            samples=samples,
            tile_size=512,
            overlap=64,
            temporal_size=64,
            temporal_overlap=5,
            vae=self.model_base_patched.vae,
        )[0]

        pil_images = []
        for image in decoded_latent:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pil_images.append(img)

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Saving ...", None)
            )

        file = generate_temp_filename(
            folder=path_manager.model_paths["temp_outputs_path"], extension="gif"
        )
        os.makedirs(os.path.dirname(file), exist_ok=True)

        compress_level=9 # Min = 0, Max = 9

        # Save GIF
        pil_images[0].save(
            file,
            compress_level=compress_level,
            save_all=True,
            duration=int(1000.0/fps),
            append_images=pil_images[1:],
            optimize=True,
            loop=0,
        )

        # Save mp4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        mp4_file = file.with_suffix(".mp4")
        out = cv2.VideoWriter(str(mp4_file), fourcc, fps, (gen_data["width"], gen_data["height"]))
        for frame in pil_images:
            out.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB))
        out.release()

        return [file]
