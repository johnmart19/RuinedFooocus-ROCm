import os
import traceback
import numpy as np
import torch
import modules.async_worker as worker
import modules.controlnet
from shared import path_manager
import comfy.utils
from spandrel import ModelLoader, ImageModelDescriptor
from spandrel_extra_arches import install as install_extra_architectures
from comfy import model_management
from comfy.model_patcher import CoreModelPatcher
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

class pipeline:
    pipeline_type = ["upscale"]
    model_hash = ""

    def parse_gen_data(self, gen_data):
        gen_data["original_image_number"] = gen_data["image_number"]
        gen_data["image_number"] = 1
        gen_data["show_preview"] = False
        return gen_data

    def load_upscaler_model(self, model_name):
        model_path = path_manager.get_file_path(
            model_name,
            default=os.path.join(path_manager.model_paths["upscaler_path"], model_name),
        )
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Upscaler not found: {model_name}")
        sd = comfy.utils.load_torch_file(str(model_path), safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
        install_extra_architectures(ignore_duplicates=True)
        out = ModelLoader().load_from_state_dict(sd).eval()
        if not isinstance(out, ImageModelDescriptor):
            raise ValueError("Select a single-image upscaler.")
        out.patcher = CoreModelPatcher(
            out.model, load_device=model_management.get_torch_device(),
            offload_device=model_management.unet_offload_device(),
        )
        return out

    def load_base_model(self, name, hash=None):
        # Check if model is already loaded
        if self.model_hash == name:
            return
        print(f"Loading model: {name}")
        self.model_hash = name
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
        return

    def refresh_controlnet(self, name=None):
        return

    def clean_prompt_cond_caches(self):
        return

    @torch.inference_mode()
    def process(
        self,
        gen_data=None,
        callback=None,
    ):
        input_image = gen_data["input_image"]
        if input_image is None:
            raise ValueError("Choose an input image for upscaling.")
        input_image = input_image.convert("RGB")
        input_image = np.array(input_image).astype(np.float32) / 255.0
        input_image = torch.from_numpy(input_image)[None,]

        worker.add_result(
            gen_data["task_id"],
            "preview",
            (-1, f"Load upscaling model ...", None)
        )

        cn_settings = modules.controlnet.get_settings(gen_data)
        upscaler_name = cn_settings.get("upscaler")
        if not upscaler_name or upscaler_name == "None":
            raise ValueError("Choose an upscaler in PowerUp.")

        try:
            upscaler_model = self.load_upscaler_model(upscaler_name)

            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Upscaling image ...", None)
            )
            decoded_latent = ImageUpscaleWithModel().upscale(
                upscaler_model, input_image
            )[0]

            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Converting ...", None)
            )
            images = [
                np.clip(255.0 * y.cpu().numpy(), 0, 255).astype(np.uint8)
                for y in decoded_latent
            ]
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Done ...", None)
            )
        except:
            traceback.print_exc()
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Oops ...", "html/error.png")
            )
            images =  []

        return images
