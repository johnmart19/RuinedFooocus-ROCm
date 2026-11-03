import os
import numpy as np
import torch
from pathlib import Path
import folder_paths
import modules.async_worker as worker
import modules.controlnet
from modules.model_identity import file_signature
from shared import path_manager


class InterruptibleUpscaler:
    """Check the current job between tiles without changing ComfyUI globals."""

    def __init__(self, model, gen_data):
        self.model = model
        self.gen_data = gen_data

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, image):
        worker.check_interrupt(self.gen_data)
        return self.model(image)


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
        signature = file_signature(model_path)
        if signature == getattr(self, 'upscaler_file_signature', None):
            return self.upscaler_model
        from comfy_extras.nodes_upscale_model import UpscaleModelLoader
        model_path = Path(model_path)
        folder_paths.add_model_folder_path("upscale_models", str(model_path.parent), is_default=True)
        out = UpscaleModelLoader.execute(model_path.name)[0]
        self.upscaler_model = out
        self.upscaler_file_signature = signature
        return out

    def load_base_model(self, name, hash=None):
        # Check if model is already loaded
        if self.model_hash == name:
            return
        print(f"Loading model: {name}")
        self.model_hash = name
        return

    def load_loras(self, loras):
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
        input_image = input_image.convert("RGBA" if input_image.has_transparency_data else "RGB")
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
            raise ValueError("Choose an upscaler in Image tools.")

        upscaler_model = self.load_upscaler_model(upscaler_name)
        worker.add_result(gen_data["task_id"], "preview", (-1, "Upscaling image ...", None))
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
        decoded = ImageUpscaleWithModel.execute(
            InterruptibleUpscaler(upscaler_model, gen_data), input_image)[0]
        worker.check_interrupt(gen_data)
        return [np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
                for image in decoded]
