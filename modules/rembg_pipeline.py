import modules.async_worker as worker
from argparser import args
from modules.onnx_runtime import new_session

import rembg
import PIL.Image
from typing import Any


class pipeline:
    def remove_background(
        self,
        image: PIL.Image.Image,
        rembg_session: Any = None,
        force: bool = False,
        check_cancelled=None,
        **rembg_kwargs,
    ) -> PIL.Image.Image:
        do_remove = True
        if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
            do_remove = False
        do_remove = do_remove or force
        if do_remove:
            if rembg_session is None:
                if getattr(self, '_rembg_session', None) is None:
                    self._rembg_session = new_session(cpu=args.cpu)
                rembg_session = self._rembg_session
            if hasattr(rembg_session, 'check_cancelled'):
                rembg_session.check_cancelled = check_cancelled
            try:
                image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
            finally:
                if hasattr(rembg_session, 'check_cancelled'):
                    rembg_session.check_cancelled = None
        return image

    pipeline_type = ["rembg"]
    model_hash = ""

    # Optional function
    def parse_gen_data(self, gen_data):
        gen_data["original_image_number"] = gen_data["image_number"]
        gen_data["image_number"] = 1
        gen_data["show_preview"] = False
        if gen_data.get('input_image') is not None:
            gen_data['width'], gen_data['height'] = gen_data['input_image'].size
        return gen_data

    def load_base_model(self, name, hash=None):
        return

    def load_loras(self, loras):
        return

    def process(
        self,
        gen_data=None,
        callback=None,
    ):

        worker.add_result(
            gen_data["task_id"],
            "preview",
            (-1, f"Removing background ...", None)
        )

        if gen_data["input_image"] is None:
            print(f"ERROR: Could not find input image.")
            return ["html/error.png"]

        worker.check_interrupt(gen_data)
        image = self.remove_background(gen_data["input_image"],
                                       check_cancelled=lambda: worker.check_interrupt(gen_data))
        worker.check_interrupt(gen_data)

        # Return finished image to preview
        if callback is not None:
            callback(1, 0, 0, 1, image)

        return [image]
