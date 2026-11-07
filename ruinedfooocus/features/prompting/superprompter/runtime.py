"""Optional T5 prompt expansion on the shared worker, with releasable weights."""

from pathlib import Path
import random
import threading

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from ruinedfooocus.core.paths import APP_ROOT, data_path
from ruinedfooocus.features.image_processing.auxiliary_resources import register, release_except
from ruinedfooocus.features.prompting.superprompter.download_models import download_models


class Runtime:
    resource_group = "superprompt"

    def __init__(self, folder=None):
        self.folder = Path(folder) if folder else data_path("superprompter", "model_files")
        legacy = APP_ROOT / "superprompter" / "model_files"
        if (
            folder is None
            and not (self.folder / "config.json").is_file()
            and (legacy / "config.json").is_file()
        ):
            self.folder = legacy
        self.model = self.tokenizer = None
        self.device = "cpu"

    def load(self):
        if self.model is not None:
            return
        from ruinedfooocus.app.arguments import args

        if not (self.folder / "config.json").is_file():
            download_models(self.folder)
        tokenizer = T5Tokenizer.from_pretrained(self.folder, local_files_only=True)
        model = T5ForConditionalGeneration.from_pretrained(
            self.folder,
            local_files_only=True,
            torch_dtype=torch.float32 if args.cpu or args.directml is not None else torch.float16,
        )
        self.tokenizer, self.model = tokenizer, model
        register(self)

    def close(self):
        if self.model is not None and self.device != "cpu":
            self.model.to("cpu")
            self.device = "cpu"

    def generate(self, input_text, **parameters):
        from ruinedfooocus.app.arguments import args

        release_except(self.resource_group)
        self.load()
        device = "cpu"
        if not args.cpu and args.directml is None:
            from comfy import model_management
            from ruinedfooocus.app import state as shared

            active = shared.state.get("pipeline")
            if active is not None and "llama" in getattr(active, "pipeline_type", ()):
                active.unload()
            model_management.unload_all_models()
            device = model_management.get_torch_device()
        self.device = str(device)
        try:
            self.model.to(device)
            seed = parameters.pop("seed")
            torch.manual_seed(random.randint(1, 1000000) if seed == -1 else seed)
            inputs = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
            outputs = self.model.generate(inputs, do_sample=True, **parameters)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        finally:
            # Image sampling may resume in this same worker job. Return VRAM
            # before that happens, retaining reusable CPU weights and files.
            self.close()


runtime = Runtime()


def answer(
    input_text="",
    max_new_tokens=512,
    repetition_penalty=1.2,
    temperature=0.5,
    top_p=1,
    top_k=1,
    seed=-1,
):
    from ruinedfooocus.app import async_worker as worker

    values = dict(
        input_text=input_text,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )
    if threading.current_thread() is worker.worker_thread:
        return runtime.generate(**values)
    from ruinedfooocus.app.api_runtime import submit, wait_for_result

    def operation(job, task_id):
        worker.check_interrupt({"_api_job": job})
        result = runtime.generate(**values)
        worker.check_interrupt({"_api_job": job})
        return result

    operation.resource_group = runtime.resource_group
    return wait_for_result(submit(operation))
