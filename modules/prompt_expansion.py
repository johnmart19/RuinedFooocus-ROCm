from modules.resource_defaults import resource_name
from .util import remove_empty_str
from comfy.model_patcher import ModelPatcher
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import comfy.model_management as model_management
from transformers.generation.logits_process import LogitsProcessorList
from molbal_comfyui_gguf.nodes import DualCLIPLoaderGGUF
from comfy.sd import CLIPType, load_clip
from comfy_extras.nodes_textgen import TextGenerate
from modules.model_identity import file_signature, read_safetensors_header
from shared import path_manager, settings
import os
import json
import random
import torch
from math import prod


fooocus_expansion_path = "prompt_expansion"

SEED_LIMIT_NUMPY = 2**32
neg_inf = -8192.0


def safe_str(x):
    x = str(x)
    for _ in range(16):
        x = x.replace("  ", " ")
    return x.strip(",. \r\n")


class FooocusExpansion:
    tokenizer = None
    model = None

    def __init__(self):
        self.load_model_and_tokenizer(fooocus_expansion_path)
        self.offload_device = model_management.text_encoder_offload_device()
        self.patcher = ModelPatcher(
            self.model,
            load_device=self.model.device,
            offload_device=self.offload_device,
        )

    @classmethod
    def load_model_and_tokenizer(cls, model_path):
        if cls.tokenizer is None or cls.model is None:
            cls.tokenizer = AutoTokenizer.from_pretrained(model_path)
            cls.model = AutoModelForCausalLM.from_pretrained(model_path)
            cls.model.to("cpu")

    def __call__(self, prompt, seed):
        seed = int(seed) % SEED_LIMIT_NUMPY
        text = safe_str(prompt) + ","
        tokenized_kwargs = self.tokenizer(text, return_tensors="pt")
        tokenized_kwargs.data["input_ids"] = tokenized_kwargs.data["input_ids"].to(
            self.patcher.load_device
        )
        tokenized_kwargs.data["attention_mask"] = tokenized_kwargs.data[
            "attention_mask"
        ].to(self.patcher.load_device)
        current_token_length = int(tokenized_kwargs.data["input_ids"].shape[1])
        max_new_tokens = min(75 - current_token_length % 75,
                             self.model.config.max_position_embeddings - current_token_length)
        if max_new_tokens <= 0:
            return prompt
        set_seed(seed)
        with open(os.path.join(fooocus_expansion_path, "positive.txt"), encoding="utf-8") as source:
            positive_words = {"Ġ" + word.lower() for word in source.read().splitlines() if word}
        vocab = self.tokenizer.get_vocab()
        self.logits_bias = torch.full((1, len(vocab)), neg_inf, dtype=torch.float32)
        allowed = [vocab[word] for word in positive_words if word in vocab]
        self.logits_bias[0, allowed] = 0
        try:
            features = self.model.generate(
                **tokenized_kwargs,
                top_k=100,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                logits_processor=LogitsProcessorList([self.logits_processor])
            )

            response = self.tokenizer.batch_decode(features[:, current_token_length:], skip_special_tokens=True)
            suffix = safe_str(response[0])
            result = prompt + (", " + suffix if suffix else "")
        except Exception as e:
            print(f"Error during prompt expansion: {e}")
            result = prompt
        return result

    def logits_processor(self, input_ids, scores):
        assert scores.ndim == 2 and scores.shape[0] == 1
        self.logits_bias = self.logits_bias.to(scores)

        bias = self.logits_bias.clone()
        bias[0, input_ids[0].to(bias.device).long()] = neg_inf
        bias[0, 11] = 0
        return scores + bias


class PromptExpansion:
    # Define the expected input types for the node
    @staticmethod
    @torch.no_grad()
    def expand_prompt(text):
        expansion = FooocusExpansion()

        prompt = remove_empty_str([text], default="")[0]

        max_seed = int(1024 * 1024 * 1024)
        seed = random.randint(1, max_seed)
        if seed < 0:
            seed = -seed
        seed = seed % max_seed

        expansion_text = expansion(prompt, seed)
        final_prompt = expansion_text

        return final_prompt


class Erniehancer:
    _cache = None

    @staticmethod
    def _load_clip(paths, clip_type, options):
        device = options['load_device']
        low_precision = any(getattr(model_management.args, key, False) for key in
                            ('fp16_text_enc', 'bf16_text_enc', 'fp8_e4m3fn_text_enc', 'fp8_e5m2_text_enc'))
        if device.type == 'cpu' and len(paths) == 1 and not low_precision:
            header = read_safetensors_header(paths[0])
            tensors = {key: value for key, value in header.items() if key != '__metadata__'}
            unquantized = bool(tensors) and not header.get('__metadata__') and all(
                value['dtype'] in ('F16', 'BF16', 'F32') or (key == 'tekken_model' and value['dtype'] == 'U8')
                for key, value in tensors.items())
            fp32_bytes = sum(prod(value['shape']) * 4 for value in tensors.values()
                             if value['dtype'] in ('F16', 'BF16', 'F32'))
            if unquantized and fp32_bytes * 2.5 < model_management.get_free_memory(device):
                from comfy.utils import load_torch_file
                from comfy.sd import load_text_encoder_state_dicts
                state = load_torch_file(paths[0], safe_load=True)
                # BF16/FP16 values promote exactly; disk weights remain unchanged.
                state = {key: value.float() if value.dtype in (torch.float16, torch.bfloat16) else value
                         for key, value in state.items()}
                return load_text_encoder_state_dicts([state], clip_type=clip_type,
                    model_options=options, disable_dynamic=True)
        return load_clip(ckpt_paths=paths, clip_type=clip_type, model_options=options)

    def __init__(self):
        self.clip = None
        self.clip_names = []
        self.clip_paths = []
        self.clip_type = CLIPType.FLUX2

        clip_name = resource_name("clip_ernie_enhancer", settings.default_settings)
        self.clip_names.append(str(clip_name))
        clip_path = path_manager.get_folder_file_path(
            "clip",
            clip_name,
            default = os.path.join(path_manager.model_paths["clip_path"], clip_name)
        )
        self.clip_paths.append(str(clip_path))

        device = model_management.text_encoder_device()
        offload = model_management.text_encoder_offload_device()
        key = (tuple(tuple(file_signature(path)) for path in self.clip_paths), self.clip_type,
               str(device), str(offload), str(model_management.text_encoder_dtype(device)),
               tuple(getattr(model_management.args, name, False) for name in
                     ('fp16_text_enc', 'bf16_text_enc', 'fp8_e4m3fn_text_enc', 'fp8_e5m2_text_enc')))
        cached = type(self)._cache
        if cached is not None and cached[0] == key:
            self.clip = cached[1]
            return

        print(f"Loading Erniehancer: {self.clip_names}")
        if all(name.endswith(".safetensors") for name in self.clip_paths):
            self.clip = self._load_clip(self.clip_paths, self.clip_type,
                {'load_device': device, 'offload_device': offload})
        else:
            clip_loader = DualCLIPLoaderGGUF()
            self.clip = clip_loader.load_patcher(
                self.clip_paths,
                self.clip_type,
                clip_loader.load_data(self.clip_paths)
            )
        type(self)._cache = (key, self.clip)

    def execute(self, prompt, image=None, thinking=False):
        # Ernie's tokenizer is raw: the instruction delimiters are supplied by the caller.
        instruction = ('You are an image prompt enhancer. Expand the supplied prompt into one concise visual '
                       'description in the same language. Preserve every requested subject, exact count, color, '
                       'spatial relationship, exclusion and quoted text. Add only compatible lighting, texture '
                       'and composition details. Output only the enhanced description, with no explanation.')
        request = '[SYSTEM_PROMPT]' + instruction + '[/SYSTEM_PROMPT][INST]' + json.dumps(
            {'prompt': prompt}, ensure_ascii=False) + '[/INST]'
        text = TextGenerate.execute(clip=self.clip, prompt=request, image=image, thinking=thinking,
            max_length=512, sampling_mode=dict(sampling_mode='on', temperature=.6, top_k=64,
                top_p=.8, min_p=.05, repetition_penalty=1.05, presence_penalty=0.,
                seed=random.randint(1, 2**32)))[0]
        return text or prompt
