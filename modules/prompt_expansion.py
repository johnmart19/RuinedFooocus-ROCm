from modules.resource_defaults import resource_name
from .util import remove_empty_str
from comfy.model_patcher import ModelPatcher
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import comfy.model_management as model_management
from transformers.generation.logits_process import LogitsProcessorList
from molbal_comfyui_gguf.nodes import DualCLIPLoaderGGUF
from comfy.sd import CLIPType, load_clip
from shared import path_manager, settings
import os
import random
import torch


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

        print(f"Loading Erniehancer: {self.clip_names}")
        if all(name.endswith(".safetensors") for name in self.clip_paths):
            model_options = {}
            device = model_management.get_torch_device()
            model_options["load_device"] = model_options["offload_device"] = device
            self.clip = load_clip(ckpt_paths=self.clip_paths, clip_type=self.clip_type, model_options=model_options)
        else:
            clip_loader = DualCLIPLoaderGGUF()
            self.clip = clip_loader.load_patcher(
                self.clip_paths,
                self.clip_type,
                clip_loader.load_data(self.clip_paths)
            )

    def execute(self, prompt, image=None, thinking=False):
        tokens = self.clip.tokenize(prompt, image=image, skip_template=False, min_length=1, thinking=thinking)

        # Get sampling parameters
        do_sample = True
        max_length = 512
        temperature = 0.6
        top_k = 64
        top_p = 0.8
        min_p = 0.05
        repetition_penalty = 1.5
        presence_penalty = 0.0
        seed = random.randint(1, 2**32)

        generated_ids = self.clip.generate(
            tokens,
            do_sample=do_sample,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seed=seed
        )

        generated_text = self.clip.decode(generated_ids, skip_special_tokens=True)
        print(f"Erniehanced prompt:\n{generated_text}")
        return generated_text
