"""Recovered 8B MiniMax-H3 encoder subset; see PROVENANCE.md and LICENSE."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file

import comfy.sd
import comfy.sd1_clip
import comfy.supported_models_base
import comfy.utils
import comfy.model_management
import folder_paths
from comfy.text_encoders.hunyuan_video import llama_detect
from comfy.text_encoders.minimax import MiniMaxQwen3VL
from comfy.text_encoders.qwen3vl import Qwen3VLSDTokenizer

ARA_TARGETS = ("self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(directory: Path, filename: str) -> dict[str, Any]:
    path = directory / filename
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid manifest: {path}")
    return value


def _verified(directory: Path, entry: dict[str, Any]) -> Path:
    path = (directory / entry["file"]).resolve()
    if path.parent != directory.resolve():
        raise ValueError(f"Model file must be inside its package: {path}")
    actual = _sha256(path)
    if actual != entry["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path.name}: {actual} != {entry['sha256']}")
    return path


def _submodule(root: Any, dotted_name: str) -> Any:
    current = root
    for part in dotted_name.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


class RecoveryLinear(nn.Module):
    def __init__(self, base: nn.Module, rank: int, alpha: float, device: torch.device, prefix: str = "lora"):
        super().__init__()
        self.base = base
        self.prefix = prefix
        setattr(self, f"{prefix}_down", nn.Linear(base.in_features, rank, bias=False, device=device, dtype=torch.float32))
        setattr(self, f"{prefix}_up", nn.Linear(rank, base.out_features, bias=False, device=device, dtype=torch.float32))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        down = getattr(self, f"{self.prefix}_down")
        up = getattr(self, f"{self.prefix}_up")
        delta = up(down(x.float())) * self.scaling
        return base_output + delta.to(dtype=base_output.dtype)


class ConditioningAdapter(nn.Module):
    def __init__(self, input_width: int, output_width: int, bottleneck: int, device: torch.device):
        super().__init__()
        self.norm = nn.RMSNorm(input_width, device=device, dtype=torch.float32)
        self.proj = nn.Linear(input_width, output_width, bias=False, device=device, dtype=torch.float32)
        self.down = nn.Linear(input_width, bottleneck, bias=False, device=device, dtype=torch.float32)
        self.up = nn.Linear(bottleneck, output_width, bias=False, device=device, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(hidden.float())
        return self.proj(normalized) + self.up(torch.nn.functional.silu(self.down(normalized)))


class RecoveredMiniMaxTextModel(MiniMaxQwen3VL):
    model_type = "qwen3vl_8b"

    def __init__(self, config_dict, dtype, device, operations):
        super().__init__({"num_hidden_layers": 24, "final_norm": False, "lm_head": False}, dtype, device, operations)
        # ponytail: text-only package; tokenizer rejects image/reference inputs.
        del self.visual
        adapter = config_dict["adapter"]
        for layer in config_dict["ara_layers"]:
            for suffix in ARA_TARGETS:
                parent_name, leaf = f"layers.{layer}.{suffix}".rsplit(".", 1)
                parent = _submodule(self.model, parent_name)
                base = getattr(parent, leaf)
                setattr(parent, leaf, RecoveryLinear(base, config_dict["ara_rank"], config_dict["ara_alpha"], device))
        self.adapter = ConditioningAdapter(adapter["input_width"], adapter["output_width"], adapter["bottleneck"], device)

    def forward(self, input_ids, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None,
                final_layer_norm_intermediate=True, dtype=None, embeds_info=None, **kwargs):
        result = super().forward(
            input_ids, attention_mask=attention_mask, embeds=embeds, num_tokens=num_tokens,
            intermediate_output=None, final_layer_norm_intermediate=False, dtype=dtype,
            embeds_info=embeds_info or [], **kwargs,
        )
        return self.adapter(result[0]), None


def _recovered_clip_model(recovery: dict[str, Any], quantization_metadata: dict[str, Any] | None):
    class RecoveredClipModel(comfy.sd1_clip.SDClipModel):
        def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options=None):
            options = dict(model_options or {})
            if quantization_metadata is not None:
                options["quantization_metadata"] = quantization_metadata
            super().__init__(
                device=device, layer="last", layer_idx=None, textmodel_json_config={
                    "adapter": recovery["adapter"], "ara_layers": recovery["ara"]["layers"],
                    "ara_rank": recovery["ara"]["rank"], "ara_alpha": recovery["ara"]["alpha"],
                }, dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False,
                model_class=RecoveredMiniMaxTextModel, enable_attention_masks=False,
                return_attention_masks=False, model_options=options,
            )

        def encode_token_weights(self, token_weight_pairs):
            output = super().encode_token_weights(token_weight_pairs)
            tags = getattr(self.transformer, "last_token_tags", None)
            if tags is None:
                tags = torch.ones(output[0].shape[1], dtype=torch.long)
            extra = output[2] if len(output) > 2 and isinstance(output[2], dict) else {}
            extra["minimax_token_tags"] = tags
            return output[0], output[1], extra
    return RecoveredClipModel


def _recovered_te_model(recovery: dict[str, Any], quantization_metadata: dict[str, Any] | None):
    clip_model = _recovered_clip_model(recovery, quantization_metadata)
    class RecoveredTEModel(comfy.sd1_clip.SD1ClipModel):
        def __init__(self, device="cpu", dtype=None, model_options=None):
            super().__init__(device=device, dtype=dtype, name="qwen3vl_8b", clip_model=clip_model, model_options=model_options or {})
    return RecoveredTEModel


class RecoveredMiniMaxTokenizer(comfy.sd1_clip.SD1Tokenizer):
    """The official H3 verbatim text token contract, without vision inputs."""
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        tokenizer = lambda *args, **kwargs: Qwen3VLSDTokenizer(*args, **kwargs, embedding_size=4096, embedding_key="qwen3vl_8b")
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data or {}, name="qwen3vl_8b", tokenizer=tokenizer)

    def tokenize_with_weights(self, text, return_word_ids=False, images=None, minimax_ref_items=None, **kwargs):
        if images or minimax_ref_items:
            raise ValueError("Recovered MiniMax-H3 encoders support text-only T2V. Do not use image or reference inputs.")
        tokens = [(token, 1.0) for token in self.qwen3vl_8b.tokenizer(text, add_special_tokens=False)["input_ids"]]
        if not tokens:
            tokens = [(151643, 1.0)]
        if return_word_ids:
            tokens = [token + (0,) for token in tokens]
        return {"qwen3vl_8b": [tokens]}


def _translate_recovered_base(state: dict[str, torch.Tensor], ara_layers: list[int]) -> dict[str, torch.Tensor]:
    translated: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        output = key
        for layer in ara_layers:
            for suffix in ARA_TARGETS:
                prefix = f"model.layers.{layer}.{suffix}."
                if key.startswith(prefix):
                    output = prefix + "base." + key.removeprefix(prefix)
                    break
        translated[output] = value
    return translated


def _load_recovered(directory: Path, package: dict[str, Any], variant_id: str):
    recovery = {
        "ara": package["ara"],
        "adapter": package["adapter"],
    }
    variant = package["base_variants"][variant_id]
    base_path = _verified(directory, variant)
    adapter_path = _verified(directory, recovery["adapter"])
    ara_path = _verified(directory, recovery["ara"])
    state, metadata = comfy.utils.load_torch_file(str(base_path), safe_load=True, return_metadata=True)
    state, _ = comfy.utils.convert_old_quants(state, metadata=metadata)
    quantization_metadata = llama_detect(state)["llama_quantization_metadata"] if any(
        key.endswith(".comfy_quant") for key in state
    ) else None
    target = comfy.supported_models_base.ClipTarget(
        RecoveredMiniMaxTokenizer, _recovered_te_model(recovery, quantization_metadata)
    )
    # The base, ARA, and adapter state is loaded below. Keep the empty model on
    # CPU during construction; otherwise CLIP forces GPU placement before the
    # custom RecoveryLinear wrappers have populated weights.
    clip = comfy.sd.CLIP(
        target,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        parameters=6_000_000_000,
        model_options={"initial_device": comfy.model_management.text_encoder_offload_device()},
    )
    state = _translate_recovered_base(state, recovery["ara"]["layers"])
    adapter = {f"adapter.{key}": value for key, value in load_file(str(adapter_path), device="cpu").items()}
    ara = {f"model.{key}": value for key, value in load_file(str(ara_path), device="cpu").items()}
    state.update(adapter)
    state.update(ara)
    missing, unexpected = clip.load_sd(state, full_model=False)
    real_missing = [key for key in (missing or []) if not key.endswith("lm_head.weight")]
    if real_missing or unexpected:
        raise RuntimeError(f"Recovered encoder load mismatch: missing={real_missing[:8]} unexpected={list(unexpected or [])[:8]}")
    return clip
