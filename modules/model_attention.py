"""Use memory-bounded Qwen attention on Windows ROCm."""

import platform


def configure_minimax_workspace(patcher, quantized=False):
    """Keep workspace for H3's long sequences on 24GB Windows ROCm GPUs."""
    # The measured margin addressed the large safetensors model. Applying it
    # to smaller GGUFs would force unnecessary offloading before measurement.
    if quantized:
        return False
    import torch
    from comfy import model_management
    from comfy.cli_args import args
    model = patcher.model
    if model.__class__.__name__ != "MiniMaxH3" or getattr(model, "_rf_workspace_reserved", False):
        return False
    if platform.system() != "Windows" or not torch.version.hip or args.cpu:
        return False
    device = model_management.get_torch_device()
    capacity = model_management.get_total_memory(device)
    if not 20 * 1024**3 <= capacity <= 26 * 1024**3:
        return False
    native_memory = model.memory_required
    model.memory_required = lambda *a, **kw: native_memory(*a, **kw) + 6 * 1024**3
    model._rf_workspace_reserved = True
    print("MiniMax H3: reserving 6 GiB of additional GPU workspace.")
    return True


def configure_qwen_attention(patcher):
    import torch
    from comfy import model_management
    from comfy.cli_args import args
    from comfy.ldm.modules.attention import attention_sub_quad

    if patcher.model.__class__.__name__ != "QwenImage":
        return False
    if platform.system() != "Windows" or not torch.version.hip or args.cpu:
        return False
    choices = (
        "use_pytorch_cross_attention",
        "use_split_cross_attention",
        "use_quad_cross_attention",
        "use_sage_attention",
        "use_flash_attention",
        "use_ck_attention",
    )
    if any(getattr(args, name, False) for name in choices):
        return False
    options = patcher.model_options.setdefault("transformer_options", {})
    if "optimized_attention_override" in options:
        return False

    def bounded_attention(original, *inputs, **kwargs):
        return attention_sub_quad(*inputs, **kwargs)

    options["optimized_attention_override"] = bounded_attention
    # Match the backend's non-flash workspace estimate while keeping the text
    # encoder's native attention. This is a per-model adjustment, not global.
    # Add workspace margin for WDDM allocations not reported as Torch tensors.
    if (
        model_management.pytorch_attention_flash_attention()
        or model_management.xformers_enabled()
    ):
        dtype_size = model_management.dtype_size(patcher.model.get_dtype_inference())
        patcher.model.memory_usage_factor *= 1.25 * 0.15 / (0.01 * dtype_size)
    print("Qwen Image: using memory-bounded attention for Windows ROCm.")
    return True
