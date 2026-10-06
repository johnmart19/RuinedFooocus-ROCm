"""Ask the installed backend to classify tensor shapes without loading weights."""


def detect_tensor_architecture(header):
    import torch
    from comfy import model_detection

    tensors = {}
    for name, entry in header.items():
        if name == "__metadata__" or not isinstance(entry, dict):
            continue
        shape = entry.get("shape")
        if (not isinstance(shape, list) or len(shape) > 8
                or any(type(size) is not int or size < 0 or size > 2**31 for size in shape)):
            continue
        try:
            tensors[name] = torch.empty(shape, device="meta")
        except (RuntimeError, OverflowError):
            continue
    if not tensors:
        return None
    prefixes = dict.fromkeys((model_detection.unet_prefix_from_state_dict(tensors),
                              "", "model.diffusion_model.", "net.", "model."))
    for prefix in prefixes:
        try:
            config = model_detection.detect_unet_config(tensors, prefix)
            if config is None:
                continue
            supported = model_detection.model_config_from_unet_config(config, tensors, prefix)
            if supported is not None:
                name = type(supported).__name__
                return {"SD15": "SD 1.5", "SD20": "SD 2.0", "SDXL": "SDXL 1.0",
                        "Flux": "Flux.1 D", "FluxSchnell": "Flux.1 S",
                        "PixArtAlpha": "PixArt", "PixArtSigma": "PixArt",
                        "QwenImage21": "QwenImage"}.get(name, name)
        except (KeyError, IndexError, ValueError, TypeError, RuntimeError, AttributeError):
            # Partial/unsupported headers are a normal metadata miss.
            continue
    return None
