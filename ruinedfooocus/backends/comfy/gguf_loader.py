"""Load local diffusion GGUFs through the backend's quantization-aware loader."""

from pathlib import Path

import folder_paths
from molbal_comfyui_gguf.nodes import UnetLoaderGGUF


def compact_minimax_operations():
    """Keep mixed U16G Q4 layers packed while using native INT8 for Q8_CR."""
    from molbal_comfyui_gguf.ops import GGMLOps, get_gguf_q8_ops
    from molbal_comfyui_gguf.dequant import is_quantized
    native = get_gguf_q8_ops()

    class CompactOperations(native):
        class Linear(native.Linear):
            def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
                weight = state_dict.get(prefix + "weight")
                if (weight is not None and is_quantized(weight)
                        and prefix + "comfy_quant" not in state_dict):
                    # The upstream mixed loader materializes these Q4 matrices
                    # as BF16. Its ordinary GGML Linear already provides packed
                    # storage, patch support and transient dequantization.
                    self.__class__ = GGMLOps.Linear
                    self.weight_comfy_model_dtype = self.factory_kwargs["dtype"]
                    self.bias_comfy_model_dtype = self.factory_kwargs["dtype"]
                    return self._load_from_state_dict(state_dict, prefix, *args, **kwargs)
                return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
    return CompactOperations()


def load_diffusion_model(filename):
    path = Path(filename).resolve(strict=True)
    if path.name == "minimax_h3_fl2va_pruned_fp8_U16G.gguf":
        import comfy.sd
        from molbal_comfyui_gguf.loader import gguf_sd_loader
        from molbal_comfyui_gguf.nodes import GGUFModelPatcher
        state, extra = gguf_sd_loader(str(path))
        model = comfy.sd.load_diffusion_model_state_dict(
            state, model_options={"custom_operations": compact_minimax_operations()},
            metadata=extra.get("metadata", {}))
        if model is None:
            raise RuntimeError("The installed backend could not recognize MiniMax U16G.")
        model = GGUFModelPatcher.clone(model)
        model.patch_on_device = True
        return model
    folder_paths.add_model_folder_path("diffusion_models", str(path.parent), is_default=True)
    return UnetLoaderGGUF().load_unet(path.name, patch_on_device=True)[0]
