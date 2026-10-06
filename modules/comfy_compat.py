"""Compatibility with the pinned ComfyUI backend."""

import os
import platform


def configure_device_environment(args):
    """Apply GPU visibility before importing Torch, matching backend startup."""
    devices = args.cuda_device
    if devices is not None and devices != "all":
        if not all(part.isdecimal() for part in devices.split(",")):
            raise ValueError("--cuda-device expects comma-separated GPU indices or all")
    if args.default_device is not None and not 0 <= args.default_device < 32:
        raise ValueError("--default-device expects an index from 0 to 31")
    if devices is None and args.default_device is not None:
        order = list(range(32))
        order.remove(args.default_device)
        devices = ",".join(map(str, [args.default_device, *order]))
    if args.gpu_device_id is not None:
        if args.gpu_device_id < 0:
            raise ValueError("--gpu-device-id expects a non-negative index")
        devices = str(args.gpu_device_id)
    if devices is not None and devices != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        os.environ["HIP_VISIBLE_DEVICES"] = devices
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = devices
    if args.oneapi_device_selector is not None:
        os.environ["ONEAPI_DEVICE_SELECTOR"] = args.oneapi_device_selector
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def configure_comfy_runtime(launch_args, comfy_args):
    """Apply embedded-backend flags before model management is imported."""
    comfy_args.offline = launch_args.offline or os.environ.get("RF_OFFLINE") == "1"
    if comfy_args.offline:
        comfy_args.disable_partner_nodes = True
    from modules.comfy_args import COMFY_DESTINATIONS
    from comfy.cli_args import PerformanceFeature, LatentPreviewMethod

    # The embedded backend defaults to no previews; the app displays them.
    comfy_args.preview_method = LatentPreviewMethod.from_string(getattr(launch_args, "preview_method", "auto"))

    for name in COMFY_DESTINATIONS:
        setattr(comfy_args, name, getattr(launch_args, name))
    if launch_args.normalvram:
        comfy_args.disable_dynamic_vram = True
        comfy_args.enable_dynamic_vram = False
    if comfy_args.force_fp16:
        comfy_args.fp16_unet = True
    if comfy_args.disable_comfy_compiler:
        comfy_args.disable_cuda_graphs = True
    features = launch_args.fast
    comfy_args.fast = (
        set()
        if features is None
        else set(PerformanceFeature)
        if not features
        else {PerformanceFeature(x.value) for x in features}
    )


def initialize_comfy_memory():
    """Initialize the pinned DynamicVRAM controller without starting its server."""
    from comfy.cli_args import args, enables_dynamic_vram

    if enables_dynamic_vram():
        import comfy_aimdo.control

        headroom = (
            None if args.reserve_vram is None else int(args.reserve_vram * 1024**3)
        )
        comfy_aimdo.control.init(
            simple_vram_headroom=headroom, nvml_pressure=not args.disable_nvml_pressure
        )


def configure_torch_allocator(torch_platform, os_platform, launch_args=None):
    if launch_args is not None and (
        launch_args.cuda_malloc or launch_args.disable_cuda_malloc
    ):
        backend = "cudaMallocAsync" if launch_args.cuda_malloc else "native"
        config = os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_ALLOC_CONF", "")
        )
        parts = [
            part
            for part in config.split(",")
            if part and not part.startswith(("backend:", "expandable_segments:"))
        ]
        parts.extend([f"backend:{backend}", "expandable_segments:False"])
        os.environ["PYTORCH_ALLOC_CONF"] = ",".join(parts)
        if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ["PYTORCH_ALLOC_CONF"]
        return
    # ROCDXG cannot transfer model-sized tensors using expandable allocations.
    wsl_rocm = (
        os_platform == "Linux"
        and torch_platform.startswith("rocm")
        and "microsoft" in platform.release().lower()
    )
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ.setdefault(
            "PYTORCH_ALLOC_CONF",
            "expandable_segments:False" if wsl_rocm else "expandable_segments:True",
        )


def configure_torch_compatibility():
    import torch
    import typing

    if not (2, 4) <= tuple(int(v) for v in torch.__version__.split(".")[:2]) <= (2, 6):
        return
    # Torch 2.4–2.6 understand typing.List but not list[T] in custom-op schemas.
    # Newer comfy-kitchen uses the latter; both describe the same tensor operation.
    from torch._library.infer_schema import (
        SUPPORTED_PARAM_TYPES,
        SUPPORTED_RETURN_TYPES,
    )

    for supported in (SUPPORTED_PARAM_TYPES, SUPPORTED_RETURN_TYPES):
        for annotation, schema in list(supported.items()):
            if typing.get_origin(annotation) is list:
                supported.setdefault(list[typing.get_args(annotation)[0]], schema)
    if tuple(int(v) for v in torch.__version__.split(".")[:2]) == (2, 4):
        # DirectML's torch predates resolution of postponed custom-op annotations.
        import torch._custom_op.impl as custom_ops

        if not getattr(custom_ops.infer_schema, "_rf_annotations", False):
            original = custom_ops.infer_schema

            def infer_schema(fn, *args, **kwargs):
                annotations = fn.__annotations__
                try:
                    fn.__annotations__ = typing.get_type_hints(fn)
                    if fn.__annotations__.get("return") is type(None):
                        fn.__annotations__["return"] = None
                    return original(fn, *args, **kwargs)
                finally:
                    fn.__annotations__ = annotations

            infer_schema._rf_annotations = True
            custom_ops.infer_schema = infer_schema
