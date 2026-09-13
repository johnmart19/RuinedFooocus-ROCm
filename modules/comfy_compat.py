"""Compatibility with the pinned ComfyUI backend."""

import os
import platform


def configure_torch_allocator(torch_platform, os_platform):
    # ROCDXG cannot transfer model-sized tensors using expandable allocations.
    wsl_rocm = (os_platform == "Linux" and torch_platform.startswith("rocm")
                and "microsoft" in platform.release().lower())
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ.setdefault("PYTORCH_ALLOC_CONF",
                              "expandable_segments:False" if wsl_rocm else "expandable_segments:True")


def configure_torch_compatibility():
    import torch
    import typing

    if not (2, 4) <= tuple(int(v) for v in torch.__version__.split(".")[:2]) <= (2, 6):
        return
    # Torch 2.4–2.6 understand typing.List but not list[T] in custom-op schemas.
    # Newer comfy-kitchen uses the latter; both describe the same tensor operation.
    from torch._library.infer_schema import SUPPORTED_PARAM_TYPES, SUPPORTED_RETURN_TYPES
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
