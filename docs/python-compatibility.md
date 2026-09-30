# Python compatibility

Gradio 6.28.0 requires Python 3.10 or newer. GPU wheels must also support
the chosen Python version.

The 6.28.0 update was checked on Windows/Python 3.12 with full application
startup, `pip check`, and responsive browser checks. The older Python results
below belong to 6.27.0; they have not been rerun with 6.28.0.
See [UI layout](ui-layout.md) for viewport coverage.

For a GTX 1080 Ti, use Python 3.12 and `TORCH_PLATFORM=cu124`. The official
PyTorch 2.6.0 CUDA 12.4 wheels include `sm_61` (Pascal) and are available for
Python 3.10–3.13 on Windows and Linux x86_64. They are not available for
Python 3.14. Keep CUDA 12.4 for this GPU; CUDA 13 drops Pascal support.

Validation with Gradio 6.27.0:

| Python | Checks |
| --- | --- |
| 3.10.21 | Full app startup and browser UI with PyTorch 2.6.0+cu124 in CPU mode |
| 3.12.10 | Full app, browser UI, Qwen chat and Janku generation on AMD ROCm |
| 3.13.15 | Python syntax and API contract tests |
| 3.14.7 | Full app startup and browser UI in CPU mode |

Syntax and API contract tests passed on all four versions. Those earlier checks
used AMD or CPU execution. Python 3.14 startup does not imply
that every optional GPU backend supports it.

A subsequent clean Windows RTX 4060 test on Python 3.12.10 verified CUDA 13.2
PyTorch, JANKU generation, and native/xllamacpp GPU chat. See the dated
[NVIDIA setup results](gpu-setup.md#verified-windows-nvidia-setup-2026-09-30).

Sources: [Gradio](https://pypi.org/project/gradio/6.28.0/),
[CUDA 12.4 wheels](https://download.pytorch.org/whl/cu124/torch/),
[CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html).

## ComfyUI compatibility

Automatic CPU offload is the default. `--gpu-only` keeps models on the GPU;
`--lowvram` allows more offload; `--cpu-vae` decodes on CPU.
Large images use tiled VAE decoding when the estimated memory exceeds the GPU
budget, including `--reserve-vram`. Output dimensions stay unchanged.
CLIP Skip 1 preserves the native encoder layer.

Backend revisions are pinned in `launch.py`: ComfyUI `7193f562` and
molbal/ComfyUI-GGUF `c6e14d9`. GGUF diffusion loading uses the backend's
quantization-aware loader and preserves model metadata.

Backend imports and GGUF loader tests pass on Python 3.10 / PyTorch 2.6 cu124
and Python 3.14. Janku image generation and Wan 2.1 Q8 video generation were
tested on Windows with an RX 7900 XTX. The subsequent RTX 4060 JANKU check is recorded in [GPU setup](gpu-setup.md).
PyAV 17.1 is retained for Python 3.10; Python 3.11+ uses PyAV 18.1.
