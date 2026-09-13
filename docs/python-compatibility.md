# Python compatibility

Gradio 6.27.0 requires Python 3.10 or newer. GPU wheels must also support
the chosen Python version.

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

Syntax and API contract tests passed on all four versions. No physical NVIDIA
GPU was available for generation testing. Python 3.14 startup does not imply
that every optional GPU backend supports it.

Sources: [Gradio](https://pypi.org/project/gradio/6.27.0/),
[CUDA 12.4 wheels](https://download.pytorch.org/whl/cu124/torch/),
[CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html).
