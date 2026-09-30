# Python compatibility

Gradio 6.29.0 requires Python 3.10 or newer. GPU wheels must also support
the chosen Python version.

The September 30 dependency update was installed and checked on Windows:

| Python / runtime | Checks |
| --- | --- |
| 3.10.21 / Torch 2.6.0 cu124, CPU execution | Full UI/API startup, backend imports, dependency resolution |
| 3.12.10 / Torch 2.14.0 cu132, RTX 4060 | Full UI/API, JANKU image and short Wan GGUF video generation, chat/tool calls, upscaling, media codecs |
| 3.12.10 / Torch 2.4.1 DirectML | Full UI/API startup, backend imports, GPU arithmetic |
| 3.14.7 / Torch 2.14.0 CPU | Full UI/API startup, backend imports, dependency resolution |

Dependency resolution also passed for Linux x86_64 Python 3.10/cu124 and
3.14/CPU, and Windows Python 3.11/cu124. These are resolver checks, not Linux
runtime tests.

These checks do not establish Pascal or AMD hardware execution for this update,
or generation support for every model. See [UI layout](ui-layout.md) for browser
coverage and [GPU setup](gpu-setup.md) for runtime selection.

For an existing GTX 1080 Ti CUDA 12.4 setup, use Python 3.12 and
`TORCH_PLATFORM=cu124`. The official
PyTorch 2.6.0 CUDA 12.4 wheels include `sm_61` (Pascal) and are available for
Python 3.10–3.13 on Windows and Linux x86_64. They are not available for
Python 3.14. Automatic setup can also select compatible CUDA 12.6 wheels;
CUDA 13 drops Pascal support.

Historical validation with Gradio 6.27.0 (before the current backend pins):

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
[NVIDIA setup results](nvidia-support.md#verified-nvidia-configuration).

Sources: [Gradio](https://pypi.org/project/gradio/6.29.0/),
[CUDA 12.4 wheels](https://download.pytorch.org/whl/cu124/torch/),
[CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html).

## ComfyUI compatibility

Automatic CPU offload is the default. `--gpu-only` keeps models on the GPU;
`--lowvram` allows more offload; `--cpu-vae` decodes on CPU.
Large images use tiled VAE decoding when the estimated memory exceeds the GPU
budget, including `--reserve-vram`. Output dimensions stay unchanged.
CLIP Skip 1 preserves the native encoder layer.

Backend revisions are pinned in `launch.py`: ComfyUI `fb2315f1` and
molbal/ComfyUI-GGUF `48de657`. GGUF diffusion loading uses the backend's
quantization-aware loader and preserves model metadata.

The current pins passed backend imports on Python 3.10 / PyTorch 2.6 cu124 and
Python 3.14; JANKU and short Wan 2.1 Q8 generation passed on RTX 4060. Earlier
Windows RX 7900 XTX checks used the previous backend pins and do not validate
this update on AMD. See [NVIDIA validation](nvidia-support.md#verified-nvidia-configuration).
PyAV 17.1 is retained for Python 3.10, 18.1 for Python 3.11, and 19.0 for
Python 3.12+. Kornia, rembg, ONNX Runtime, websockets and pygit2 also retain
older-Python builds through requirement markers.

## Dependency compatibility limits

- Gradio's MCP extra requires Pydantic 2.12.5; its matching core is resolved
  together instead of independently pinning an incompatible newer core.
- DirectML requires NumPy 1.26.4 and SciPy 1.15.3. Bounded OpenCV/rembg
  requirements allow their compatible older builds while other runtimes can
  use current releases.
- Transformers 5.17 requires Torch 2.5 or newer. The installer constrains
  Torch 2.4 installations to Transformers 5.13 and resolves its matching
  Tokenizers version. This also applies to module reinstallation.
- Existing working CUDA/ROCm Torch bundles stay constrained during application
  upgrades. A newer application package must not replace vendor wheels.

See [PyPI package metadata](https://pypi.org/) and the pinned backend
[requirements](https://github.com/Comfy-Org/ComfyUI/blob/fb2315f11db0ebfaafa9099a5df5227dc6bb42bc/requirements.txt).
The standalone ComfyUI frontend and workflow-template packages are not used by
RuinedFooocus's embedded backend and are not installed as UI dependencies.
