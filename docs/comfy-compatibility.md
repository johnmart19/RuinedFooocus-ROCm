# ComfyUI compatibility

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
tested on Windows with an RX 7900 XTX. NVIDIA hardware was not available.
PyAV 17.1 is retained for Python 3.10; Python 3.11+ uses PyAV 18.1.
