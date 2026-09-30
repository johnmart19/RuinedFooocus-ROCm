# NVIDIA validation and AMD handoff

Reviewed 2026-09-30 against release `ac83de9` (26.09.30). This review is on
`codex/nvidia-validation`, separate from `development`. It changes CUDA detection
and validation, not AMD packages or the saved local configuration. Keep the code
fix and this evidence together when integrating the branch.

## What is covered

| Hardware | Automatic stable CUDA candidates, newest first | Evidence / limitation |
| --- | --- | --- |
| Maxwell: GTX 750/900, corresponding Quadro/Tesla; CC 5.x | 12.6, 12.4 | Architecture selection tested with simulated adapters; no physical test |
| Pascal: GTX 10, P-series; CC 6.x | 12.6, 12.4 | Includes GTX 1080 Ti; Python wheel availability verified, not hardware execution |
| Volta: V100/TITAN V; CC 7.0 | 12.6, 12.4 | Simulated selection only |
| Turing: GTX 16/RTX 20/T4; CC 7.5 | 13.2, 13.0, 12.8, 12.6, 12.4 | Simulated selection only |
| Ampere / Ada / Hopper: RTX 30/40, A/L/H-series; CC 8.x/9.0 | 13.2, 13.0, 12.8, 12.6, 12.4 | RTX 4060 execution below; other devices simulated |
| Blackwell: RTX 50/RTX PRO/B200/B300; CC 10.x/12.0 | 13.2, 13.0, 12.8 | Simulated selection only; runtime kernels still need validation |
| GB10/DGX Spark CC 12.1, CC 11.0 devices | 13.2, 13.0 | Architecture routing only; ARM64/Jetson setup and all application dependencies unverified |
| Kepler/Fermi/Tesla and earlier, CC below 5.0 | None | Detected hardware gets an explicit legacy-environment error; no automatic CUDA support |

Candidates are additionally filtered by driver CUDA ceiling and the running
interpreter's wheel tags. This is not a claim that every listed card has enough
VRAM, supports every model/dtype, or has been physically tested. Laptop and
workstation names do not need a separate allowlist when `nvidia-smi` reports their
capability. Unknown capability, missing driver information and absent wheels
stop installation with a diagnostic. CPU mode must be explicitly requested when
known NVIDIA hardware cannot run a supported CUDA bundle.

PyTorch 2.14 is the last planned release with CUDA 12.6 support for
Maxwell/Pascal/Volta. Do not replace these pins with the latest CUDA unconditionally.
See [PyTorch's release architecture matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#pytorch-cuda-support-matrix),
[NVIDIA's current GPU table](https://developer.nvidia.com/cuda/gpus),
[legacy table](https://developer.nvidia.com/cuda/gpus/legacy), and
[CUDA 12.6 retirement notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432).

## Multi-GPU and explicit selection

Installation now intersects the compatible bundles for every selected adapter.
For example, Pascal plus Ada can use CUDA 12.6; Pascal plus Blackwell has no common
bundled runtime. Native CUDA chat performs the same architecture checks.
Explicit `TORCH_PLATFORM` reinstall requests are checked before pip replaces wheels.
Existing compatible installed bundles are still retained.

Prefer UUIDs to numeric indices when choosing a GPU:

```bat
nvidia-smi -L
set "CUDA_VISIBLE_DEVICES=GPU-<full-UUID-from-nvidia-smi>"
RuinedFooocus.bat
```

In PowerShell use `$env:CUDA_VISIBLE_DEVICES = "GPU-<UUID>"`; in Linux/WSL use
`export CUDA_VISIBLE_DEVICES=GPU-<UUID>`. Multiple UUIDs may be comma-separated.
For numeric selection on a multi-GPU host (including `--gpu-device-id`), set
`CUDA_DEVICE_ORDER=PCI_BUS_ID` first; indices then follow PCI bus order. The
installer rejects ambiguous default FASTEST_FIRST ordering instead of guessing.
These are CUDA controls, not Vulkan adapter selection. MIG partitions and Jetson
vendor environments are not validated by this installer and need separate setup.
See [NVIDIA's environment-variable specification](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html).

## Rechecks performed for this branch

- 47 focused simulated cases: architecture routing, driver limits, mixed GPUs,
  UUID/numeric visibility, invalid selectors, legacy/unknown capability, explicit
  reinstall choices, and native CUDA chat validation. Python 3.10 grammar and
  compilation checked; this is not execution under Python 3.10.
- 42 existing AMD/NVIDIA/CPU/DirectML isolation and override cases passed for
  simulated Windows and Linux paths. AMD probe/install logic was not changed.
- Official PyTorch indexes contain the pinned Torch/TorchVision/TorchAudio
  bundles for CUDA 12.4, 12.6, 12.8, 13.0 and 13.2 on Windows/Linux x86_64 for
  CPython 3.10, 3.12 and 3.13. The newer bundles use the existing CPU TorchAudio
  2.11 stable-ABI companion. Availability does not prove full application compatibility.
- On the actual RTX 4060: new detection reported CC 8.9 and driver CUDA 13.4;
  fresh selection was cu132. Installed CUDA arithmetic/native-extension validation
  passed with Torch 2.14.0+cu132, TorchVision 0.29.0+cu132 and TorchAudio 2.11.0+cpu.
- Native CUDA 13.3 backend validation passed; the pinned xllamacpp CUDA wheel was
  available and its installed backend enumerated CUDA. `pip check` passed.
- No drivers/packages were replaced and no running generation was interrupted.
  No new physical Pascal, Blackwell, Linux NVIDIA or mixed-GPU test was performed.
  The earlier generation/chat results below were inspected, not rerun for this branch.

Ad hoc checks and raw local logs remain in ignored `tmp/local-tests/`, not in Git.

### Verified Windows NVIDIA setup (2026-09-30)

A clean RTX 4060 8 GB installation with driver 617.14 and Python 3.12.10
automatically selected PyTorch 2.14.0 + CUDA 13.2. A CUDA matrix calculation,
the 20-test regression suite and `pip check` passed. The driver reported CUDA
UMD 13.4; that did not require a nightly PyTorch installation.

Qwen 2.5 7B Q4_K_M answered a text question and emitted a valid requested image
tool call with native llama.cpp b10917 on both Vulkan (Auto) and CUDA 13.3,
and xllamacpp 2026.9.10809 on CUDA. Runtimes were tested sequentially and unloaded
between runs. These short checks are functional tests, not comparative benchmarks. AMD
Windows RX 7900 XTX execution was verified in the earlier review; a new AMD
clean installation was not repeated on this NVIDIA machine.

JANKU v7.77 generated an 832×1216 image through the API using **Illustrious XL**:
30 steps, CFG 5, Euler ancestral, normal schedule and CLIP Skip 2. The PNG's
metadata confirmed those settings. The request completed in approximately 26 seconds;
the inspected image followed the main prompt details, and a subsequent API chat
request succeeded. The same profile and seed were then verified through the
Gradio Generate button, including the rendered preview and saved metadata.
No additional LoRAs or embeddings were used.

This validates this RTX 4060 configuration, not Pascal, WSL, DirectML or every
video/vision workflow. Keep the CUDA 12.4 guidance for older Pascal GPUs.

## Resume on the AMD PC

1. Fetch `development` and `codex/nvidia-validation`; compare before integrating.
   Do not copy the NVIDIA virtual environment, CUDA DLLs, or runtime cache to AMD.
   Reuse the AMD machine's own environment and official GPU driver.
2. Check `TORCH_PLATFORM`, `CUDA_VISIBLE_DEVICES`, `CUDA_DEVICE_ORDER`,
   `LLAMA_SERVER` and saved chat backend selections. Remove NVIDIA-specific
   overrides; choose Auto/Vulkan or the validated ROCm chat backend on AMD.
3. Check `nvidia-smi` is not causing detection from only a leftover driver header.
   On Linux/WSL AMD, verify `rocminfo` and `libnuma1`; on Windows verify the
   installed compatible ROCm bundle. Follow [GPU setup](gpu-setup.md).
4. Start the existing prepared environment without Git/package updates using
   `python launch.py --offline --nobrowser --port 7863`. Confirm the log reports
   ROCm for image generation and the intended chat backend. Reinstall markers
   remain pending offline; inspect those requests before the next online launch.
5. Repeat a small JANKU image with the recorded Illustrious XL recipe, a normal
   chat and requested image tool call on native llama.cpp and xllamacpp, then
   Load/Unload and cancellation recovery. Test Wan/LTX only with the AMD machine's
   available models and suitable memory budget. These remain AMD-side checks.

The separately developed guest/LAN privacy feature is still uncommitted in the
NVIDIA working checkout. It is not included in this branch or the release baseline;
transfer that work separately if needed. Do not assume fetching this branch brings
those changes, local settings, downloaded models, private prompts or outputs.
