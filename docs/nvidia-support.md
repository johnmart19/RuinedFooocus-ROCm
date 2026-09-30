# NVIDIA support and validation handoff

This guide describes release 26.09.30, implementation
`3236d01ffdee2ac4e9dfd4603ec2c1ff2f56bf51`. The Windows evidence predates the WSL runtime-builder and
model-switch memory fixes; the sections below distinguish those later WSL checks.
History consolidation preserves the verified final source tree. Future changes
require fresh checks; versions here are a dated snapshot, not a perpetual support claim.

AMD setup is documented separately in [GPU setup](gpu-setup.md). NVIDIA does not
need ROCm packages or an AMD runtime. GPU architecture recognition is not proof
that every model fits in VRAM or that every supported GPU has been tested.

## Clean installation

### Windows

Install Git, 64-bit Python and the NVIDIA driver, then open a new terminal:

```bat
git clone --branch development https://github.com/johnmart19/RuinedFooocus-ROCm.git
cd RuinedFooocus-ROCm
RuinedFooocus.bat
```

The batch launcher respects `PYTHON`, an activated environment or the checkout's
`venv`; otherwise it creates `venv` with a working `python` or `py -3`. Python 3.12
is the tested RTX 4060 configuration. For legacy CUDA 12.4, Python 3.10–3.13 wheels
are available; Python 3.14 cannot use that bundle. See [Python compatibility](python-compatibility.md).

### Linux and WSL

Install the NVIDIA driver appropriate to the host and a Python environment with
venv support. In WSL, use the Windows NVIDIA driver and verify `nvidia-smi` inside
WSL; do not install a Linux display driver over the WSL driver interface.
The driver installation is outside RuinedFooocus's launcher.

```sh
git clone --branch development https://github.com/johnmart19/RuinedFooocus-ROCm.git
cd RuinedFooocus-ROCm
PYTHON=python3.12 ./RuinedFooocus.sh
```

Check the [NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
for host requirements. A fresh Ubuntu WSL installation was also checked on the
RTX 4060, as recorded below.

Existing Windows weights can be reused with individual symlinks in WSL:

```sh
mkdir -p models/checkpoints
ln -s /mnt/c/RuinedFooocus/models/checkpoints/example.safetensors models/checkpoints/example.safetensors
```

Use the actual file and Windows mount path. Link other components into their
matching folders (`clip`, `vae`, `llm`, and so on), keeping Linux settings, cache,
outputs and Python environment separate. Symlinks save disk space, but still read
through the Windows mount. For models you load frequently, copy the weights into
the WSL checkout's model folders instead. No second download is required.
Microsoft also recommends keeping files used by Linux tools in the
[WSL filesystem](https://learn.microsoft.com/en-us/windows/wsl/interop).

On this RTX 4060, Qwen 2.5 7B Q4_K_M loaded through `/mnt/c` in 34.0 seconds,
versus 1.8 seconds from an identical WSL-local copy using the same native CUDA
binary and 8192-token context. Both offloaded all 29 layers. In the actual Chat
bots UI, replacing only its symlink with the local copy reduced loading from
32.3 to 2.8 seconds. Chat decoding was already about 52 tokens/second before the
copy: the delay was model loading, not CPU inference or missing CUDA packages.
These are observed, cache-dependent timings, not guaranteed performance figures.
Forcing non-mapped reads of the mounted file still took 28.8 seconds, so no global
loading-mode override was added.

Add `--api` for OpenAI-compatible endpoints. Once dependencies are installed,
run `venv\Scripts\python.exe launch.py --offline --nobrowser --port 7863 --api`
on Windows, or `venv/bin/python launch.py --offline --nobrowser --port 7863 --api`
on Linux. This skips Git/startup dependency updates, but missing model components
can still download. Follow [Open WebUI setup](open-webui.md) to configure clients.

## CUDA selection

Selection uses the GPU compute capability, driver CUDA ceiling and wheel tags
for the active Python, OS and CPU architecture. `nvidia-smi` headers `CUDA Version`
and `CUDA UMD Version` are recognized. These are driver capabilities, not the
PyTorch CUDA version. An actual NVIDIA adapter must be present; a stale CUDA
header on an AMD machine is insufficient.

`TORCH_PLATFORM` can request `cu124`, `cu126`, `cu128`, `cu130`, `cu132`, or the
opt-in nightly `cu134`. A compatible installed bundle remains installed unless
reinstall is requested. Application dependency updates constrain its versions,
including when `freezetorch` was created by another installer.

| CUDA |PyTorch | TorchVision | TorchAudio |
| --- | --- | --- | --- |
| cu124 | 2.6.0 | 0.21.0 | 2.6.0 cu124 |
| cu126 | 2.14.0 | 0.29.0 | 2.11.0 CPU |
| cu128 | 2.11.0 | 0.26.0 | 2.11.0 cu128 |
| cu130 | 2.14.0 | 0.29.0 | 2.11.0 CPU |
| cu132 | 2.14.0 | 0.29.0 | 2.11.0 CPU |
| cu134 (opt-in) | nightly | nightly | 2.11.0 CPU |

The CPU TorchAudio wheel used with cu126/cu130/cu132 uses PyTorch's [stable ABI](https://github.com/pytorch/audio/blob/main/docs/source/installation.rst).

### Architecture selection and rf-setup

[rf-setup](https://github.com/yownas/rf-setup) creates an isolated Python runtime
and can preinstall/freeze PyTorch. RuinedFooocus respects that environment; its
setup script does not need to install AMD dependencies. `rocm-bootstrap` is now
installed only when AMD detection is needed for an AMD installation path.

GTX 1080 Ti/Pascal selects CUDA 12.6, falling back to 12.4 when required by the
driver or available Python wheels. PyTorch 2.14 is the
[last release with CUDA 12.6](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432).
The CUDA 12.4 bundle supports Python 3.10–3.13. Compatible installed bundles
remain unchanged, including existing CUDA 12.4 installations.

Turing (including GTX 1660) and newer prefer stable CUDA 13.2, then compatible
older bundles. Blackwell requires CUDA 12.8 or newer; GB10/DGX Spark uses 13.x
and requires matching ARM64 wheels. Nightly is never chosen automatically.
If no bundle matches, startup reports the Python/GPU/driver combination without
silently installing CPU torch. Wheel availability does not prove that every
application dependency supports the same Python version.

Kepler and older GPUs (compute capability below 5.0) need a separate legacy
environment. Jetson, MIG and ARM64 application setups are not validated. Detection
uses compute capability, so consumer, laptop and workstation names do not require
separate allowlists. See the [PyTorch architecture matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#pytorch-cuda-support-matrix).

### Multiple GPUs

The installer selects a CUDA bundle compatible with every selected adapter.
Pascal plus Ada can use CUDA 12.6; Pascal plus Blackwell needs a narrower device
selection. Explicit CUDA reinstall choices are validated before replacing wheels.

Prefer `CUDA_VISIBLE_DEVICES=GPU-<UUID>` using UUIDs from `nvidia-smi -L`;
multiple UUIDs may be comma-separated. For numeric selection on multi-GPU hosts,
including `--gpu-device-id`, set `CUDA_DEVICE_ORDER=PCI_BUS_ID` first. Indices
then follow PCI bus order; ambiguous default ordering is rejected. These controls
apply to CUDA, not Vulkan. See [NVIDIA's environment variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html).

When moving between NVIDIA and AMD machines, use each machine's own environment
and driver. Do not copy the virtual environment or runtime DLLs. Remove stale
`TORCH_PLATFORM`, `CUDA_VISIBLE_DEVICES` and `LLAMA_SERVER` overrides, and select
an appropriate chat backend on the destination machine.

### Nightly opt-in

Use `RuinedFooocus.bat --cuda-nightly` or `./RuinedFooocus.sh --cuda-nightly`
to install nightly PyTorch/TorchVision for a supported NVIDIA GPU. The launcher
checks CUDA 13.4, 13.2, then 13.0 nightly indexes against the driver and current
Python wheel tags, and resolves the package pair before installation. TorchAudio
uses the compatible stable CPU build. Installed CUDA and GPU execution are checked
afterward. A compatible bundle must exist; the flag does not install another Python.

Current nightlies require Turing or newer. Pascal/GTX 1080 Ti remains on stable
CUDA 12.6/12.4. `TORCH_PLATFORM=cu132` can restrict the nightly selection to that
CUDA version. CPU, DirectML, ROCm, offline and frozen automatic installs cannot be
combined with this flag. Omit it on later starts to retain the installed bundle
without requesting another nightly installation. Nightly builds are experimental.

## Chat backends

Native llama.cpp Auto uses Vulkan on NVIDIA. Windows also offers official
CUDA 12.4 and 13.3 builds; their matching `cudart` archives are downloaded and
verified together. GPU capability and the driver are checked before selecting
these builds. Pascal uses CUDA 12.4 or Vulkan, not CUDA 13. The pinned upstream
release has no Ubuntu CUDA archive, so Linux uses Vulkan unless a custom
`LLAMA_SERVER` is supplied.

xllamacpp uses its CUDA 12.8/13.2 indexes for matching Torch backends and Vulkan
for the other supported CUDA versions. If the pinned CUDA wheel is unavailable
for the current Python/OS, installation tries Vulkan. Installed CUDA device
availability is checked independently of the package version, so a CPU/Vulkan
wheel with the same version cannot silently satisfy a CUDA installation request.
RTX 4060 execution is covered by the validation above; this does not
establish execution support on every NVIDIA generation.

## Reinstallation and isolation

For an explicit selection, use `set TORCH_PLATFORM=cu124` in Windows cmd, or
`TORCH_PLATFORM=cu124 ./RuinedFooocus.sh` on Linux. Remove `freezetorch` to change
the installed bundle automatically, or request a Torch reinstall in Settings. Existing
CUDA bundles are checked in a child process, including a small GPU operation;
a fresh CUDA installation must pass the same check before startup continues.

Automatic NVIDIA fallback detection requires an actual adapter from
`nvidia-smi`; a leftover CUDA driver header alone cannot select NVIDIA on an AMD
machine. An adapter whose capability cannot be queried gets a diagnostic instead
of a silent CPU install. CUDA installation and native CUDA chat validate all
selected NVIDIA GPUs; incompatible mixed architectures require device selection.
Installed CUDA and ROCm bundles use separate probes, and offline UI
startup reports the compatible installed bundle. Explicit CPU/DirectML choices
remain authoritative.

Reinstall all first resolves dependencies with the existing GPU bundle
constrained, then force-reinstalls the exact resolved application versions without
dependencies. It must not upgrade Torch or select a newer incompatible OpenCV,
rembg or Transformers build behind the resolver. DirectML uses a separate
environment with Torch 2.4.1 and Transformers 5.13; modern CUDA uses Transformers
5.17. See [shared reinstall behavior](gpu-setup.md#installation-and-reinstalls).

## Verified NVIDIA configuration

On September 30, 2026: Windows, RTX 4060 8 GB (compute capability 8.9), driver
617.14, Python 3.12.10. The working Torch 2.14.0+cu132 / TorchVision 0.29.0+cu132 /
TorchAudio 2.11.0+cpu bundle was preserved through the dependency update.

| Component | Checked revision/version |
| --- | --- |
| Gradio / client | 6.29.0 / 2.7.1 |
| ComfyUI | `fb2315f11db0ebfaafa9099a5df5227dc6bb42bc` |
| ComfyUI-GGUF | `48de657b3aa830ae6981960e928b31cb51fd16aa` |
| Native llama.cpp | b10917, Vulkan and CUDA 13.3 |
| xllamacpp | 2026.9.11063, CUDA |

Completed checks:

- **Images:** JANKU v7.77 through API and rendered UI, with Illustrious XL selected
  explicitly: 832×1216, 30 steps, CFG 5, Euler ancestral, normal schedule, CLIP Skip
  2. A woman in a teal sweater holding a white coffee cup by a café window matched
  the smoke-test prompt. This does not verify automatic profile selection for every
  checkpoint or identity preservation.
- **Chat:** Qwen 2.5 7B Q4_K_M answered `6 × 7` with `42` without an image call,
  and returned valid `generate_image` arguments for an explicit red-teapot image
  request. Native Vulkan, native CUDA and xllamacpp CUDA passed. UI chat also worked.
- **Video:** Wan 2.1 T2V 1.3B Q8_0, UMT5 FP8 encoder and Wan VAE generated a
  512×288, nine-frame clip with 20 steps. First and last frames showed the requested
  red teapot. This is a short functional check, not long-video quality testing,
  Wan 2.2 dual-expert validation, or an LTX full-workflow result.
- **Upscaling/media:** RealESRGAN x4 anime enlarged 130×192 to 520×768. MP4/PyAV
  and GIF round-trips passed for an eight-frame sample.
- **UI/API/privacy:** All four tabs rendered at 390×844 and 1920×1080 without
  horizontal overflow; generation previews appeared. API model/docs endpoints,
  owner file restrictions, authenticated guest media isolation and private cache
  headers passed focused checks. Public sharing controls remained disabled.
- **Dependencies:** `pip check` passed in each installed environment below.

| Environment | Actual validation |
| --- | --- |
| Windows Python 3.10.21, Torch 2.6.0 cu124 | Imports and full UI/API startup, CPU execution only |
| Windows Python 3.12.10, Torch 2.14.0 cu132 | RTX 4060 image/video/chat/upscaling checks above |
| Windows Python 3.12.10, Torch 2.4.1 DirectML | Imports, full UI/API startup and GPU arithmetic; no full image generation |
| Windows Python 3.14.7, Torch 2.14.0 CPU | Imports and full UI/API startup |
| Linux x86_64 Python 3.10/cu124 and 3.14/CPU | Dependency resolution only |
| Windows Python 3.11/cu124 | Dependency resolution only |

Vendor-isolation and mixed-GPU selection checks covered 42 and 47 cases,
respectively. These simulated selection paths do not replace physical tests.
Pascal/GTX 1080 Ti, other NVIDIA cards, CUDA nightlies, ARM64/Jetson/MIG and AMD
were not hardware-tested in this dependency refresh. Earlier AMD observations
are historical and must be repeated for a fresh AMD release claim.

## Fresh NVIDIA WSL validation

On September 30, 2026, a clean clone and `./RuinedFooocus.sh --nobrowser --port
7865 --api` were tested on Ubuntu 26.04.1 under WSL2, Python 3.14.4 and the RTX
4060 8 GB. The launcher created its venv and selected CUDA 13.2 automatically:
Torch 2.14.0+cu132, TorchVision 0.29.0+cu132 and TorchAudio 2.11.0+cpu.
`pip check` passed. Model weights were symlinked individually from Windows;
settings, caches and outputs belonged to the WSL checkout.

- JANKU v7.77 completed the 832×1216 Illustrious XL image check described above;
  the inspected image matched the café/teal-sweater prompt.
- xllamacpp 2026.9.11063 reported CUDA, answered normal chat without an image-tool
  call, streamed a complete reply, and returned valid image-tool arguments for an
  explicit image request.
- The Settings build action compiled native llama.cpp b10917 with CUDA 13.2 for
  SM 8.9, exposed **CUDA (local build)** only after success, and preserved the saved
  selection after restarting. A repeat build reused the compilation. Qwen 2.5 7B
  then returned valid image-tool arguments and a normal streamed `42` reply;
  Settings reported **llama.cpp [CUDA]**. The published binary resolved its private
  CUDA libraries without `LLAMA_SERVER` or a shell library-path override.
- Wan 2.1 T2V 1.3B Q8_0 completed nine frames at 512×288 and 20 steps. The first
  and last frames showed the requested red teapot. This remains a short functional
  check, not full video-family validation.
- RealESRGAN 4× upscaling and eight-frame GIF/MP4 round-trips passed. Main,
  Settings and Image Browser rendered; the browser indexed the generated images.
- IPv4 and IPv6 listener checks rejected live servers while allowing immediate
  restart after closed connections entered TIME_WAIT.

Linux xllamacpp probes now import Torch before inspecting devices, matching the
application's load order and resolving CUDA libraries bundled with Torch. Without
that import, the otherwise working CUDA wheel was falsely reported unavailable.
The installed native Vulkan binary reported no GPU on this WSL machine; CUDA
availability does not imply Vulkan passthrough. Use xllamacpp CUDA or the optional
[local native CUDA build](chat-runtime.md#build-a-local-cuda-runtime).

These results do not validate every Python version, GPU, video model or AMD setup.
The system compiler prerequisites were installed separately with Ubuntu's package
manager; the Settings builder installs its CUDA toolkit and CMake privately.

### Memory when switching to chat

The worker must release its local reference to the preloaded image pipeline after
startup; `shared.state` owns that pipeline until a different task replaces it.
Keeping the local reference in the permanent worker thread retained an unused
checkpoint throughout chat sessions. Releasing it reduced the measured WSL app
process RSS from about 7.9 GiB to 3.7 GiB after loading Qwen. The native chat server
was separate at roughly 0.7 GiB RSS. These values vary with models and allocator
caches; they are not memory limits.

Qwen 2.5 7B Q4_K_M at an 8192-token context used about 4.7 GiB of CUDA allocations
for weights, context and compute buffers. The desktop, driver and app add to the
GPU total. Unload released the chat server and lowered the observed total from
6.6 GiB to 1.5 GiB before the worker fix. A large Task Manager shared-GPU maximum
is a capacity, not allocated memory.

Avoid leaving separate Windows and WSL instances loaded unless both are needed.
Compare process RSS/PSS and Linux `MemAvailable` with Windows Task Manager;
reclaimable filesystem cache and Windows compressed memory are separate from live
model allocations. WSL provides [automatic memory reclamation](https://learn.microsoft.com/en-us/windows/wsl/wsl-config#experimental-settings).
Do not clear system caches or change global WSL limits on every generation.

### Chat-to-JANKU memory check

On September 30, the WSL UI completed five consecutive Qwen 2.5 7B Q4_K_M
messages: a cafe illustration, a beach revision, a text-only story, a teapot
illustration, and a color/flowers revision. All four images used JANKU v7.77,
the Illustrious XL profile, 832x1216, 30 steps, CFG 5, Euler ancestral, normal
schedule and CLIP skip 2, confirmed in their saved PNG metadata. Force image
generation and vision were off; image tools were enabled and history was 7.

Each image call closed the native chat server before diffusion. Qwen reloaded
in roughly 2.1 seconds between image requests and decoded at about 50 tokens/s.
Total observed GPU usage peaked at 7757 MiB of 8188 MiB, including desktop and
driver allocations. No OOM or CPU fallback was observed. After the final image,
the UI Unload action reduced total GPU usage to 899 MiB. Image execution took
23-28 seconds; complete chat/tool turns took 46-54 seconds, including loading
JANKU from the Windows-mounted model file.

Host memory is a separate limit: Python RSS peaked at 17.3 GiB during checkpoint
loading and remained about 11.1 GiB after Unload; Linux still reported 19 GiB
available and the process had no swap usage. This short run verifies working
GPU handoffs, not complete host-memory reclamation or a long-session leak test.
It does not validate GPT-OSS, xllamacpp image-tool switching, or other GPUs.

The revised prompts retained context and the story stayed text-only. Visual
inspection found imperfect prompt adherence: brown hair was longer than requested
and the red teapot was mostly white; the blue revision had blue decoration and
yellow flowers. These outputs establish successful execution, not exact character
identity or color preservation. Local evidence is in `chat-janku-memory.csv`,
`wsl-memory-fixed.log`, and `janku-chat-results/` under `tmp/local-tests/`.

## Handoff to the AMD workstation

1. Read `AGENTS.md` and this guide on the `documents` branch; run the application
   from `development`. Both branches have identical implementation files at this
   publication. The final three development commits isolate launchers, runtime
   detection and GPU installation. Do not rewrite history for routine fixes.
2. Use the AMD machine's own environment and driver. Do not transfer the NVIDIA
   `venv` or native runtime DLLs. Check any inherited environment overrides and
   `freezetorch` before startup; preserve user model weights, settings and artwork.
3. On WSL, verify the GPU with `rocminfo` and PyTorch HIP execution. `/dev/dxg`
   devices may not appear in the PCI list. ROCm 10 additionally needs the compatible
   ROCDXG bridge; do not infer its availability from NVIDIA testing.
4. Confirm the detected vendor and installed Torch backend before requesting a
   reinstall. Recheck normal startup, module reinstall and Torch reinstall
   separately in a disposable environment. They must retain AMD wheels and never
   infer NVIDIA from a leftover CUDA header.
5. Run one bounded JANKU image, Qwen normal chat plus image-tool call, and a short
   Wan clip if memory allows. Inspect output and loaded chat backend. Exercise
   cancellation/recovery separately if changing worker or memory-management code.

Useful local diagnostics (run with the application's Python):

```sh
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip, torch.cuda.is_available())"
python -c "import torch; print(torch.ones(1, device='cuda').cpu())"
```

The final command deliberately requires an available CUDA/HIP device. For NVIDIA,
also inspect `nvidia-smi -L`; for AMD WSL, inspect `rocminfo`. A successful import
or full VRAM allocation alone does not prove generation support.

## Code and evidence map

| Responsibility | Source |
| --- | --- |
| Windows/Linux environment entry | `RuinedFooocus.bat`, `RuinedFooocus.sh`, `entry_with_update.py` |
| Vendor detection and existing bundle probes | `modules/runtime_support.py` |
| NVIDIA architecture/driver/wheel selection | `modules/cuda_selection.py` |
| Torch constraints, GPU install and xllamacpp index | `modules/gpu_installer.py` |
| AMD-specific installation | `modules/rocm_installer.py` |
| Startup integration and exact reinstalls | `launch.py`, `modules/launch_util.py` |
| Native server download and process lifetime | `modules/llama_installer.py`, `modules/llama_server.py`, `modules/child_process.py` |

Raw checks and generated samples are local-only under `tmp/local-tests/` and
`outputs/`; they are not shipped in Git. On the NVIDIA workstation, useful evidence
includes `nvidia-chat-results.json`, `nvidia-api-results.json`,
`updated-video-result.json`, `updated-media-smoke.log`, `updated-privacy.log`,
`startup-310.log`, `startup-314.log` and `startup-directml.log`. Those files are
not guaranteed to exist on another machine; this guide records the bounded results.
Do not publish user settings, credentials, cached conversations or generated media
as part of a diagnostics handoff.
