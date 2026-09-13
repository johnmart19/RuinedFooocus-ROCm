# GPU setup

Use a supported x86_64 Python environment and install your GPU driver first.

- **Windows AMD:** keep the driver's compatible AI Bundle, or let the launcher
  install official [ROCm wheels](https://github.com/ROCm/TheRock/blob/main/RELEASES.md).
- **Ubuntu/WSL AMD:** install the [AMD runtime](https://rocm.docs.amd.com/en/latest/install/rocm.html)
  and `libnuma1`; `rocminfo` must list the GPU. WSL ROCm 10 needs ROCDXG 1.2.2 or newer.
- **NVIDIA:** selection uses GPU compute capability, the driver's reported CUDA
  ceiling, and wheel tags for the running Python, OS and CPU architecture.
  `TORCH_PLATFORM` can explicitly select `cu124`, `cu126`, `cu128`, `cu130`,
  `cu132`, or opt into nightly `cu134`.
  A compatible installed CUDA bundle is retained unless reinstall is requested.
  Its versions are constrained during application/LLM dependency installation,
  including when `freezetorch` was created by an external installer.

| CUDA | PyTorch | TorchVision | TorchAudio |
| --- | --- | --- | --- |
| cu124 | 2.6.0 | 0.21.0 | 2.6.0 cu124 |
| cu126 | 2.14.0 | 0.29.0 | 2.11.0 CPU |
| cu128 | 2.11.0 | 0.26.0 | 2.11.0 cu128 |
| cu130 | 2.14.0 | 0.29.0 | 2.11.0 CPU |
| cu132 | 2.14.0 | 0.29.0 | 2.11.0 CPU |
| cu134 (opt-in) | nightly | nightly | 2.11.0 CPU |

The cu132 audio wheel uses PyTorch's [stable ABI](https://github.com/pytorch/audio/blob/main/docs/source/installation.rst).
AMD Windows and ROCm 10 use Vulkan for xllamacpp; image generation uses ROCm.
Vulkan needs a working driver and may fall back to CPU under WSL.

Run `python entry_with_update.py`, or `python launch.py` to skip Git updates.
For WSL, create and activate `venv` first; `RuinedFooocus.sh` activates it
on later runs. Both launchers forward extra arguments.
`freezetorch` skips automatic torch installation. Explicit `reinstalltorch` or
`reinstall` requests override it for that restart; the freeze file stays intact.
The launcher installs Python packages, not drivers.

Settings reinstall buttons apply on the next online restart, in the Python
environment running RuinedFooocus. Torch reinstall uses the selected GPU backend
and its supported package bundle. Reinstall all also repairs bootstrap packages,
declared application packages, xllamacpp and API packages when `--api` is enabled.
Dependency resolution preserves the installed GPU package versions; application
packages are then force-reinstalled without replacing their Torch dependencies.
Offline starts and failed reinstalls leave the request pending.

### NVIDIA nightlies

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

### CPU and DirectML

`--cpu` takes precedence over automatic GPU detection and `TORCH_PLATFORM`.
It reuses a working installed PyTorch bundle for CPU execution; fresh installs
and explicit reinstalls use the CPU wheel index. Both chat runtimes disable
GPU layers, KV-cache and operation offloading. Native server extra arguments
cannot override CPU mode.

`--directml` (or `--directml 0` for a particular adapter) selects DirectML for
ComfyUI and Vulkan for chat. DirectML is not a llama.cpp backend. A saved native
CPU chat selection remains CPU; other native chat backend selections use Vulkan
during a DirectML session. GPU drivers must expose Vulkan for accelerated chat.

Microsoft's published DirectML package requires x86_64 Python 3.10–3.12 on Windows
or WSL and PyTorch 2.4.1. The installer pins matching TorchVision/TorchAudio and
compatible NumPy/SciPy, and preserves them during application dependency checks.
Use a separate environment from ROCm/CUDA: switching to DirectML replaces Torch.
Native Linux and newer Python versions are rejected before package installation.

Validation: Windows DirectML GPU arithmetic, ComfyUI imports, complete UI startup,
dependency resolution, and short CPU/Vulkan replies from both chat runtimes.
Full image/video generation and every model architecture have not been validated
on DirectML. Its operator/dtype coverage is limited by the
[Microsoft backend](https://github.com/microsoft/DirectML/wiki/PyTorch-DirectML-Operator-Roadmap);
these checks do not establish parity with CUDA/ROCm for all workflows.

### NVIDIA chat packages

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
NVIDIA package selection and release assets were verified; execution on physical
NVIDIA hardware remains untested.

### NVIDIA and rf-setup

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

AMD RDNA 3, RDNA 3.5 and RDNA 4 use detected `gfx110*`, `gfx115*` and `gfx120*`
device extras, respectively. The installer resolves the official packages for
the current Python before changing them. Windows RDNA 4 builds are not yet
qualified in [TheRock's support table](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md);
the launcher reports this and verifies GPU execution after installation.

For an explicit selection, use `set TORCH_PLATFORM=cu124` in Windows cmd, or
`TORCH_PLATFORM=cu124 ./RuinedFooocus.sh` on Linux. Remove `freezetorch` to change
the installed bundle automatically, or request a Torch reinstall in Settings. Existing
CUDA bundles are checked in a child process, including a small GPU operation;
a fresh CUDA installation must pass the same check before startup continues.

Installer validation includes simulated GTX 1080 Ti/RTX 4090 detection on both
OS paths and official wheel-index checks. This does not substitute for testing
generation on NVIDIA hardware.

On 2026-09-13, rf-setup's Windows Python 3.10.20 embedded download returned 404;
its Python 3.13.13 download was available. Use the working 3.13.13 option there
(cu124 wheels include cp313), or create your own Python 3.10/3.12 environment.
The external setup script's Python download failure happens before this launcher.

To install or repair AMD ROCm 10, run `python launch.py --rocm10` on Windows
or `./RuinedFooocus.sh --rocm10` on Linux/WSL. This reinstalls the
official AMD packages for the detected GPU. Omit the flag for normal startup;
NVIDIA CUDA selection is unchanged. The flag requires online setup and cannot
be combined with `freezetorch`, CPU or DirectML mode.


Linux/WSL AMD native chat uses the official ROCm 10 build with matching libraries in a separate cache environment. Other GPU platforms use Vulkan. PyTorch is unchanged by chat runtime installation.
