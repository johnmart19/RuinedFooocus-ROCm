# GPU setup

Use each machine's own driver and Python environment. The launchers install
Python packages, not GPU drivers. For NVIDIA installation, CUDA selection,
chat backends and device validation, read [NVIDIA support](nvidia-support.md).
This page covers the shared installation rules and AMD/CPU/DirectML paths.

## Installation and reinstalls

Install Git, a supported 64-bit Python and your GPU driver's runtime. Start
`RuinedFooocus.bat` on Windows or `./RuinedFooocus.sh` on Linux/WSL. The launchers
create or reuse the local environment; see [launchers](launchers.md) for interpreter
selection. Python 3.12 is a tested option; see [Python compatibility](python-compatibility.md)
for version-specific dependencies and validation limits.

`python launch.py` skips checkout updates. For an already prepared environment,
`python launch.py --offline --nobrowser --port 7863 --api` skips startup package
and backend updates. Missing auxiliary model files can still download, and API
dependencies must already be installed. See [API setup](api.md).

Working installed GPU bundles are preserved. `freezetorch` prevents automatic
Torch installation; explicit Settings reinstall requests override it for that
restart, keeping the freeze file. Settings reinstall buttons act in the environment
running RuinedFooocus, on its next online start. Reinstall all repairs bootstrap,
application, chat and enabled API packages; Torch reinstall selects the detected
or explicitly chosen backend. Vendor Torch versions remain constrained while
application dependencies resolve, and reinstalls use those exact resolved versions.
Offline starts and failures leave reinstall requests pending.

## AMD setup

On Windows, keep the driver's compatible AI Bundle, or let startup install
official [ROCm wheels](https://github.com/ROCm/TheRock/blob/main/RELEASES.md).
On Ubuntu/WSL, install the [AMD runtime](https://rocm.docs.amd.com/en/latest/install/rocm.html)
and `libnuma1`; `rocminfo` must list the GPU. WSL ROCm 10 requires ROCDXG 1.2.2
or newer. PCI enumeration alone can miss WSL's `/dev/dxg` device.

AMD RDNA 3, RDNA 3.5 and RDNA 4 use detected `gfx110*`, `gfx115*` and `gfx120*`
device extras, respectively. The installer resolves the official packages for
the current Python before changing them. Windows RDNA 4 builds are not yet
qualified in [TheRock's support table](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md);
the launcher reports this and verifies GPU execution after installation.

To install or repair AMD ROCm 10, run `python launch.py --rocm10` on Windows
or `./RuinedFooocus.sh --rocm10` on Linux/WSL. This reinstalls the
official AMD packages for the detected GPU. Omit the flag for normal startup;
NVIDIA CUDA selection is unchanged. The flag requires online setup and cannot
be combined with `freezetorch`, CPU or DirectML mode.
Linux/WSL AMD native chat uses the official ROCm 10 build with matching libraries
in a separate cache environment. Other GPU platforms use Vulkan by default.
PyTorch is unchanged by chat runtime installation.

AMD Windows and ROCm 10 use Vulkan for xllamacpp while image generation
uses ROCm. Linux ROCm 6.4/7.2 can use matching xllamacpp packages when available
for the running interpreter. Vulkan requires a functioning driver and can fall
back to CPU under WSL. Check the actual loaded chat backend.

## CPU and DirectML

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

## Moving between AMD and NVIDIA

Use a fresh virtual environment on the destination machine; do not copy `venv`,
cached runtime binaries or DLLs between vendors. Model weights and wanted outputs
can be transferred separately. Review copied settings and custom paths before use.
Remove stale `TORCH_PLATFORM`, `CUDA_VISIBLE_DEVICES` and `LLAMA_SERVER` overrides
and choose the destination chat backend. A copied `freezetorch` file can prevent
initial GPU installation, so keep it only when deliberately maintaining a prepared
environment.

NVIDIA detection requires an actual adapter, not a leftover CUDA driver header.
Installed CUDA and ROCm use separate probes; explicit CPU/DirectML flags take
precedence. Do not infer fresh AMD validation from the NVIDIA results below.

Read the [NVIDIA handoff](nvidia-support.md#handoff-to-the-amd-workstation) before
rechecking the implementation on AMD.
