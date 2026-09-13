"""Select a runtime without importing PyTorch into the installer process."""

import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys

from modules.shared_functions import broken_torch_platforms


def _read_installed_torch(backend="rocm"):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None
    try:
        import torch
    except Exception:
        if backend in version.lower() or (backend == "cuda" and "+cu" in version.lower()):
            raise
        return None
    runtime = "cpu" if backend == "cpu" else torch.version.hip if backend == "rocm" else torch.version.cuda
    if not runtime:
        return None
    if backend == "cpu":
        if (torch.ones((2, 2), device="cpu") @ torch.ones((2, 2), device="cpu"))[0, 0].item() != 2:
            raise RuntimeError("CPU kernel check failed.")
    if backend == "cuda":
        if not torch.cuda.is_available():
            return None
        # Validate kernels too: an import can succeed with unsupported GPU wheels.
        x = torch.ones((16, 16), device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        if y[0, 0].item() != 16:
            raise RuntimeError("CUDA kernel check failed.")

    import torchvision  # Validate native extensions before preserving the bundle.
    versions = {name: importlib.metadata.version(name) for name in ("torch", "torchvision")}
    try:
        versions["torchaudio"] = importlib.metadata.version("torchaudio")
        if backend == "cuda":
            import torchaudio  # A matching version alone does not validate native DLLs.
    except importlib.metadata.PackageNotFoundError:
        pass
    for name in versions:
        for raw in importlib.metadata.requires(name) or []:
            requirement = Requirement(raw)
            dependency = canonicalize_name(requirement.name)
            if dependency not in versions or (requirement.marker and not requirement.marker.evaluate()):
                continue
            if not requirement.specifier.contains(versions[dependency], prereleases=True):
                raise RuntimeError(f"{name} requires {requirement}, found {versions[dependency]}")
    return {"hip" if backend == "rocm" else backend: runtime, "versions": versions}


def inspect_installed_cpu():
    try:
        result = subprocess.run([sys.executable, "-c",
            "import json; from modules.runtime_support import _read_installed_torch; "
            "print(json.dumps(_read_installed_torch('cpu')))"],
            capture_output=True, text=True, timeout=90)
        return json.loads(result.stdout.strip().splitlines()[-1]) if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def _read_installed_rocm():
    return _read_installed_torch("rocm")


def inspect_installed_cuda(os_platform):
    override = os.environ.get("TORCH_PLATFORM", "")
    if os_platform not in ("Windows", "Linux") or (override and not override.startswith("cu")):
        return None
    try:
        if "rocm" in importlib.metadata.version("torch").lower():
            return None
    except importlib.metadata.PackageNotFoundError:
        return None
    try:
        result = subprocess.run([sys.executable, "-c",
            "import json; from modules.runtime_support import _read_installed_torch; "
            "print(json.dumps(_read_installed_torch('cuda')))"],
            capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        print("Installed CUDA validation timed out; checking installation requirements.")
        return None
    if result.returncode:
        # Allow the installer to repair a broken CUDA environment.
        print("Installed CUDA packages need repair; checking installation requirements.")
        return None
    return json.loads(result.stdout.strip().splitlines()[-1])


def installed_cuda_platform(bundle):
    major, minor, *_ = bundle["cuda"].split(".")
    return f"cu{major}{minor}"


def inspect_installed_rocm(os_platform, repair=False):
    """Probe in a child process so pip never replaces a loaded torch DLL."""
    override = os.environ.get("TORCH_PLATFORM", "")
    if os_platform not in ("Windows", "Linux") or (override and not override.startswith("rocm")):
        return None
    try:
        if "+cu" in importlib.metadata.version("torch").lower():
            return None
    except importlib.metadata.PackageNotFoundError:
        return None
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import json; from modules.runtime_support import _read_installed_rocm; "
             "print(json.dumps(_read_installed_rocm()))"],
            capture_output=True, text=True, timeout=90,
        )
    except subprocess.TimeoutExpired as error:
        if repair:
            print("Installed ROCm validation timed out; checking AMD installation requirements.")
            return None
        raise RuntimeError("Installed ROCm validation timed out; request a Torch reinstall to repair it.") from error
    if result.returncode:
        if repair:
            print("Installed ROCm packages need repair; checking AMD installation requirements.")
            return None
        raise RuntimeError(
            "Cannot validate the installed ROCm bundle. Repair torch/torchvision; "
            "RuinedFooocus has not replaced them.\n" + result.stderr
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def installed_rocm_platform(bundle):
    # ROCm release and HIP component versions differ (ROCm 10 ships HIP 7.15).
    version = bundle["versions"]["torch"]
    match = re.search(r"\+rocm(\d+)\.(\d+)", version)
    if not match:
        match = re.match(r"(\d+)\.(\d+)", bundle["hip"])
    if not match:
        raise RuntimeError(f"Unrecognized ROCm build: {version}")
    return f"rocm{match[1]}.{match[2]}"


def select_torch_platform(torchruntime, os_platform, installed_rocm=None, installed_cuda=None):
    override = os.environ.get("TORCH_PLATFORM")
    if override:
        selected = override
    elif installed_rocm:
        selected = installed_rocm_platform(installed_rocm)
    elif installed_cuda:
        selected = installed_cuda_platform(installed_cuda)
    else:
        gpus = torchruntime.device_db.get_gpus()
        from modules.cuda_selection import select_cuda, nvidia_info
        nvidia = any(gpu.vendor_id.lower() == "10de" for gpu in gpus)
        if not nvidia and os_platform in ("Windows", "Linux"):
            # WSL and virtual machines may not expose NVIDIA in the PCI list.
            capability, driver = nvidia_info()
            nvidia = capability is not None or driver is not None
        selected = select_cuda(gpus) if nvidia else torchruntime.platform_detection.get_torch_platform(gpus)
    return broken_torch_platforms(selected, os_platform)[0]
