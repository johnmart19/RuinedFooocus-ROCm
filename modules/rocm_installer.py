"""Automatic AMD wheel selection, separate from OS driver installation."""

import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

# Release-ready Radeon targets in TheRock/SUPPORTED_GPUS.md.
# Windows RDNA 4 is additionally available as an unqualified build (reported below).
_RADEON = frozenset({
    "gfx1010", "gfx1011", "gfx1012",
    "gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036",
    "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152",
})
SUPPORTED_TARGETS = {
    "Windows": _RADEON | {"gfx1200", "gfx1201"},
    "Linux": _RADEON | {"gfx942", "gfx1200", "gfx1201"},
}
ROCM_PLATFORM = "rocm10.0"


@dataclass(frozen=True)
class RocmInstallPlan:
    platform: str
    args: list


ROCM_INDEX = "https://stable.repo.amd.com/rocm/whl-next/"


def parse_rocminfo_targets(output):
    # Match agent names, not ISA lists (which also contain generic gfx targets).
    return sorted(set(re.findall(r"^\s*Name:\s+(gfx[0-9a-f]+)\s*$", output, re.M)))


def detect_rocm_targets(os_platform):
    # AMD-only dependency: never install it from the shared bootstrap requirements.
    try:
        import rocm_bootstrap
    except ModuleNotFoundError:
        subprocess.run([sys.executable, "-m", "pip", "install", "rocm-bootstrap==0.2.0"], check=True)
    from rocm_bootstrap import detect_gfx_targets

    targets = sorted({target.name for target in detect_gfx_targets()})
    if targets or os_platform != "Linux":
        return targets
    return detect_rocminfo_targets()


def detect_rocminfo_targets():
    # WSL exposes /dev/dxg, not the KFD topology used by rocm-bootstrap.
    rocminfo = shutil.which("rocminfo")
    if not rocminfo or os.environ.get("ROCM_BOOTSTRAP_DISABLE_DETECTION") == "1":
        return []
    try:
        result = subprocess.run(
            [rocminfo], capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return parse_rocminfo_targets(result.stdout) if result.returncode == 0 else []


def automatic_rocm_plan(os_platform, selected_platform, force=False):
    """Return a checked installation plan, or None for another backend/override."""
    override = os.environ.get("TORCH_PLATFORM")
    if os_platform not in SUPPORTED_TARGETS:
        return None
    target_platform = ROCM_PLATFORM
    if not force and override and override != target_platform:
        return None
    # Keep the existing choice on mixed AMD/NVIDIA or AMD/Intel machines.
    if not force and not override and selected_platform not in ("cpu", "directml") and not selected_platform.startswith("rocm"):
        return None
    targets = []
    if not force and not override and selected_platform in ("cpu", "directml"):
        from torchruntime.device_db import get_gpus
        if not any(gpu.vendor_id.lower() == "1002" for gpu in get_gpus()):
            # WSL can expose AMD through DXG without a PCI device entry.
            if os_platform == "Linux" and "microsoft" in platform.release().lower():
                targets = detect_rocminfo_targets()
            if not targets:
                return None
    targets = targets or detect_rocm_targets(os_platform)
    if not targets:
        if selected_platform.startswith("rocm"):
            raise RuntimeError(
                "ROCm requested, but no AMD GFX target could be determined. "
                "Check the AMD driver and clinfo (Windows) or rocminfo (Linux/WSL)."
            )
        return None
    unsupported = set(targets) - SUPPORTED_TARGETS[os_platform]
    if unsupported:
        raise RuntimeError(
            f"Automatic ROCm installation is not validated for {os_platform}: "
            f"{', '.join(sorted(unsupported))}. Install a compatible AMD bundle manually "
            "or explicitly select another TORCH_PLATFORM."
        )
    if platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError("Automatic ROCm wheels require x86_64 Python.")
    if os_platform == "Windows" and set(targets) & {"gfx1200", "gfx1201"}:
        print("Windows RDNA 4 wheels are available but not yet qualified by TheRock; GPU validation is required after installation.")
    extras = ",".join(f"device-{target}" for target in targets)
    print(f"AMD GPU targets: {', '.join(targets)}")
    return RocmInstallPlan(target_platform, [
        f"torch[{extras}]==2.13.0+rocm10.0.0",
        f"torchvision[{extras}]==0.28.0+rocm10.0.0",
        "torchaudio==2.11.0.2+rocm10.0.0",
        "--index-url", ROCM_INDEX,
        "--only-binary=torch,torchvision,torchaudio",
    ])


def install_rocm(args, reinstall=False):
    command = [sys.executable, "-m", "pip", "install", *args]
    if reinstall:
        command.append("--force-reinstall")
    # Resolve Python/OS wheel tags and device extras before changing the environment.
    subprocess.run([*command, "--dry-run"], check=True)
    subprocess.run(command, check=True)
    # Do not continue with a CPU fallback if an incompatible driver/runtime loaded.
    subprocess.run([
        sys.executable, "-c",
        "import torch; "
        "assert torch.version.hip and torch.cuda.is_available(), "
        "'ROCm installed but no GPU is available; check the AMD driver/WSL runtime'; "
        "weights=torch.ones(8192,768,dtype=torch.float16).to('cuda'); "
        "assert weights[0,0].item()==1; "
        "x=torch.ones(16,16,device='cuda'); y=x@x; torch.cuda.synchronize(); "
        "assert y[0,0].item()==16; print('ROCm verified:',torch.cuda.get_device_name(0))",
    ], check=True)
