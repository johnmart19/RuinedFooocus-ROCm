"""CUDA selection using GPU architecture, driver and this interpreter's wheel tags."""

import re
import os
import shutil
from pathlib import Path
import subprocess
import sys
from functools import lru_cache
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen
from urllib.error import URLError

from packaging.tags import sys_tags
from packaging.utils import InvalidWheelFilename, parse_wheel_filename


CUDA_PACKAGES = {
    "cu124": ("2.6.0", "0.21.0", "2.6.0"),
    "cu126": ("2.14.0", "0.29.0", None),
    "cu128": ("2.11.0", "0.26.0", "2.11.0"),
    "cu130": ("2.14.0", "0.29.0", None),
    "cu132": ("2.14.0", "0.29.0", None),
}


def cuda_index(runtime, nightly=False):
    channel = f"nightly/{runtime}" if nightly or runtime == "cu134" else runtime
    return f"https://download.pytorch.org/whl/{channel}"


@lru_cache(maxsize=32)
def wheel_available(index, package, version=None):
    tags = set(sys_tags())
    try:
        with urlopen(f"{index.rstrip('/')}/{package}/", timeout=30) as response:
            page = response.read().decode("utf-8")
    except (URLError, OSError) as error:
        raise RuntimeError(f"Could not check {package} wheels at {index}: {error}. Retry when the index is reachable.") from error
    for link in re.findall(r'href=[\"\']([^\"\']+)', page):
        filename = unquote(urlsplit(link).path).rsplit("/", 1)[-1]
        try:
            _, release, _, wheel_tags = parse_wheel_filename(filename)
        except InvalidWheelFilename:
            continue
        if (version is None or release.base_version == version) and tags.intersection(wheel_tags):
            return True
    return False


def cuda_wheels_available(runtime, nightly=False):
    versions = (None, None, None) if nightly else CUDA_PACKAGES.get(runtime, (None, None, None))
    index = cuda_index(runtime, nightly)
    if not all(wheel_available(index, name, version)
               for name, version in zip(("torch", "torchvision"), versions[:2])):
        return False
    audio_index = index if versions[2] else cuda_index("cpu")
    return wheel_available(audio_index, "torchaudio", versions[2] or "2.11.0")


def select_cuda_nightly(os_platform, override=None):
    if os_platform not in ("Windows", "Linux"):
        raise RuntimeError("--cuda-nightly requires Windows or Linux.")
    capability, driver = nvidia_info()
    if capability is None or driver is None:
        raise RuntimeError("--cuda-nightly requires an NVIDIA GPU and working nvidia-smi detection.")
    if capability < 7.5:
        raise RuntimeError("Current CUDA nightlies require Turing or newer. Keep CUDA 12.6/12.4 for GTX 1080 Ti and other Pascal GPUs.")
    candidates = ["cu134", "cu132", "cu130"]
    if override:
        if override not in candidates:
            raise RuntimeError("TORCH_PLATFORM conflicts with --cuda-nightly; remove it or select cu130, cu132 or cu134.")
        candidates = [override]
    for candidate in candidates:
        if (int(candidate[2:-1]), int(candidate[-1])) <= driver and cuda_wheels_available(candidate, nightly=True):
            print(f"CUDA nightly: {candidate}; Python {sys.version.split()[0]}; GPU capability {capability}")
            return candidate
    raise RuntimeError(f"No supported CUDA nightly bundle matches Python {sys.version.split()[0]}, this OS/CPU architecture and driver CUDA {driver}. Existing packages were not replaced.")


def nvidia_info():
    executable = shutil.which("nvidia-smi")
    if executable is None:
        # Driver utilities are not always on PATH, especially in WSL.
        candidates = []
        if sys.platform == "win32":
            candidates = [
                Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/nvidia-smi.exe",
                Path(os.environ.get("ProgramW6432", os.environ.get("ProgramFiles", "C:/Program Files"))) / "NVIDIA Corporation/NVSMI/nvidia-smi.exe",
            ]
        elif sys.platform == "linux":
            candidates = [Path("/usr/lib/wsl/lib/nvidia-smi")]
        executable = next((str(path) for path in candidates if path.is_file()), "nvidia-smi")
    try:
        output = subprocess.check_output([executable], text=True, timeout=15,
                                         stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None, None
    cuda = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", output)
    driver = (int(cuda[1]), int(cuda[2])) if cuda else None
    try:
        capabilities = subprocess.check_output(
            [executable, "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, timeout=15, stderr=subprocess.DEVNULL)
        caps = [float(value.strip()) for value in capabilities.splitlines()]
        return min(caps), driver
    except (OSError, subprocess.SubprocessError, ValueError):
        return None, driver


def cuda_candidates(capability, driver_cuda=None):
    candidates = ["cu126", "cu124"] if capability < 7.5 else ["cu132", "cu130", "cu128"]
    if 7.5 <= capability < 10:
        candidates += ["cu126", "cu124"]
    if capability >= 12.1:
        candidates = ["cu132", "cu130"]  # Includes GB10/Spark; Python tags decide ARM wheel availability.
    if driver_cuda:
        candidates = [c for c in candidates if (int(c[2:-1]), int(c[-1])) <= driver_cuda]
    return candidates


def select_cuda(gpus):
    capability, driver = nvidia_info()
    if capability is None:
        from torchruntime.gpu_db import get_nvidia_arch
        capability = get_nvidia_arch({gpu.device_name for gpu in gpus if gpu.vendor_id.lower() == "10de"})
    if capability is None or capability < 5:
        raise RuntimeError("This NVIDIA architecture needs an older PyTorch environment.")
    print(f"NVIDIA compute capability: {capability}; driver CUDA: {driver}; Python: {sys.version.split()[0]}")
    for candidate in cuda_candidates(capability, driver):
        if cuda_wheels_available(candidate):
            return candidate
    raise RuntimeError(f"No compatible CUDA wheel bundle for Python {sys.version.split()[0]}, "
                       f"GPU compute capability {capability}, driver CUDA {driver}. "
                       "Use a supported Python environment or update the NVIDIA driver.")
