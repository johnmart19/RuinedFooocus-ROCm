"""PyTorch and llama.cpp package selection."""

import importlib.metadata
import os
import re
import sys
import subprocess
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from modules.runtime_support import installed_rocm_platform
from modules.runtime_support import installed_cuda_platform


def preserve_installed_cuda(bundle, torch_platform, reinstall=False):
    from packaging.version import Version
    return bool(bundle and not reinstall
                and installed_cuda_platform(bundle) == torch_platform
                and Version(bundle["versions"]["torch"]) >= Version("2.6.0"))


def installed_torch_constraints():
    versions = {}
    packages = ["torch", "torchvision", "torchaudio"]
    if os.environ.get("TORCH_PLATFORM") == "directml":
        packages += ["numpy", "scipy", "torch-directml"]
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"versions": versions} if versions else None


def preserve_installed_rocm(installed_rocm, torch_platform, reinstall=False):
    if not installed_rocm or reinstall:
        return False
    # An explicit different ROCm version is a request to change environments.
    return torch_platform == installed_rocm_platform(installed_rocm)


@contextmanager
def torch_constraints(installed_rocm):
    """Keep vendor builds intact when resolving transitive dependencies."""
    if not installed_rocm:
        yield ""
        return
    with TemporaryDirectory(prefix="ruinedfooocus-torch-") as directory:
        path = Path(directory) / "constraints.txt"
        path.write_text(
            "".join(f"{name}=={version}\n" for name, version in installed_rocm["versions"].items()),
            encoding="utf-8",
        )
        yield f' --constraint "{path}"'


def torch_install_commands(torchruntime, torch_platform, os_platform, nightly=False):
    if torch_platform == "cpu":
        return [["torch==2.14.0", "torchvision==0.29.0", "torchaudio==2.11.0", "--index-url", "https://download.pytorch.org/whl/cpu"]]
    if torch_platform == "directml":
        import platform
        if os_platform not in ("Windows", "Linux") or not (3, 10) <= sys.version_info[:2] <= (3, 12):
            raise RuntimeError("DirectML requires Python 3.10–3.12 on Windows or WSL.")
        if platform.machine().lower() not in ("amd64", "x86_64"):
            raise RuntimeError("The published DirectML wheels require x86_64 Python.")
        if os_platform == "Linux" and "microsoft" not in platform.release().lower():
            raise RuntimeError("DirectML requires Windows or WSL, not native Linux. Use --cpu or your GPU's native backend.")
        return [["torch==2.4.1+cpu", "torchvision==0.19.1+cpu", "torchaudio==2.4.1+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"],
                ["--no-deps", "torch-directml==0.2.5.dev240914", "numpy==1.26.4", "scipy==1.15.3"]]
    if torch_platform == "cu124" and not (3, 10) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError("CUDA 12.4 requires Python 3.10–3.13 in RuinedFooocus. Use Python 3.10 or 3.12 for GTX 1080 Ti.")
    # Keep the selected CUDA runtime: newer wheels may drop older GPU support.
    from modules.cuda_selection import CUDA_PACKAGES, cuda_index, cuda_wheels_available
    if os_platform not in ("Windows", "Linux") or torch_platform not in (*CUDA_PACKAGES, "cu134"):
        return torchruntime.installer.get_install_commands(torch_platform, [])
    if not cuda_wheels_available(torch_platform, nightly=nightly):
        raise RuntimeError(f"No {torch_platform} wheel bundle matches Python {sys.version.split()[0]} and this OS/CPU architecture.")
    if nightly or torch_platform == "cu134":
        return [["--pre", "--upgrade", "--force-reinstall", "--only-binary=:all:", "torch", "torchvision", "--index-url", cuda_index(torch_platform, nightly)],
                ["--no-deps", "torchaudio==2.11.0+cpu", "--index-url", cuda_index("cpu")]]
    torch, vision, audio = CUDA_PACKAGES[torch_platform]
    packages = [f"torch=={torch}+{torch_platform}", f"torchvision=={vision}+{torch_platform}"]
    if audio:
        packages.append(f"torchaudio=={audio}+{torch_platform}")
    commands = [packages + ["--index-url", f"https://download.pytorch.org/whl/{torch_platform}"]]
    if not audio:
        # TorchAudio 2.11 uses the stable torch ABI. No cu132 wheel exists;
        # its CPU extension works with newer torch and avoids CUDA DLL conflicts.
        commands.append(["--no-deps", "torchaudio==2.11.0+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"])
    # Windows Triton is optional; an unbounded version breaks older torch builds.
    return commands


def xllamacpp_index(torch_platform, os_platform, version=None):
    if os_platform == "Windows" and torch_platform.startswith("rocm"):
        backend = "vulkan"
    else:
        backend = {
            "cu124": "vulkan", "cu126": "vulkan", "cu128": "cu128",
            "cu130": "vulkan", "cu132": "cu132", "cu134": "vulkan",
            "rocm6.4": "rocm-6.4.1", "rocm7.2": "rocm-7.2.4",
            "rocm10.0": "vulkan",  # ROCm 7.x LLM libraries conflict with ROCm 10.
            "vulkan": "vulkan", "directml": "vulkan",
        }.get(torch_platform)
    root = "https://xorbitsai.github.io/xllamacpp/whl"
    if version and backend:
        from modules.cuda_selection import wheel_available
        if not wheel_available(f"{root}/{backend}", "xllamacpp", version):
            if backend.startswith("cu"):
                print(f"No matching xllamacpp {backend} wheel for this Python; trying Vulkan.")
                backend = "vulkan"
            if not wheel_available(f"{root}/{backend}", "xllamacpp", version):
                raise RuntimeError(f"No xllamacpp {version} {backend} wheel matches this Python/OS. Use native llama.cpp or a supported Python version.")
    return f"{root}/{backend}" if backend else "https://pypi.org/simple"


def has_cuda_xllamacpp():
    # Check the loaded backend, not just the shared distribution version.
    try:
        result = subprocess.run([sys.executable, "-c",
            "import xllamacpp as x; "
            "print('RF_CUDA=' + str(int(any(d['name'].startswith('CUDA') for d in x.get_device_info()))))"],
            capture_output=True, text=True, timeout=30)
        return result.returncode == 0 and "RF_CUDA=1" in result.stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return False


def has_vulkan_xllamacpp():
    # Backend wheels share a distribution version; version alone is insufficient.
    try:
        files = importlib.metadata.files("xllamacpp") or []
        return any(
            (file.name.lower() == "ggml-vulkan.dll"
             or re.fullmatch(r"vulkan-1(?:-[0-9a-f]+)?\.dll", file.name.lower())
             or re.fullmatch(r"libvulkan(?:-[0-9a-f]+)?\.so(?:\.[0-9]+)*", file.name.lower()))
            and Path(file.locate()).is_file()
            for file in files
        )
    except importlib.metadata.PackageNotFoundError:
        return False
