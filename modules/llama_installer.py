"""Official llama.cpp binaries, cached separately from models and Python packages."""

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import re
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import zipfile
import venv

import requests

VERSION = "b10917"
RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/"


def validate_cuda_backend(backend):
    if not backend.startswith("CUDA "):
        return
    from modules.cuda_selection import nvidia_info
    capability, driver = nvidia_info()
    required = tuple(int(part) for part in backend.split()[1].split("."))
    if capability is None or driver is None:
        raise RuntimeError("Cannot validate the NVIDIA GPU/driver. Check nvidia-smi or select Vulkan/CPU.")
    if driver < required:
        raise RuntimeError(f"{backend} requires a newer NVIDIA driver; select Vulkan or update the driver.")
    if (required[0] >= 13 and capability < 7.5) or (required == (12, 4) and capability >= 10):
        raise RuntimeError(f"{backend} does not support this GPU architecture. Select a compatible CUDA backend or Vulkan.")


def backend_choices(system=None, machine=None):
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    choices = ["Auto", "CPU"]
    if system == "Windows" and machine in ("amd64", "x86_64"):
        choices += ["Vulkan", "ROCm", "CUDA 12.4", "CUDA 13.3"]
    elif system == "Linux":
        choices += ["Vulkan"]
        if machine in ("amd64", "x86_64"):
            choices += ["ROCm"]
    return choices


def package_name(system, machine, gpu, rocm=False, backend="Auto"):
    if backend not in backend_choices(system, machine):
        raise RuntimeError(f"No official {backend} build for {system}/{machine}; use LLAMA_SERVER for a custom build.")
    if backend == "CPU":
        gpu = False
    elif backend != "Auto":
        gpu = True
    arch = {"amd64": "x64", "x86_64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(machine.lower())
    if not arch:
        raise RuntimeError("Set LLAMA_SERVER to a llama-server binary for this CPU.")
    if system == "Windows":
        backend = ({"ROCm": "rocm-10.0", "CUDA 12.4": "cuda-12.4",
                    "CUDA 13.3": "cuda-13.3", "Vulkan": "vulkan"}.get(backend)
                   or ("vulkan" if gpu and arch == "x64" else "cpu"))
        return f"llama-{VERSION}-bin-win-{backend}-{arch}.zip"
    if system == "Linux":
        use_rocm = gpu and (backend == "ROCm" or backend == "Auto" and rocm)
        backend = "rocm-10.0-" if use_rocm else "vulkan-" if gpu else ""
        return f"llama-{VERSION}-bin-ubuntu-{backend}{arch}.tar.gz"
    if system == "Darwin":
        return f"llama-{VERSION}-bin-macos-{arch}.tar.gz"
    raise RuntimeError("Set LLAMA_SERVER to a llama-server binary for this OS.")


def rocm_target(gpu, windows=False):
    if not gpu or platform.system() != "Linux" and not windows:
        return None
    import torch
    if not torch.version.hip or not torch.cuda.is_available():
        return None
    target = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if not re.fullmatch(r"gfx[0-9a-f]+", target):
        raise RuntimeError(f"Cannot select llama.cpp ROCm libraries for GPU {target}")
    return target


def runtime_environment(gpu, reinstall=False, backend="Auto"):
    env = os.environ.copy()
    if backend == "ROCm" and platform.system() == "Windows" and gpu and not os.environ.get("LLAMA_SERVER"):
        return windows_rocm_environment(env, reinstall)
    target = rocm_target(gpu) if backend in ("Auto", "ROCm") else None
    if not target or os.environ.get("LLAMA_SERVER"):
        return env
    # Native subprocess dependencies must not replace the app's working torch/ROCm.
    directory = Path("cache/llama.cpp/rocm-10.0").resolve()
    python = directory / "bin/python"
    marker = directory / f"{target}.ready"
    if reinstall or not marker.exists():
        marker.unlink(missing_ok=True)
        if not python.exists():
            venv.EnvBuilder(with_pip=True).create(directory)
        command = [str(python), "-m", "pip", "install",
            f"rocm[libraries,device-{target}]==10.0.0", "--index-url",
            "https://stable.repo.amd.com/rocm/whl-next/"]
        if reinstall:
            command.append("--force-reinstall")
        subprocess.run(command, check=True)
        marker.touch()
    site = next(directory.glob("lib/python*/site-packages"))
    libraries = [site / "_rocm_sdk_core/lib", site / "_rocm_sdk_libraries/lib"]
    libraries.append(Path("/usr/lib/wsl/lib"))
    env["LD_LIBRARY_PATH"] = ":".join(str(path) for path in libraries if path.is_dir())
    if os.environ.get("LD_LIBRARY_PATH"):
        env["LD_LIBRARY_PATH"] += ":" + os.environ["LD_LIBRARY_PATH"]
    return env


def windows_rocm_environment(env, reinstall=False, install=True):
    # Reuse a complete ROCm 10 SDK, otherwise install into a separate environment.
    script = ("import json, rocm_sdk as r; assert r.__version__ == '10.0.0'; "
              "print(json.dumps(list(dict.fromkeys(str(p.parent) for p in "
              "r.find_libraries('amdhip64', 'hipblas', 'hipblaslt')))))")

    def paths(python):
        result = subprocess.run([str(python), "-c", script], capture_output=True,
                                text=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            return None
        return json.loads(result.stdout)

    libraries = None if reinstall else paths(sys.executable)
    if not libraries and not install:
        python = Path("cache/llama.cpp/rocm-10.0-windows/Scripts/python.exe").resolve()
        libraries = paths(python) if python.exists() else None
        if not libraries:
            return env
    if not libraries:
        target = rocm_target(True, windows=True)
        if not target:
            raise RuntimeError("Cannot detect the AMD GPU target for ROCm 10. Select Vulkan or install a compatible ROCm SDK.")
        directory = Path("cache/llama.cpp/rocm-10.0-windows").resolve()
        python = directory / "Scripts/python.exe"
        libraries = paths(python) if python.exists() and not reinstall else None
        if not libraries:
            if not python.exists():
                venv.EnvBuilder(with_pip=True).create(directory)
            command = [str(python), "-m", "pip", "install",
                       f"rocm[libraries,device-{target}]==10.0.0", "--index-url",
                       "https://stable.repo.amd.com/rocm/whl-next/"]
            if reinstall:
                command.append("--force-reinstall")
            subprocess.run(command, check=True)
            libraries = paths(python)
        if not libraries:
            raise RuntimeError("ROCm 10 runtime libraries are unavailable. Select Vulkan or check the SDK installation.")
    env["PATH"] = os.pathsep.join(libraries + [env.get("PATH", "")])
    return env


def server_path(gpu=True, backend="Auto"):
    override = os.environ.get("LLAMA_SERVER")
    if override:
        executable = shutil.which(override) or override
        if not Path(executable).is_file():
            raise FileNotFoundError(f"LLAMA_SERVER does not exist: {override}")
        return Path(executable).resolve()
    if gpu:
        validate_cuda_backend(backend)
    package = package_name(platform.system(), platform.machine(), gpu,
                           bool(rocm_target(gpu)) if backend == "Auto" else False, backend)
    directory = Path("cache/llama.cpp") / VERSION / package.removesuffix(".zip").removesuffix(".tar.gz")
    executable = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    found = list(directory.rglob(executable))
    if found:
        return found[0].resolve()
    response = requests.get(RELEASES + VERSION, timeout=(15, 30))
    response.raise_for_status()
    assets = response.json()["assets"]
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory.parent) as temporary:
        temporary = Path(temporary)
        extracted = temporary / "runtime"
        download_package(package, assets, temporary, extracted)
        if "-win-cuda-" in package:
            dependency = "cudart-llama-bin-win-" + package.split("-bin-win-", 1)[1]
            download_package(dependency, assets, temporary, extracted)
        if not list(extracted.rglob(executable)):
            raise RuntimeError("The downloaded package has no llama-server executable.")
        extracted.rename(directory)
    return next(directory.rglob(executable)).resolve()


def download_package(package, assets, temporary, extracted):
    asset = next((item for item in assets if item["name"] == package), None)
    if not asset or not asset.get("digest", "").startswith("sha256:"):
        raise RuntimeError(f"No verified official llama.cpp package available: {package}")
    print(f"Downloading llama.cpp {VERSION} ({package})")
    archive = temporary / package
    digest = hashlib.sha256()
    with requests.get(asset["browser_download_url"], stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        with archive.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
    if digest.hexdigest() != asset["digest"].split(":", 1)[1]:
        raise RuntimeError("llama.cpp download checksum mismatch; retry the download.")
    if package.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extracted)
    else:
        if not hasattr(tarfile, "data_filter"):
            raise RuntimeError("Update Python to a version supporting safe tar extraction, or set LLAMA_SERVER.")
        with tarfile.open(archive) as bundle:
            bundle.extractall(extracted, filter="data")
