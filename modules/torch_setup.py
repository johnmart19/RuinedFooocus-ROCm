"""Explicit PyTorch selection for help/rf-setup, using the launcher's validated bundles."""

import platform
import subprocess
import sys

CHOICES = ("cu124", "cu126", "cu128", "cu130", "cu132", "cu134", "rocm", "cpu")


def install(choice):
    if choice not in CHOICES:
        raise SystemExit(f"Choose one of: {', '.join(CHOICES)}")
    import torchruntime
    os_platform = platform.system()
    if choice == "rocm":
        from modules.rocm_installer import ROCM_PLATFORM, automatic_rocm_plan, install_rocm
        plan = automatic_rocm_plan(os_platform, ROCM_PLATFORM, force=True)
        if plan is None:
            raise RuntimeError("ROCm requires a supported AMD GPU on Windows or Linux/WSL.")
        install_rocm(plan.args, reinstall=True)
        return
    from modules.gpu_installer import torch_install_commands
    # Validates the GPU, driver and wheel tags before anything is replaced.
    commands = torch_install_commands(torchruntime, choice, os_platform, nightly=choice == "cu134")
    commands = torchruntime.installer.get_pip_commands([["--force-reinstall", *command] for command in commands])
    for command in commands:
        subprocess.run([*command, "--dry-run"], check=True)
    for command in commands:
        subprocess.run(command, check=True)
    if choice.startswith("cu"):
        from modules.runtime_support import inspect_installed_cuda, installed_cuda_platform
        installed = inspect_installed_cuda(os_platform)
        if not installed or installed_cuda_platform(installed) != choice:
            raise RuntimeError(f"PyTorch {choice} was installed but could not use the NVIDIA GPU. Check the driver.")
        print(f"Verified PyTorch {installed['versions']['torch']} on CUDA {installed['cuda']}")


if __name__ == "__main__":
    install(sys.argv[1] if len(sys.argv) == 2 else "")
