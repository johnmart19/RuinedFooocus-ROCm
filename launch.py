import os
import sys
import subprocess
import version
import warnings
from pathlib import Path
import ssl
from tempfile import gettempdir
from modules.rocm_installer import automatic_rocm_plan, install_rocm
from modules.runtime_support import inspect_installed_rocm, inspect_installed_cuda, inspect_installed_cpu, installed_cuda_platform, select_torch_platform
from modules.gpu_installer import (
    preserve_installed_rocm, preserve_installed_cuda, installed_torch_constraints, torch_constraints, torch_install_commands,
    xllamacpp_index, has_vulkan_xllamacpp, has_cuda_xllamacpp,
)
from modules.comfy_compat import configure_torch_allocator

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CMAKE_POLICY_VERSION_MINIMUM"] = "3.5"

ssl._create_default_https_context = ssl._create_unverified_context

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="kornia")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="gradio")
warnings.filterwarnings("ignore", category=UserWarning, module="torchsde")
warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.transforms.functional_tensor"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

from modules.launch_util import (
    is_installed,
    run,
    python,
    run_pip,
    pip_rm,
    repo_dir,
    requirements_met,
    script_path,
    dir_repos,
)


git_repos = [
    {
        "name": "ComfyUI",
        "path": "ComfyUI",
        "url": "https://github.com/comfy-org/ComfyUI",
        "hash": "7193f5627f036701e5efc23beaea20fa37ceaadd",
        "add_path": "ComfyUI",
    },
#    {
#        "name": "Calcuis-GGUF",
#        "path": "calcuis_gguf",
#        "url": "https://github.com/calcuis/gguf",
#        "hash": "34a4e030afea0137c5e781e07400bdbe00e9d524",
#        "add_path": "",
#    },
#    {
#        "name": "ComfyUI-GGUF",
#        "path": "comfyui_gguf",
#        "url": "https://github.com/city96/ComfyUI-GGUF",
#        "hash": "6ea2651e7df66d7585f6ffee804b20e92fb38b8a",
#        "add_path": "",
#    },
    {
        "name": "molbal/ComfyUI-GGUF",
        "path": "molbal_comfyui_gguf",
        "url": "https://github.com/molbal/ComfyUI-GGUF",
        "hash": "c6e14d9c2475bb39a100f85d582e224c6e884f80",
        "add_path": "",
    },
]


def prepare_environment(offline=False):
    print(f"Python {sys.version}")
    print(f"RuinedFooocus version: {version.version}")

    requirements_file = "requirements_versions.txt"

    modules_file = "pip/modules.txt"

    if offline:
        print("Skip pip check.")
    else:
        run(
            f'"{python}" -m pip install --upgrade pip',
            "Check pip",
            "Couldn't check pip",
            live=False,
        )
        if not getattr(sys, "_rf_bootstrap_reinstalled", False):
            run(
                f'"{python}" -m pip install {"--force-reinstall " if REINSTALL_ALL else ""}-r "{requirements_file}"',
                "Check pre-requirements",
                "Couldn't check pre-reqs",
                live=False,
            )
        run(
            f'"{python}" -m pip uninstall -y llama-cpp-python',
            "Check for old modules",
            "Couldn't check old modules",
            live=False,
        )


        import torchruntime
        import platform
        os_platform = platform.system()
        if args.cuda_nightly:
            from modules.cuda_selection import select_cuda_nightly
            os.environ["TORCH_PLATFORM"] = select_cuda_nightly(os_platform, os.environ.get("TORCH_PLATFORM"))
        installed_rocm = inspect_installed_rocm(os_platform, repair=REINSTALL_ALL or REINSTALL_TORCH or args.rocm10)
        installed_cuda = inspect_installed_cuda(os_platform)
        installed_cpu = inspect_installed_cpu() if args.cpu else None
        keep_cpu = bool(installed_cpu and not (REINSTALL_ALL or REINSTALL_TORCH))
        torch_platform = select_torch_platform(torchruntime, os_platform, installed_rocm, installed_cuda)
        keep_cuda = preserve_installed_cuda(installed_cuda, torch_platform, REINSTALL_ALL or REINSTALL_TORCH or args.rocm10 or args.cuda_nightly)
        keep_rocm = preserve_installed_rocm(
            installed_rocm, torch_platform, REINSTALL_ALL or REINSTALL_TORCH or args.rocm10,
        )

        rocm_plan = None
        if args.rocm10:
            rocm_plan = automatic_rocm_plan(os_platform, "rocm10.0", force=True)
            if rocm_plan is None:
                raise RuntimeError("--rocm10 requires a supported AMD GPU on Windows or Linux/WSL.")
            torch_platform = rocm_plan.platform
            os.environ["TORCH_PLATFORM"] = torch_platform
        elif not keep_rocm and not TORCH_FROZEN:
            rocm_plan = automatic_rocm_plan(os_platform, torch_platform)
            if rocm_plan:
                torch_platform = rocm_plan.platform

        # Forward automatic fallback modes to ComfyUI and chat as well as pip.
        if torch_platform == "cpu":
            args.cpu = True
        elif torch_platform == "directml" and args.directml is None:
            args.directml = -1
        os.environ["TORCH_PLATFORM"] = torch_platform
        configure_torch_allocator(torch_platform, os_platform)

        print(f"Torch platform: {os_platform}: {torch_platform}")

    if offline:
        print("Skip check of required modules.")
    else:
        os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"

        # Run torchruntime install
        if keep_cpu:
            print(f"Using installed PyTorch {installed_cpu['versions']['torch']} on CPU")
        elif keep_rocm:
            print(f"Using installed {os_platform} ROCm PyTorch {installed_rocm['versions']['torch']}")
        elif keep_cuda:
            print(f"Using installed {os_platform} CUDA PyTorch {installed_cuda['versions']['torch']}")
        elif rocm_plan:
            install_rocm(rocm_plan.args, REINSTALL_ALL or REINSTALL_TORCH or args.rocm10)
            installed_rocm = inspect_installed_rocm(os_platform)
            if not installed_rocm:
                raise RuntimeError("ROCm installation did not provide a valid HIP PyTorch build.")
            torch_platform = select_torch_platform(torchruntime, os_platform, installed_rocm)
            keep_rocm = True
        elif not TORCH_FROZEN:
            if os_platform == "Windows" and torch_platform.startswith("rocm"):
                raise RuntimeError(
                    "Install a compatible Windows ROCm bundle from AMD for this explicit version. "
                    "Remove reinstall/reinstalltorch after repairing the bundle."
                )
            cmds = torch_install_commands(torchruntime, torch_platform, os_platform, nightly=args.cuda_nightly)
            if REINSTALL_ALL or REINSTALL_TORCH:
                for idx in range(len(cmds)):
                    cmds[idx].insert(0, "--force-reinstall")
            cmds = torchruntime.installer.get_pip_commands(cmds)
            # Resolve every stage before replacing a working runtime.
            for command in cmds:
                subprocess.run([*command, "--dry-run"], check=True)
            for command in cmds:
                subprocess.run(command, check=True)
            if torch_platform == "directml":
                subprocess.run([python, "-c",
                    "import torch, torch_directml as dml; "
                    f"device=dml.device({args.directml}) if {args.directml} >= 0 else dml.device(); "
                    "x=torch.ones((2,2)).to(device); assert (x@x).cpu()[0,0].item()==2; "
                    "print('DirectML GPU verified')"], check=True)
            if torch_platform.startswith("cu"):
                installed_cuda = inspect_installed_cuda(os_platform)
                if not installed_cuda or installed_cuda_platform(installed_cuda) != torch_platform:
                    raise RuntimeError("CUDA installation did not validate the selected runtime. Check the NVIDIA driver and CUDA/Python versions; the reinstall request remains pending.")
                if args.cuda_nightly:
                    from packaging.version import Version
                    if not Version(installed_cuda["versions"]["torch"]).is_devrelease:
                        raise RuntimeError("CUDA nightly was requested, but installed PyTorch is not a nightly build.")
            torchruntime.configure()
        else:
            print(f"Torch frozen...")

        installed_torch = installed_torch_constraints()
        if REINSTALL_ALL or not requirements_met(modules_file):
            print("This next step may take a while")
            with torch_constraints(installed_torch) as constraints:
                run_pip(f'install -r "{modules_file}"{constraints}', "required modules")
                if REINSTALL_ALL:
                    # Dependencies were resolved above. Do not reinstall vendor torch from PyPI.
                    run_pip(f'install --force-reinstall --no-deps -r "{modules_file}"{constraints}', "reinstall required modules")

        try:
            xlc_version = "xllamacpp==2026.9.10809"
            index = xllamacpp_index(torch_platform, os_platform, version=xlc_version.split("==")[1])
            needs_vulkan = index.endswith("/vulkan")
            needs_cuda = index.rsplit("/", 1)[-1] in ("cu128", "cu132")
            wrong_backend = (needs_vulkan and not has_vulkan_xllamacpp()
                             or needs_cuda and not has_cuda_xllamacpp())
            if REINSTALL_ALL or not is_installed(xlc_version) or wrong_backend:
                with torch_constraints(installed_torch) as constraints:
                    run_pip(
                        f'install {xlc_version} --index-url {index} --only-binary=xllamacpp{constraints}',
                        "XLlamacpp",
                    )
                    if REINSTALL_ALL or wrong_backend:
                        run_pip(f'install --force-reinstall --no-deps {xlc_version} --index-url {index} --only-binary=xllamacpp', "reinstall XLlamacpp backend")
                if needs_cuda and not has_cuda_xllamacpp():
                    raise RuntimeError("xllamacpp CUDA installed but no CUDA device is available. Check the NVIDIA driver, or use native llama.cpp with Vulkan.")
        except Exception as e:
            if REINSTALL_ALL:
                raise  # Keep the request pending when a requested reinstall fails.
            print("WARNING: Failed to install/update llm modules.")
            print(e)

        if args.rocm10:
            subprocess.run([python, "-c",
                "from modules.llama_installer import server_path, runtime_environment; "
                "server_path(True); runtime_environment(True, reinstall=True)"], check=True)

    if args.api:
        print("Check API dependencies")
        api_requirements = "requirements_api_versions.txt"
        if REINSTALL_ALL or not requirements_met(api_requirements):
            if offline:
                if not requirements_met(api_requirements):
                    raise RuntimeError("API dependencies are missing. Launch once without --offline to install them.")
            else:
                with torch_constraints(installed_torch) as constraints:
                    run_pip(f'install -r "{api_requirements}"{constraints}', "API dependencies")
                    if REINSTALL_ALL:
                        run_pip(f'install --force-reinstall --no-deps -r "{api_requirements}"{constraints}', "reinstall API dependencies")

def clone_git_repos(offline=False):
    from modules.launch_util import git_clone

    for repo in git_repos:
        if not offline:
            git_clone(repo["url"], repo_dir(repo["path"]), repo["name"], hash=repo["hash"])
        add_path = str(Path(script_path) / dir_repos / repo["add_path"])
        if add_path not in sys.path:
            sys.path.append(add_path)


def download_models():
    from modules.util import load_file_from_url
    from shared import path_manager

    model_filenames = [
        (
            path_manager.model_paths["vae_approx_path"],
            "taesdxl_decoder",
            "https://github.com/madebyollin/taesd/raw/main/taesdxl_decoder.pth",
        ),
        (
            "prompt_expansion",
            "pytorch_model.bin",
            "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin",
        ),
    ]

    for model_dir, file_name, url in model_filenames:
        load_file_from_url(
            url=url,
            model_dir=model_dir,
            file_name=file_name,
        )


from argparser import args

REINSTALL_ALL = False
if os.path.exists("reinstall"):
    REINSTALL_ALL = True
REINSTALL_TORCH = False
if os.path.exists("reinstalltorch"):
    REINSTALL_TORCH = True
TORCH_FROZEN = os.path.exists("freezetorch") and not (REINSTALL_ALL or REINSTALL_TORCH)

if args.gpu_device_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device_id)
    print("Set device to:", args.gpu_device_id)

offline = os.environ.get("RF_OFFLINE") == "1" or "--offline" in sys.argv or "--iINSTallLEDmYOwNPaCKaGeS" in sys.argv
if args.cpu and args.directml is not None:
    raise RuntimeError("Choose --cpu or --directml, not both.")
if args.directml is not None and args.directml < -1:
    raise RuntimeError("--directml expects a non-negative GPU index, or no value for the default GPU.")
if args.cpu or args.directml is not None:
    os.environ["TORCH_PLATFORM"] = "cpu" if args.cpu else "directml"
if os.environ.get("TORCH_PLATFORM") == "cpu":
    args.cpu = True
elif os.environ.get("TORCH_PLATFORM") == "directml" and args.directml is None:
    args.directml = -1

if args.rocm10 and (offline or args.cpu or args.directml is not None or os.path.exists("freezetorch")):
    raise RuntimeError("--rocm10 cannot be combined with offline, CPU, DirectML or freezetorch mode.")
if args.cuda_nightly and (offline or args.rocm10 or args.cpu or args.directml is not None or TORCH_FROZEN):
    raise RuntimeError("--cuda-nightly cannot be combined with offline, ROCm, CPU, DirectML or freezetorch mode.")

if offline:
    print("Skip checking python modules.")
    if REINSTALL_ALL or REINSTALL_TORCH:
        print("Reinstall request pending: restart without --offline to apply it.")

prepare_environment(offline)

if not offline and os.path.exists("reinstall"):
    try:
        os.remove("reinstall")
    except:
        print("ERROR: Failed to remove 'reinstall'. Pleae remove manually.")
if not offline and os.path.exists("reinstalltorch"):
    try:
        os.remove("reinstalltorch")
    except:
        print("ERROR: Failed to remove 'reinstalltorch'. Pleae remove manually.")

try:
    clone_git_repos(offline)
except:
    print(f"WARNING: Failed checking git-repos. Trying to start without update.")

if not offline:
    download_models()

gradio_cache = os.path.join(gettempdir(), 'ruinedfooocus_cache')
os.environ['GRADIO_TEMP_DIR'] = gradio_cache
# Delete old data
import shutil
try:
    if args.clean_cache and gradio_cache.endswith('ruinedfooocus_cache'):
        shutil.rmtree(gradio_cache)
except FileNotFoundError:
    pass
except PermissionError:
    pass
except Exception as e:
    print(f"An error occurred: {str(e)}")

def launch_ui():
    print("Starting webui")
    import webui
launch_ui()
