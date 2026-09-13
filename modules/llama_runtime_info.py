"""Inspect installed chat runtimes without downloading or loading models."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def runtime_choices():
    from shared import state
    pipeline = state.get("pipeline")
    loaded = getattr(pipeline, "runtime_name", None) if getattr(pipeline, "llm", None) else None
    label = getattr(pipeline, "runtime_label", None) or "loading"
    return [(f"{name} [{label if name == loaded else 'not loaded'}]", name)
            for name in ("llama.cpp", "xllamacpp")]


def _run(command, env=None):
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=15, env=env,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode:
        raise RuntimeError("Runtime probe failed")
    return result.stdout + "\n" + result.stderr


def runtime_info():
    sections = []
    override = os.environ.get("LLAMA_SERVER")
    if override:
        binaries = [Path(shutil.which(override) or override)]
    else:
        root = Path("cache/llama.cpp")
        executable = "llama-server.exe" if os.name == "nt" else "llama-server"
        binaries = sorted(root.glob(f"b*/**/{executable}"))
    if not binaries:
        sections.append("**llama.cpp:** not installed; downloads on first use.")
    for binary in binaries:
        label = "Custom llama.cpp" if override else binary.parent.name
        try:
            env = os.environ.copy()
            if not override and "rocm" in str(binary):
                if os.name == "nt":
                    from modules.llama_installer import windows_rocm_environment
                    env = windows_rocm_environment(env, install=False)
                else:
                    sites = Path("cache/llama.cpp/rocm-10.0").glob("lib/python*/site-packages")
                    paths = [str(site / package / "lib") for site in sites
                             for package in ("_rocm_sdk_core", "_rocm_sdk_libraries")]
                    env["LD_LIBRARY_PATH"] = ":".join(paths + ["/usr/lib/wsl/lib", env.get("LD_LIBRARY_PATH", "")])
            version = _run([str(binary.resolve()), "--version"], env).strip()
            devices = _run([str(binary.resolve()), "--list-devices"], env).strip()
            sections.append(f"**{label}**\n```text\n{version}\nCPU; GPU devices reported by this build:\n{devices}\n```")
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            sections.append(f"**{label}:** installed, but could not query this build's devices.")
    try:
        script = (
            "import json, xllamacpp as x; from importlib.metadata import version; "
            "print('RF_RUNTIME_INFO=' + json.dumps({'version':version('xllamacpp'), "
            "'devices':[{'name':d['name'],'description':d['description']} for d in x.get_device_info()]}))"
        )
        output = _run([sys.executable, "-c", script])
        line = next(line for line in output.splitlines() if line.startswith("RF_RUNTIME_INFO="))
        info = json.loads(line.split("=", 1)[1])
        devices = "\n".join(f"{d['name']}: {d['description']}" for d in info["devices"])
        sections.append(f"**xllamacpp {info['version']}**\n```text\n{devices or 'No devices reported'}\n```")
    except (OSError, RuntimeError, subprocess.TimeoutExpired, StopIteration, ValueError, KeyError):
        from importlib.metadata import PackageNotFoundError, version
        try:
            sections.append(f"**xllamacpp {version('xllamacpp')}:** installed; device information unavailable.")
        except PackageNotFoundError:
            sections.append("**xllamacpp:** not installed.")
    return "\n\n".join(sections)
