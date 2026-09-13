"""Managed local llama-server with the completion API used by Chat bots."""

import json
import os
from pathlib import Path
import secrets
import re
import shlex
import socket
import subprocess
import time
import weakref

import requests
from modules.llama_installer import server_path, runtime_environment, rocm_target
from modules.child_process import start_process
from modules.llama_vision import projector_path


class Server:
    def __init__(self, model, settings, gpu=True):
        if not Path(model).is_file():
            raise FileNotFoundError(f"LLM model not found: {model}")
        self.process = None
        self.session = requests.Session()
        self.session.trust_env = False
        token = secrets.token_urlsafe(32)
        self.session.headers["Authorization"] = "Bearer " + token
        with socket.socket() as socket_:
            socket_.bind(("127.0.0.1", 0))
            port = socket_.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        backend = settings.get("llama_backend", "Auto")
        if not gpu or backend == "CPU":
            backend, gpu = "CPU", False
        executable = server_path(gpu, backend=backend)
        environment = runtime_environment(gpu, backend=backend)
        expected_device = backend.split()[0] if backend not in ("Auto", "CPU") else None
        if backend == "Auto" and rocm_target(gpu):
            expected_device = "ROCm"
        if not os.environ.get("LLAMA_SERVER") and expected_device:
            devices = subprocess.run([str(executable), "--list-devices"], env=environment,
                capture_output=True, text=True, timeout=30,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if devices.returncode or expected_device not in devices.stdout:
                raise RuntimeError(f"llama.cpp could not detect a {expected_device} GPU. Check its runtime libraries and driver, "
                                   "or select CPU mode explicitly.")
        extra = shlex.split(settings.get("llama_server_args", ""))
        projector = projector_path(settings)
        if projector:
            extra += ["--mmproj", str(projector)]
        if not gpu:
            extra += ["--no-mmproj-offload", "--device", "none", "--n-gpu-layers", "0", "--no-kv-offload", "--no-op-offload"]
        layers = int(settings.get("llm_n_gpu_layers", -1)) if gpu else 0
        gpu_layers = "auto" if layers < 0 else str(layers)
        command = [str(executable), "--model", str(Path(model).resolve()),
                   "--ctx-size", str(int(settings.get("llm_n_ctx", 8192))),
                   "--n-predict", str(int(settings.get("llm_n_predict") or -1)),
                   "--n-gpu-layers", gpu_layers,
                   "--parallel", "1", "--jinja", "--reasoning-format", "deepseek", *extra,
                   "--host", "127.0.0.1", "--port", str(port), "--api-key", token, "--no-webui"]
        # Probe the selected binary, not PyTorch: its backend can be different.
        if not expected_device or os.environ.get("LLAMA_SERVER"):
            devices = subprocess.run([str(executable), "--list-devices"], env=environment,
                capture_output=True, text=True, timeout=30,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        device_names = re.findall(r"^\s*(CUDA|ROCm|Vulkan|Metal|SYCL)\d*:", devices.stdout, re.MULTILINE)
        if gpu and (devices.returncode or not device_names):
            self.session.close()
            raise RuntimeError("llama.cpp has no usable GPU device. Check the selected backend and driver, or select CPU explicitly.")
        self.backend = ("/".join(dict.fromkeys(device_names)) or "CPU") if devices.returncode == 0 else "unknown"
        # Extra arguments follow the defaults, so account for a GPU-layer override.
        layer_flags = re.findall(r"(?:--n-gpu-layers|--gpu-layers|-ngl)(?:=|\s+)(\S+)", " ".join(extra))
        if gpu and layer_flags:
            layers = layer_flags[-1]
        self.runtime_label = self.backend
        if str(layers) == "0" and self.backend != "CPU":
            self.runtime_label += "; CPU weights"
        from modules.chat_storage import chat_folder, migrate_legacy_chat_files
        migrate_legacy_chat_files()
        log_path = chat_folder("logs") / f"server-{port}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("w", encoding="utf-8") as log:
                self.process, job = start_process(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            self._cleanup = weakref.finalize(self, self._stop, self.process, self.session, job)
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"llama-server exited with code {self.process.returncode}; see {log_path}.")
                try:
                    if self.session.get(self.url + "/health", timeout=2).ok:
                        return
                except requests.ConnectionError:
                    pass
                time.sleep(0.2)
            raise TimeoutError(f"llama-server did not finish loading; see {log_path}.")
        except Exception:
            self.close()
            raise

    @staticmethod
    def _stop(process, session, job=None):
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        finally:
            if job:
                job.close()
            session.close()

    def close(self):
        if hasattr(self, "_cleanup"):
            self._cleanup()
        else:
            self.session.close()

    def _request(self, endpoint, data, callback=None):
        with self.session.post(self.url + endpoint, json=data, stream=bool(callback), timeout=(10, 600)) as response:
            if not response.ok:
                template_kwargs = data.get("chat_template_kwargs") or {}
                if (endpoint == "/v1/chat/completions" and response.status_code == 400
                        and data.get("tools") and "tools_in_user_message" not in template_kwargs
                        and "Unable to generate parser" in response.text
                        and "Cannot put tools in the first user message" in response.text):
                    # Some Llama templates fail during the parser's empty-history probe.
                    # Keep tools enabled, but render their definitions in the system message.
                    retry = {**data, "chat_template_kwargs": {
                        **template_kwargs, "tools_in_user_message": False,
                    }}
                    response.close()
                    return self._request(endpoint, retry, callback)
                raise RuntimeError(f"llama-server: {response.status_code}: {response.text[:1000]}")
            if callback is None:
                return response.json()
            for line in response.iter_lines():
                if line.startswith(b"data: "):
                    payload = line[6:]
                    if payload == b"[DONE]":
                        break
                    chunk = json.loads(payload)
                    if "error" in chunk:
                        raise RuntimeError(str(chunk["error"]))
                    callback(chunk)

    def handle_completions(self, data):
        return self._request("/v1/completions", data)

    def handle_chat_completions(self, data, callback=None):
        return self._request("/v1/chat/completions", data, callback)
