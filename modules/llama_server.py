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
        self.model_path = str(model)
        self.launch_settings = dict(settings)
        self.use_gpu = gpu
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
        self.executable, self.environment = executable, environment
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
        from modules.chat_context import prepare
        return self._request("/v1/chat/completions", prepare(self, data), callback)

    def context_size(self):
        response = self.session.get(self.url + "/props", timeout=10)
        response.raise_for_status()
        return int(response.json()["default_generation_settings"]["n_ctx"])

    def count_chat_tokens(self, data):
        # This endpoint uses the runtime's multimodal tokenizer, including
        # projector-dependent image tokens. Never estimate those from text.
        response = self.session.post(self.url + "/v1/chat/completions/input_tokens",
                                     json=data, timeout=(10, 120))
        if response.ok:
            count = response.json().get("input_tokens")
            if isinstance(count, int) and count > 0:
                return count
            raise ValueError("Runtime returned an invalid input token count.")
        if response.status_code not in (404, 405, 501):
            raise ValueError("Runtime could not count this chat request: " + response.text[:300])
        if any(isinstance(m.get("content"), list) and any(
                p.get("type") != "text" for p in m["content"]) for m in data.get("messages", [])):
            raise ValueError("This runtime lacks reliable vision token counting. Update its runtime before using automatic vision context management.")
        rendered = self._request("/apply-template", data)["prompt"]
        return len(self._request("/tokenize", {"content": rendered, "parse_special": True})["tokens"])

    def try_grow_context(self, needed):
        """Grow only with known model limits and conservative device headroom."""
        current = self.context_size()
        target = min(current * 2, 32768)
        # Respect explicit server overrides and avoid repeated reloads per request.
        if target <= current or self.launch_settings.get("llama_server_args", "").strip():
            return False
        try:
            import gguf
            import psutil
            reader = gguf.GGUFReader(self.model_path)
            def field(name):
                value = reader.get_field(name)
                if value is None:
                    raise ValueError("Unknown model context layout")
                return value.contents()
            architecture = field("general.architecture")
            prefix = architecture + "."
            trained = int(field(prefix + "context_length"))
            target = min(target, trained)
            heads = int(field(prefix + "attention.head_count"))
            kv_heads = int(field(prefix + "attention.head_count_kv"))
            layers = int(field(prefix + "block_count"))
            width = int(field(prefix + "embedding_length")) // heads
            for name in ("key_length", "value_length"):
                value = reader.get_field(prefix + "attention." + name)
                if value is not None:
                    width = max(width, int(value.contents()))
            # FP32 K and V, doubled for allocation/scratch overhead. Unknown
            # layouts do not grow automatically; compaction remains available.
            cost = (target - current) * layers * kv_heads * width * 16
            del reader
            ram_free = psutil.virtual_memory().available
            if target <= current or ram_free < cost + 2 * 1024**3:
                return False
            if self.use_gpu:
                if self.free_gpu_memory() < cost + 2 * 1024**3:
                    return False
        except (ImportError, KeyError, ValueError, TypeError, ZeroDivisionError, OSError, subprocess.SubprocessError):
            return False
        try:
            self.reload_context(target)
        except Exception:
            self.reload_context(current)
            return False
        print(f"Chat context: increased from {current} to {target} tokens on {self.runtime_label}.")
        return True

    def free_gpu_memory(self):
        devices = subprocess.run([str(self.executable), "--list-devices"],
            env=self.environment, capture_output=True, text=True, timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        free = re.findall(r"([0-9]+) MiB free", devices.stdout)
        return int(free[0]) * 1024**2 if not devices.returncode and len(free) == 1 else 0

    def reload_context(self, context):
        settings = {**self.launch_settings, "llm_n_ctx": context}
        model, gpu = self.model_path, self.use_gpu
        self.close()
        self.__init__(model, settings, gpu=gpu)
