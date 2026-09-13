"""Use the embedded runtime's HTTP tokenizer with shared context management."""

import gc
import secrets

import requests

from modules.llama_server import Server


class EmbeddedServer(Server):
    def __init__(self, params):
        self.params = params
        self.model_path = params.model.path
        self.launch_settings = {"llm_n_ctx": params.n_ctx}
        self.use_gpu = params.n_gpu_layers != 0
        self.raw = None
        self.session = None
        self._start()

    def _start(self):
        import xllamacpp as xlc
        self.params.hostname = "127.0.0.1"
        self.params.port = 0
        token = secrets.token_urlsafe(32)
        self.params.api_keys = [token]
        self.params.ctx_shift = False
        self.raw = xlc.Server(self.params)
        self.url = self.raw.listening_address
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers["Authorization"] = "Bearer " + token
        devices = [d for d in xlc.get_device_info() if d["name"] != "CPU"]
        self.runtime_label = "/".join(d["name"] for d in devices) if self.use_gpu else "CPU"

    def close(self):
        if self.session is not None:
            self.session.close()
        self.raw = None
        gc.collect()

    def free_gpu_memory(self):
        import xllamacpp as xlc
        devices = [d for d in xlc.get_device_info() if d["name"] != "CPU"]
        return int(devices[0]["memory_free"]) if len(devices) == 1 else 0

    def reload_context(self, context):
        self.close()
        self.params.n_ctx = context
        self.launch_settings["llm_n_ctx"] = context
        self._start()
