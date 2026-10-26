"""Optional isolated GPU mask inference with a native CPU fallback."""
import json
import logging
from pathlib import Path
import platform
import queue
import subprocess
import tempfile
import threading
import time

from modules.auxiliary_resources import register
from modules.onnx_installer import ROOT, WORKER, runtime_available, runtime_path, worker_command


class GPUWorkerError(RuntimeError):
    pass


def release_torch_weights():
    from comfy import model_management
    # ONNX has its own allocator: release unused Torch weights before it
    # creates GPU resources rather than counting on Torch's OOM recovery.
    model_management.unload_all_models()
    model_management.soft_empty_cache()


class NativeMaskSession:
    """Give installed native CUDA/ROCm sessions the same resource lifetime."""
    resource_group = 'rembg'

    def __init__(self, factory):
        self.factory = factory
        self.session = None
        self.disabled = False
        self.check_cancelled = None
        register(self)

    def predict(self, image, *args, **kwargs):
        from onnxruntime.capi.onnxruntime_pybind11_state import (
            EPFail, EngineError, Fail, RuntimeException,
        )
        if self.check_cancelled:
            self.check_cancelled()
        try:
            if self.session is None:
                if not self.disabled:
                    release_torch_weights()
                self.session = self.factory() if not self.disabled else self._cpu_session()
            result = self.session.predict(image, *args, **kwargs)
        except (EPFail, EngineError, Fail, RuntimeException) as error:
            if self.disabled:
                raise
            self.close()
            self.disabled = True
            logging.warning('Native GPU background removal failed; retrying on CPU: %s', error)
            if self.check_cancelled:
                self.check_cancelled()
            self.session = self._cpu_session()
            result = self.session.predict(image, *args, **kwargs)
        if self.check_cancelled:
            self.check_cancelled()
        return result

    @staticmethod
    def _cpu_session():
        import rembg
        return rembg.new_session(providers=['CPUExecutionProvider'])

    def close(self):
        self.session = None


class DirectMLMaskSession:
    resource_group = 'rembg'

    def __init__(self, device_id=None):
        self.device_id = device_id
        self.process = None
        self.reader = None
        self.log = None
        self.messages = queue.Queue()
        self.check_cancelled = None
        self.disabled = False
        self.cpu_session = None
        register(self)

    def _read_messages(self, process, messages):
        try:
            for line in process.stdout:
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if isinstance(value, dict):
                    messages.put(value)
        except (OSError, ValueError):
            pass
        finally:
            messages.put({'error': 'GPU mask worker exited.'})

    def _wait(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.check_cancelled:
                try:
                    self.check_cancelled()
                except BaseException:
                    self.close()
                    raise
            try:
                message = self.messages.get(timeout=0.1)
            except queue.Empty:
                continue
            if message.get('error'):
                raise GPUWorkerError(message['error'])
            return message
        raise GPUWorkerError('GPU mask worker timed out.')

    def _start(self):
        release_torch_weights()
        log_path = ROOT / 'cache/onnx/worker.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.messages = queue.Queue()
        command = worker_command(runtime_path(), device_id=self.device_id, worker=WORKER)
        try:
            self.log = log_path.open('a', encoding='utf-8')
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.log, text=True, encoding='utf-8',
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except OSError as error:
            raise GPUWorkerError(f'Cannot start the GPU mask worker: {error}') from error
        self.reader = threading.Thread(target=self._read_messages,
                                       args=(self.process, self.messages), daemon=True)
        self.reader.start()
        if not self._wait(180).get('ready'):
            raise GPUWorkerError('GPU mask worker did not initialize.')
        logging.info('Background removal: isolated DirectML GPU worker ready.')

    def _cpu_predict(self, image, *args, **kwargs):
        import rembg
        if self.check_cancelled:
            self.check_cancelled()
        if self.cpu_session is None:
            self.cpu_session = rembg.new_session(providers=['CPUExecutionProvider'])
        result = self.cpu_session.predict(image, *args, **kwargs)
        if self.check_cancelled:
            self.check_cancelled()
        return result

    def predict(self, image, *args, **kwargs):
        # Preserve uncommon model-specific Python options without narrowing the
        # native session contract or silently discarding unsupported arguments.
        if self.disabled or args or kwargs:
            self.close()
            return self._cpu_predict(image, *args, **kwargs)
        try:
            if self.process is None:
                self._start()
            jobs = ROOT / 'cache/onnx/jobs'
            jobs.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='mask-', dir=jobs) as folder:
                source = Path(folder) / 'input.png'
                image.save(source)
                try:
                    self.process.stdin.write(json.dumps({'input': str(source), 'output': folder}) + '\n')
                    self.process.stdin.flush()
                except (OSError, ValueError) as error:
                    raise GPUWorkerError('GPU mask worker connection failed.') from error
                response = self._wait(120)
                from PIL import Image
                masks = []
                for filename in response.get('masks', []):
                    path = Path(filename).resolve()
                    if path.parent != Path(folder).resolve():
                        raise GPUWorkerError('GPU mask worker returned an invalid output path.')
                    with Image.open(path) as mask:
                        masks.append(mask.copy())
                if not masks:
                    raise GPUWorkerError('GPU mask worker returned no mask.')
                return masks
        except GPUWorkerError as error:
            self.close()
            self.disabled = True
            logging.warning('GPU background removal unavailable; using CPU: %s', error)
            return self._cpu_predict(image)

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for stream in (process.stdin, process.stdout):
                if stream:
                    try:
                        stream.close()
                    except OSError:
                        # A dead worker can leave buffered input that cannot
                        # be flushed. Cleanup must not prevent CPU recovery.
                        pass
        if self.reader is not None:
            self.reader.join(timeout=1)
            self.reader = None
        if self.log is not None:
            self.log.close()
            self.log = None


def new_session(cpu=False, preference=None):
    import rembg
    if cpu or preference == 'CPU':
        return rembg.new_session(providers=['CPUExecutionProvider'])
    if preference is None:
        from shared import settings
        preference = settings.default_settings.get('rembg_backend', 'Auto')
    if preference == 'CPU':
        return rembg.new_session(providers=['CPUExecutionProvider'])
    if preference not in ('Auto', 'Windows GPU'):
        raise ValueError('Choose Auto, CPU or Windows GPU for background removal.')
    import onnxruntime as ort
    native_gpu = set(ort.get_available_providers()) & {'CUDAExecutionProvider', 'ROCMExecutionProvider'}
    from argparser import args
    if preference == 'Auto' and native_gpu and args.directml is None:
        return NativeMaskSession(rembg.new_session)
    if platform.system() == 'Windows' and runtime_available():
        device_id = args.directml if args.directml is not None and args.directml >= 0 else None
        return DirectMLMaskSession(device_id=device_id)
    if preference == 'Windows GPU':
        logging.warning('Optional Windows GPU ONNX runtime is not installed; using native/CPU removal.')
    if args.directml is not None:
        return rembg.new_session(providers=['CPUExecutionProvider'])
    return rembg.new_session()
