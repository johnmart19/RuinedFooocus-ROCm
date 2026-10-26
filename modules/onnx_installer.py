"""Install the optional Windows GPU ONNX wheel without replacing CPU ONNX."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig

VERSION = '1.30.0.202609102321'
ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name('onnx_worker.py')


def worker_command(root, probe=False, device_id=None, worker=None):
    # Windows venv python.exe is a redirector. Launch the real interpreter so
    # terminating an owned worker cannot leave its interpreter child behind.
    python = getattr(sys, '_base_executable', None) or sys.executable
    command = [python, '-I', '-u', str(worker or WORKER), '--runtime', str(root),
               '--site-dir', sysconfig.get_path('purelib')]
    if probe:
        command.append('--probe')
    if device_id is not None:
        command += ['--device-id', str(device_id)]
    return command


def runtime_path():
    abi = f'cp{sys.version_info.major}{sys.version_info.minor}'
    return ROOT / 'cache' / 'onnx' / abi / VERSION


def runtime_available():
    root = runtime_path()
    try:
        marker = json.loads((root / 'verified.json').read_text())
        return (marker['version'] == VERSION and marker['python'] == list(sys.version_info[:2])
                and hashlib.sha256((root / marker['binary']).read_bytes()).hexdigest() == marker['sha256'])
    except (OSError, ValueError, KeyError):
        return False


def install_plan(target=None):
    if platform.system() != 'Windows' or platform.machine().lower() not in ('amd64', 'x86_64'):
        raise RuntimeError('This optional DirectML runtime requires Windows x64.')
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError('The validated DirectML runtime supports Python 3.11–3.13; use native/CPU inference otherwise.')
    return [sys.executable, '-m', 'pip', 'install', '--upgrade', '--only-binary=:all:', '--no-deps',
            '--target', str(target or runtime_path()), f'onnxruntime-windowsml=={VERSION}']


def install_runtime():
    root = runtime_path()
    root.mkdir(parents=True, exist_ok=True)
    if not runtime_available():
        subprocess.run(install_plan(), check=True)
    # The CPU ONNX dependency already supplies these packages. Validate the
    # isolated wheel's actual requirements instead of silently ignoring them.
    import importlib.metadata as metadata
    from packaging.requirements import Requirement
    for distribution in metadata.distributions(path=[str(root)]):
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            if not requirement.specifier.contains(metadata.version(requirement.name), prereleases=True):
                raise RuntimeError(f'Optional ONNX runtime requires {requirement}.')
    result = subprocess.run(worker_command(root, probe=True),
                            check=True, capture_output=True, text=True, timeout=60)
    report = json.loads(result.stdout.strip().splitlines()[-1])
    if not report.get('ok') or report['version'] != VERSION.split('.2026')[0]:
        raise RuntimeError('DirectML GPU validation failed.')
    binary = next((root / 'onnxruntime/capi').glob('onnxruntime_pybind11_state*.pyd'))
    marker = dict(version=VERSION, python=list(sys.version_info[:2]),
                  binary=str(binary.relative_to(root)), sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    (root / 'verified.json').write_text(json.dumps(marker, indent=2))
    return 'Windows GPU ONNX runtime installed; DirectML inference verified.'
