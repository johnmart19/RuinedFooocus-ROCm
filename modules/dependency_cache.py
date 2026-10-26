"""Cache successful startup package maintenance using only the standard library."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

from modules.config_io import save_json


def fingerprint(root):
    root = Path(root)
    digest = hashlib.sha256()
    digest.update(repr((sys.executable, sys.prefix, sys.version)).encode())
    for path in sorted([*root.glob('requirements*.txt'), *root.glob('pip/*.txt'),
                        root / 'launch.py', root / 'entry_with_update.py']):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    try:
        digest.update(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                             stderr=subprocess.DEVNULL))
    except (OSError, subprocess.CalledProcessError):
        pass
    for directory in sorted({sysconfig.get_path('purelib'), sysconfig.get_path('platlib')}):
        for path in sorted(Path(directory).glob('*.dist-info')):
            metadata = path / 'METADATA'
            if metadata.is_file():
                stat = metadata.stat()
                digest.update(repr((str(path), stat.st_size, stat.st_mtime_ns)).encode())
    return digest.hexdigest()


def needed(root, stage, force=False):
    if force:
        return True
    try:
        saved = json.loads((Path(root) / 'cache' / 'dependency-checks.json').read_text())
        return not isinstance(saved, dict) or saved.get(stage) != fingerprint(root)
    except (OSError, ValueError):
        return True


def record(root, stage):
    path = Path(root) / 'cache' / 'dependency-checks.json'
    try:
        saved = json.loads(path.read_text())
    except (OSError, ValueError):
        saved = {}
    if not isinstance(saved, dict):
        saved = {}
    saved[stage] = fingerprint(root)
    save_json(path, saved)
