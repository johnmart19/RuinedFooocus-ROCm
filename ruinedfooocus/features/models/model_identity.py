"""Identify the local checkpoint without assuming a release's first file is it."""

from functools import lru_cache
import re
import json
from pathlib import Path


def file_signature(path):
    path = Path(path)
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def files_unchanged(signatures):
    if signatures is None:
        return False
    try:
        return all(signature == file_signature(signature[0]) for signature in signatures)
    except OSError:
        return False


def read_safetensors_header(path):
    """Bounded shape/metadata inspection without importing the tensor runtime."""
    path = Path(path)
    if path.suffix.lower() != ".safetensors":
        raise ValueError("Select a safetensors checkpoint")
    with path.open("rb") as source:
        raw = source.read(8)
        size = int.from_bytes(raw, "little")
        if len(raw) != 8 or not 2 <= size <= min(16 * 1024**2, path.stat().st_size - 8):
            raise ValueError("Invalid or oversized safetensors header")
        result = json.loads(source.read(size))
    if not isinstance(result, dict):
        raise ValueError("Invalid safetensors header")
    if any(not isinstance(value, dict) for key, value in result.items() if key != "__metadata__"):
        raise ValueError("Invalid tensor entries in safetensors header")
    return result


@lru_cache(maxsize=256)
def _cached_header(path, signature):
    return read_safetensors_header(path)


def cached_safetensors_header(path):
    """Header of an unchanged file is parsed once; a replaced file is read again."""
    path = Path(path)
    return _cached_header(str(path), tuple(file_signature(path)[1:]))


def checkpoint_hashes(metadata):
    files = metadata.get("files", [])
    files = [item for item in files if isinstance(item, dict)] if isinstance(files, list) else []
    local = metadata.get("_rf_local_sha256")
    if isinstance(local, str) and re.fullmatch(r"[a-fA-F0-9]{64}", local):
        for item in files:
            hashes = item.get("hashes", {})
            if isinstance(hashes, dict) and str(hashes.get("SHA256", "")).lower() == local.lower():
                return hashes | {"SHA256": local.upper()}
        return {"SHA256": local.upper()}
    if len(files) == 1 and isinstance(files[0].get("hashes"), dict):
        return files[0]["hashes"]
    return {}
