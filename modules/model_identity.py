"""Identify the local checkpoint without assuming a release's first file is it."""

import re
import json
from pathlib import Path


def read_safetensors_header(path):
    """Bounded shape/metadata inspection without importing the tensor runtime."""
    path = Path(path)
    if path.suffix.lower() != '.safetensors':
        raise ValueError('Select a safetensors checkpoint')
    with path.open('rb') as source:
        raw = source.read(8)
        size = int.from_bytes(raw, 'little')
        if len(raw) != 8 or not 2 <= size <= min(16 * 1024**2, path.stat().st_size - 8):
            raise ValueError('Invalid or oversized safetensors header')
        result = json.loads(source.read(size))
    if not isinstance(result, dict):
        raise ValueError('Invalid safetensors header')
    if any(not isinstance(value, dict) for key, value in result.items() if key != '__metadata__'):
        raise ValueError('Invalid tensor entries in safetensors header')
    return result


def checkpoint_hashes(metadata):
    files = metadata.get("files", [])
    files = (
        [item for item in files if isinstance(item, dict)]
        if isinstance(files, list)
        else []
    )
    local = metadata.get("_rf_local_sha256")
    if isinstance(local, str) and re.fullmatch(r"[a-fA-F0-9]{64}", local):
        for item in files:
            hashes = item.get("hashes", {})
            if (
                isinstance(hashes, dict)
                and str(hashes.get("SHA256", "")).lower() == local.lower()
            ):
                return hashes | {"SHA256": local.upper()}
        return {"SHA256": local.upper()}
    if len(files) == 1 and isinstance(files[0].get("hashes"), dict):
        return files[0]["hashes"]
    return {}
