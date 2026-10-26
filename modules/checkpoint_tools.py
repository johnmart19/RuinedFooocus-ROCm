"""Local safetensors inspection and explicit, non-destructive FP32 conversion."""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from safetensors import safe_open

CHUNK_SIZE = 16 * 1024 * 1024


def _header(path):
    path = Path(path)
    if path.suffix.lower() != ".safetensors":
        raise ValueError("Select a .safetensors checkpoint.")
    # Validate offsets, shapes and file length using the format's own reader.
    with safe_open(path, framework="pt", device="cpu"):
        pass
    with path.open("rb") as source:
        size = int.from_bytes(source.read(8), "little")
        if size > 16 * 1024 * 1024:
            raise ValueError("Checkpoint header is too large to inspect.")
        return json.loads(source.read(size)), size + 8


def inspect_checkpoint(path):
    path = Path(path)
    header, _ = _header(path)
    sizes = Counter()
    tensors = [v for k, v in header.items() if k != "__metadata__"]
    for tensor in tensors:
        start, end = tensor["data_offsets"]
        sizes[tensor["dtype"]] += end - start
    return dict(name=path.name, tensors=len(tensors), bytes=path.stat().st_size,
                dtypes=dict(sizes), saving_bytes=sizes["F32"] // 2,
                previous_repair=header.get("__metadata__", {}).get("donor_repair") == "Applied")


def _float_values(raw, dtype):
    if dtype in ("F16", "F32", "F64"):
        return np.frombuffer(raw, dtype={"F16": "<f2", "F32": "<f4", "F64": "<f8"}[dtype])
    if dtype == "BF16":
        return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    if dtype.startswith("F"):
        import torch
        names = {"F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
                 "F8_E4M3FNUZ": "float8_e4m3fnuz", "F8_E5M2FNUZ": "float8_e5m2fnuz",
                 "F8_E8M0": "float8_e8m0fnu"}
        target = getattr(torch, names.get(dtype, ""), None)
        if target is None:
            raise ValueError(f"Weight checking is not supported for {dtype}.")
        return torch.frombuffer(bytearray(raw), dtype=target).float().numpy()
    return None


def _tensors(header):
    return sorted(((k, v) for k, v in header.items() if k != "__metadata__"),
                  key=lambda item: item[1]["data_offsets"][0])


def _stamp(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def check_weights(path, progress=lambda fraction, description: None):
    path = Path(path)
    before = _stamp(path)
    header, offset = _header(path)
    digest = hashlib.sha256()
    invalid = []
    zeros = 0
    with path.open("rb") as source:
        digest.update(source.read(offset))
        for name, tensor in _tensors(header):
            start, end = tensor["data_offsets"]
            remaining = end - start
            source.seek(offset + start)
            nan = inf = 0
            all_zero = remaining > 0
            while remaining:
                raw = source.read(min(CHUNK_SIZE, remaining))
                if not raw:
                    raise ValueError("Checkpoint was truncated during checking.")
                digest.update(raw)
                remaining -= len(raw)
                values = _float_values(raw, tensor["dtype"])
                if values is not None:
                    nan += int(np.count_nonzero(np.isnan(values)))
                    inf += int(np.count_nonzero(np.isinf(values)))
                    all_zero &= not np.any(values)
                else:
                    all_zero = False
                progress(source.tell() / before[0], "Checking weights")
            if nan or inf:
                invalid.append(dict(name=name, nan=nan, inf=inf))
            zeros += int(all_zero)
    if _stamp(path) != before:
        raise ValueError("Checkpoint changed during checking; try again.")
    return dict(sha256=digest.hexdigest(), invalid=invalid, zero_tensors=zeros)


def create_fp16_copy(path, cache_dir, model_info=None, preview=None,
                     progress=lambda fraction, description: None):
    path = Path(path)
    report = inspect_checkpoint(path)
    if not report["saving_bytes"]:
        return dict(path=None, message="No FP32 tensors to reduce. No copy created.")
    if report["saving_bytes"] < sum(report["dtypes"].values()) * 0.01:
        return dict(path=None, message="FP16 would save less than 1%. No copy created.")
    destination = path.with_name(path.stem + ".fp16.safetensors")
    cache_dir = Path(cache_dir)
    cached = cache_dir / destination.name
    metadata_destination = cached.with_suffix(".json")
    preview = Path(preview) if preview else None
    preview_destination = cached.with_suffix(preview.suffix) if preview else None
    for target in (destination, metadata_destination, preview_destination):
        if target and target.exists():
            raise FileExistsError(f"Already exists: {target.name}")
    needed = report["bytes"] - report["saving_bytes"] + CHUNK_SIZE
    if shutil.disk_usage(path.parent).free < needed:
        raise ValueError("Not enough free disk space for the FP16 copy.")
    before = _stamp(path)
    checked = check_weights(path, progress)
    if checked["invalid"]:
        raise ValueError("Checkpoint contains NaN/Inf weights. Conversion stopped.")
    header, source_offset = _header(path)
    converted = {}
    position = 0
    for name, tensor in _tensors(header):
        start, end = tensor["data_offsets"]
        size = (end - start) // 2 if tensor["dtype"] == "F32" else end - start
        converted[name] = tensor | {"data_offsets": [position, position + size]}
        if tensor["dtype"] == "F32":
            converted[name]["dtype"] = "F16"
        position += size
    converted["__metadata__"] = header.get("__metadata__", {}) | {
        "rf.source_sha256": checked["sha256"], "rf.source_name": path.name,
        "rf.conversion": "FP32 to FP16; other dtypes preserved",
    }
    encoded = json.dumps(converted, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    underflow = 0
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (
        TemporaryDirectory(prefix=".checkpoint-", dir=path.parent) as directory,
        TemporaryDirectory(prefix=".checkpoint-", dir=cache_dir) as cache_temporary,
    ):
        temporary = Path(directory) / "weights.tmp"
        digest = hashlib.sha256()
        with path.open("rb") as source, temporary.open("wb") as output:
            prefix = len(encoded).to_bytes(8, "little") + encoded
            output.write(prefix)
            digest.update(prefix)
            for name, tensor in _tensors(header):
                start, end = tensor["data_offsets"]
                source.seek(source_offset + start)
                remaining = end - start
                while remaining:
                    raw = source.read(min(CHUNK_SIZE, remaining))
                    if not raw:
                        raise ValueError("Checkpoint was truncated during conversion.")
                    remaining -= len(raw)
                    if tensor["dtype"] == "F32":
                        values = np.frombuffer(raw, dtype="<f4")
                        with np.errstate(over="ignore", under="ignore"):
                            reduced = values.astype("<f2")
                        if not np.isfinite(reduced).all():
                            raise ValueError(f"FP16 overflow in {name}. Conversion stopped.")
                        underflow += int(np.count_nonzero((values != 0) & (reduced == 0)))
                        raw = reduced.tobytes()
                    output.write(raw)
                    digest.update(raw)
                    progress(source.tell() / before[0], "Writing FP16 copy")
            output.flush()
            os.fsync(output.fileno())
        with safe_open(temporary, framework="pt", device="cpu"):
            pass
        with temporary.open("rb") as output:
            verified = hashlib.sha256()
            for chunk in iter(lambda: output.read(CHUNK_SIZE), b""):
                verified.update(chunk)
            output_hash = verified.hexdigest()
        if output_hash != digest.hexdigest() or _stamp(path) != before:
            raise ValueError("File verification failed. Conversion stopped.")
        info = dict(model_info or {})
        info.update(baseModel=info.get("baseModel") or "Unknown",
                    files=[{"name": destination.name, "hashes": {"SHA256": output_hash}}],
                    rf_source={"name": path.name, "sha256": checked["sha256"]})
        metadata = Path(cache_temporary) / "metadata.tmp"
        metadata.write_text(json.dumps(info, indent=2), encoding="utf-8")
        publish = [(metadata, metadata_destination)]
        if preview:
            staged_preview = Path(cache_temporary) / "preview.tmp"
            shutil.copyfile(preview, staged_preview)
            publish.append((staged_preview, preview_destination))
        publish.append((temporary, destination))
        created = []
        try:
            for source, target in publish:
                # Atomic and no-overwrite on Windows NTFS and Linux filesystems.
                os.link(source, target)
                created.append(target)
        except OSError:
            for target in reversed(created):
                target.unlink()
            raise
    return dict(path=destination, preview=preview_destination, underflow=underflow,
                message=f"Created {destination.name}. Saved {report['saving_bytes'] / 2**30:.2f} GiB. "
                        f"Values rounded to zero: {underflow}. Original retained.")


def format_report(report, family="Unknown", checked=None):
    dtypes = ", ".join(f"{name}: {size / 2**30:.2f} GiB" for name, size in report["dtypes"].items())
    lines = [report["name"], f"Model: {family} | {report['tensors']} tensors",
             f"Tensor data: {dtypes}", "Generation also needs memory for activations and decoding.",
             f"FP32 to FP16 saving: {report['saving_bytes'] / 2**20:.2f} MiB"]
    if report["previous_repair"]:
        lines.append("Previous optimizer reports donor-replaced weights; this cannot verify their origin.")
    if checked is None:
        lines.append("Weight values have not been checked.")
    else:
        lines.append(f"NaN/Inf tensors: {len(checked['invalid'])}. All-zero tensors: {checked['zero_tensors']} (may be intentional).")
        lines.extend(f"{item['name']}: {item['nan']} NaN, {item['inf']} Inf" for item in checked["invalid"][:8])
        lines.append(f"SHA-256 (local): {checked['sha256']}")
    return "\n".join(lines)
