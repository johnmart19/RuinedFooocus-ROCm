"""Pinned public Pyodide assets; Python snippets execute only in the browser."""

import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile
import threading

import requests

VERSION = "314.0.7"
ASSET_PREFIX = f"/browser-python/{VERSION}/"
ARCHIVE_URL = f"https://github.com/pyodide/pyodide/releases/download/{VERSION}/pyodide-core-{VERSION}.tar.bz2"
ARCHIVE_SHA256 = "2abdcc2e35208af406e07724cffa85bc582ced97e9028383ecf5462541393f95"
PACKAGE_URL = f"https://cdn.jsdelivr.net/pyodide/v{VERSION}/full/"
CORE_HASHES = {
    "pyodide.js": "3141b814715a72e59b51b1b18b9ceae5bf19f7c852417e431bb0a34feadf825c",
    "pyodide.asm.mjs": "f7cdc8ece80678ceb712f8e65ebe6d3a83203a180c399865f49612a051693635",
    "pyodide.asm.wasm": "cc36e3cab04fdfc9a63ff13eb52eae2b911bf46c025cc7b281f394bd3de1d5e6",
    "python_stdlib.zip": "fa1957e5777068fc4f7437f96d860ae2fbe9c19732ba06c84e004ec16dd7dd7a",
    "pyodide-lock.json": "5dc2fc119108bc148c7457dc86e7675b5c87e1cafd420b9c34c1eaef7b36c010",
}
MIME_TYPES = {".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".json": "application/json"}
ASSET_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
    "Access-Control-Allow-Headers": "Range",
    "Cross-Origin-Resource-Policy": "cross-origin",
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "public, max-age=31536000, immutable",
}
MAX_ASSET_BYTES = 128 * 1024 * 1024
MAX_CACHE_BYTES = 512 * 1024 * 1024


def download(url, limit):
    """Fetch fixed vendor URLs with bounded transfer size and normal TLS checks."""
    with requests.get(url, stream=True, timeout=(10, 60)) as response:
        response.raise_for_status()
        if int(response.headers.get("Content-Length", 0)) > limit:
            raise ValueError("Browser Python asset exceeds the download limit.")
        content = bytearray()
        for chunk in response.iter_content(1024 * 128):
            content.extend(chunk)
            if len(content) > limit:
                raise ValueError("Browser Python asset exceeds the download limit.")
        return bytes(content)


class RuntimeAssets:
    def __init__(self, folder=None, fetch=download):
        self.folder = Path(folder) if folder else Path(__file__).resolve().parents[1] / "cache/browser-python" / VERSION
        self.fetch = fetch
        self.lock = threading.Lock()
        self.verified = {}

    def valid(self, path, digest):
        if not path.is_file() or path.is_symlink():
            return False
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns, digest)
        if self.verified.get(path) == signature:
            return True
        if stat.st_size > MAX_ASSET_BYTES or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            return False
        self.verified[path] = signature
        return True

    def write(self, name, data, digest):
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Browser Python checksum mismatch: {name}")
        self.folder.mkdir(parents=True, exist_ok=True)
        used = sum(p.stat().st_size for p in self.folder.iterdir() if p.is_file())
        previous = self.folder / name
        if used - (previous.stat().st_size if previous.is_file() else 0) + len(data) > MAX_CACHE_BYTES:
            raise ValueError("Browser Python cache limit reached.")
        with tempfile.NamedTemporaryFile(dir=self.folder, prefix=".asset-", delete=False) as temporary:
            temporary.write(data)
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, previous)
        finally:
            temporary_path.unlink(missing_ok=True)

    def prepare_core(self):
        """Call while holding the instance lock. Extract only verified basenames."""
        if all(self.valid(self.folder / name, digest) for name, digest in CORE_HASHES.items()):
            return
        archive_bytes = self.fetch(ARCHIVE_URL, 16 * 1024 * 1024)
        if hashlib.sha256(archive_bytes).hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Browser Python runtime archive checksum mismatch.")
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:bz2") as archive:
            for name, digest in CORE_HASHES.items():
                member = next((m for m in archive.getmembers()
                               if m.name in (name, "pyodide/" + name) and m.isfile()), None)
                if member is None or member.size > MAX_ASSET_BYTES:
                    raise ValueError(f"Browser Python runtime asset missing: {name}")
                stream = archive.extractfile(member)
                with stream:
                    self.write(name, stream.read(MAX_ASSET_BYTES + 1), digest)

    def asset(self, name):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", name) or ".." in name:
            raise KeyError(name)
        with self.lock:
            self.prepare_core()
            allowed = dict(CORE_HASHES)
            lockfile = json.loads((self.folder / "pyodide-lock.json").read_text(encoding="utf-8"))
            for package in lockfile["packages"].values():
                filename = package["file_name"]
                if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", filename):
                    allowed[filename] = package["sha256"]
            if name not in allowed:
                raise KeyError(name)
            path = self.folder / name
            if not self.valid(path, allowed[name]):
                self.write(name, self.fetch(PACKAGE_URL + name, MAX_ASSET_BYTES), allowed[name])
            return path


def install_routes(app, assets=None):
    """Expose public software assets only, without exposing private app files."""
    from fastapi import HTTPException
    from fastapi.responses import FileResponse, Response
    assets = assets or RuntimeAssets()

    def runtime_asset(filename: str):
        try:
            path = assets.asset(filename)
        except KeyError:
            raise HTTPException(404, "Unknown browser Python asset.") from None
        except (OSError, ValueError, requests.RequestException, tarfile.TarError) as error:
            import logging
            logging.warning("Browser Python asset unavailable: %s", error)
            raise HTTPException(503, "Could not prepare browser Python. Check the server connection and retry.") from None
        return FileResponse(path, media_type=MIME_TYPES.get(path.suffix, "application/octet-stream"), headers=ASSET_HEADERS)

    app.add_api_route(ASSET_PREFIX + "{filename}", runtime_asset, methods=["GET", "HEAD"], include_in_schema=False)
    app.add_api_route(ASSET_PREFIX + "{filename}", lambda filename: Response(status_code=204, headers=ASSET_HEADERS),
                      methods=["OPTIONS"], include_in_schema=False)
    return assets
