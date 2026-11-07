"""Resumable Hub downloads with the application's byte-progress callback."""

import os
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from tqdm import tqdm


def download_hub_file(url, destination, progress=None):
    parts = urlparse(url).path.strip("/").split("/")
    if len(parts) < 5 or parts[2] != "resolve":
        raise ValueError("Expected a Hugging Face model resolve URL")
    repository = "/".join(parts[:2])
    revision = unquote(parts[3])
    filename = unquote("/".join(parts[4:]))
    destination = Path(destination)

    class DownloadProgress(tqdm):
        def __init__(self, *args, **kwargs):
            self.started = self.last_update = time.monotonic()
            super().__init__(*args, **kwargs)
            self.initial_bytes = self.n
            if progress:
                progress(self.n, self.total or 0, 0)

        def update(self, amount=1):
            # tqdm skips its counter when terminal bars are disabled; UI still needs it.
            if self.disable:
                self.n += amount
            result = super().update(amount)
            now = time.monotonic()
            if progress and now - self.last_update >= 0.5:
                progress(
                    self.n,
                    self.total or 0,
                    (self.n - self.initial_bytes) / max(now - self.started, 0.001),
                )
                self.last_update = now
            return result

    try:
        downloaded = Path(
            hf_hub_download(
                repo_id=repository,
                filename=filename,
                revision=revision,
                local_dir=destination.parent,
                tqdm_class=DownloadProgress,
            )
        )
    except HfHubHTTPError as error:
        if error.response is not None and error.response.status_code in (401, 403):
            raise RuntimeError(
                f"Cannot download {destination.name}: Hugging Face access is required. "
                f"Request access at https://huggingface.co/{repository}, then use hf auth login "
                f"or HF_TOKEN. Alternatively, place the file at {destination}."
            ) from error
        raise
    if downloaded.resolve() != destination.resolve():
        os.replace(downloaded, destination)
    if progress:
        size = destination.stat().st_size
        progress(size, size, 0)
