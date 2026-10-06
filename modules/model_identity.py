"""Identify the local checkpoint without assuming a release's first file is it."""

import re


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
