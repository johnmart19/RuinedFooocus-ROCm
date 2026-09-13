"""Public metadata lookups; never download checkpoint weights."""

import re
from pathlib import Path
from urllib.parse import quote, urlparse

import requests


_civitai_not_found = {}


def civitai_metadata(endpoint):
    import time
    for host in ("civitai.com", "civitai.red"):
        key = (host, endpoint)
        if time.monotonic() - _civitai_not_found.get(key, -float("inf")) < 300:
            continue
        try:
            response = requests.get(f"https://{host}/api/v1/{endpoint}", timeout=(5, 20))
            if response.status_code == 404:
                # A model may exist only on Hugging Face. This is a normal miss.
                if len(_civitai_not_found) >= 256:
                    _civitai_not_found.pop(next(iter(_civitai_not_found)))
                _civitai_not_found[key] = time.monotonic()
                continue
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and data.get("id"):
                return data
        except (requests.RequestException, ValueError) as error:
            print(f"Model metadata lookup failed on {host}: {error}")
    return None


def huggingface_preview(path, sha256, source=""):
    """Use repository artwork only after matching a checkpoint's LFS SHA-256."""
    if not sha256:
        return None
    path = Path(path)
    repositories = []
    url = urlparse(source) if isinstance(source, str) else urlparse("")
    parts = url.path.strip("/").split("/")
    if url.hostname == "huggingface.co" and len(parts) >= 2:
        repositories.append("/".join(parts[:2]))
    words = re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+", path.stem)
    query = words[0] if words and len(words[0]) >= 3 else path.stem.split("_")[0]
    if "-" in path.stem:
        query = re.split(r"[-_](?:\d+b|fp\d+|bf\d+|int\d+|q\d+|transformer|distilled|comfy|convrot)",
                           path.stem, maxsplit=1, flags=re.IGNORECASE)[0]
    if path.suffix.lower() == ".gguf" and not query.lower().endswith("gguf"):
        query += "-GGUF"
    try:
        response = requests.get("https://huggingface.co/api/models",
                                params={"search": query, "limit": 20}, timeout=(5, 20))
        response.raise_for_status()
        repositories.extend(item["id"] for item in response.json() if isinstance(item, dict) and "id" in item)
    except (requests.RequestException, ValueError, TypeError) as error:
        print(f"Hugging Face model search failed: {error}")
    for repository in dict.fromkeys(repositories):
        try:
            response = requests.get(f"https://huggingface.co/api/models/{quote(repository, safe='/')}",
                                    params={"blobs": "true"}, timeout=(5, 20))
            response.raise_for_status()
            data = response.json()
            files = data.get("siblings", [])
            if not any((item.get("lfs") or {}).get("sha256", "").lower() == sha256.lower()
                       for item in files):
                continue
            revision = quote(data.get("sha") or "main", safe="")
            images = [item["rfilename"] for item in files
                      if Path(item["rfilename"]).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
            images.sort(key=lambda name: (Path(name).stem != path.stem, name))
            previews = [{"url": f"https://huggingface.co/{repository}/resolve/{revision}/{quote(name, safe='/')}",
                         "type": "image"} for name in images[:5]]
            # Some repositories publish no image files or gate their assets.
            previews.append({"url": f"https://cdn-thumbnails.huggingface.co/social-thumbnails/models/{repository}.png",
                             "type": "image"})
            return {"hf_repo_id": repository, "images": previews}
        except (requests.RequestException, ValueError, TypeError, KeyError) as error:
            print(f"Hugging Face artwork lookup failed for {repository}: {error}")
    return None
