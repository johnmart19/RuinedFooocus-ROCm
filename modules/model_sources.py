"""Public metadata lookups; never download checkpoint weights."""

import re
from pathlib import Path
from urllib.parse import quote, urlparse

import requests

_civitai_not_found = {}


def huggingface_recommendations(repository, sha256):
    """Read bounded JSON/card data only after matching a cached checkpoint hash.

    The hash is compared locally with the public manifest, never sent to HF.
    Pin card/config reads to the matched repository revision. No weights/code.
    """
    if (not isinstance(repository, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", repository)
            or not isinstance(sha256, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", sha256)):
        return None

    def read(url, limit, as_json=False, params=None, with_url=False):
        with requests.get(url, params=params, timeout=(5, 15), stream=True) as response:
            response.raise_for_status()
            chunks = []
            size = 0
            for chunk in response.iter_content(16384):
                size += len(chunk)
                if size > limit:
                    raise ValueError("Recommendation metadata exceeds size limit")
                chunks.append(chunk)
            body = b"".join(chunks).decode("utf-8")
            if as_json:
                import json
                value = json.loads(body)
                return (value, getattr(response, "url", url)) if with_url else value
            return body

    try:
        info, resolved_url = read(f"https://huggingface.co/api/models/{repository}", 2*1024*1024,
                                 as_json=True, params={"blobs": "true"}, with_url=True)
        resolved = urlparse(resolved_url)
        if (not isinstance(info, dict) or not isinstance(info.get("id"), str)
                or resolved.scheme != "https" or resolved.hostname != "huggingface.co"
                or resolved.path.lower() != f"/api/models/{info['id']}".lower()
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", info["id"])):
            return None
        repository = info["id"]
        siblings = info.get("siblings", [])
        if not isinstance(siblings, list) or not any(
                isinstance(item, dict) and isinstance(item.get("lfs"), dict)
                and item["lfs"].get("sha256", "").lower() == sha256.lower()
                for item in siblings if isinstance(item, dict)):
            return None
        revision = info.get("sha")
        if not isinstance(revision, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
            return None
        names = {item.get("rfilename") for item in siblings if isinstance(item, dict) and isinstance(item.get("rfilename"), str)}
        result = {"description": "", "generationConfig": {},
                  "source": f"Hugging Face {repository} @ {revision[:12]} (repository-wide settings)",
                  "source_url": f"https://huggingface.co/{repository}/blob/{revision}/README.md"}
        card_data = info.get("cardData")
        parents = card_data.get("base_model", []) if isinstance(card_data, dict) else []
        parents = [parents] if isinstance(parents, str) else parents
        result["source_repositories"] = [name for name in parents if isinstance(name, str)][:3] if isinstance(parents, list) else []
        base = f"https://huggingface.co/{repository}/resolve/{revision}/"
        if "README.md" in names:
            result["description"] = read(base+"README.md", 200000)
        if "generation_config.json" in names:
            try:
                config = read(base+"generation_config.json", 64000, as_json=True)
                if isinstance(config, dict):
                    result["generationConfig"] = config
            except (requests.RequestException, ValueError, UnicodeError):
                pass  # A valid card remains useful without a generation config.
        return result
    except (requests.RequestException, ValueError, UnicodeError, AttributeError) as error:
        print(f"Hugging Face recommendation lookup failed: {error}")
        return None


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
                      if Path(item["rfilename"]).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm")]
            images.sort(key=lambda name: (Path(name).stem != path.stem, name))
            previews = [{"url": f"https://huggingface.co/{repository}/resolve/{revision}/{quote(name, safe='/')}",
                         "type": "video" if Path(name).suffix.lower() in (".mp4", ".webm") else "image"} for name in images[:5]]
            return {"hf_repo_id": repository, "images": previews}
        except (requests.RequestException, ValueError, TypeError, KeyError) as error:
            print(f"Hugging Face artwork lookup failed for {repository}: {error}")
    return None
