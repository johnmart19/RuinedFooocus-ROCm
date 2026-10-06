"""Checkpoint recipes from public version metadata, with separate local overrides.

Model descriptions are data: only recognised numeric and sampling fields are read.
Prompts, example images, embedded workflows and executable instructions are ignored.
"""
import json
import math
import re
import threading
import time
from html import unescape
from pathlib import Path

from modules.config_io import save_json

FIELDS = ("custom_steps", "cfg", "sampler_name", "scheduler", "clip_skip")
RECOMMENDATION_SCHEMA = 2
FAMILY_PRESETS = {
    "Anima": "Anima", "SDXL 1.0": "SDXL", "Illustrious": "Illustrious XL",
    "NoobAI": "NoobAI XL", "Pony": "Pony XL",
    "Flux.1 D": "FLUX.1 dev", "Flux.1 S": "FLUX.1 schnell",
    "QwenImage": "Qwen Image", "Qwen": "Qwen Image", "Qwen-Image": "Qwen Image",
    "SD 3.5": "SD3.5", "SD 3.5 Large": "SD3.5",
}


def validated(values, samplers, schedulers):
    if not isinstance(values, dict):
        return {}
    result = {}
    for key, low, high in (("custom_steps", 1, 200), ("cfg", 0, 20), ("clip_skip", 1, 5)):
        value = values.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and low <= value <= high and math.isfinite(value):
            if key == "cfg" or int(value) == value:
                result[key] = value if key == "cfg" else int(value)
    for key, allowed in (("sampler_name", samplers), ("scheduler", schedulers)):
        if isinstance(values.get(key), str) and values[key] in allowed:
            result[key] = values[key]
    return result


def extract(metadata, samplers, schedulers):
    """Prefer structured version settings, then explicit labelled card fields.

    For ranges use the lower recommended value; retain the card as the source.
    Ambiguous sampler alternatives are left for the user to select.
    """
    config = metadata.get("generationConfig") or {}
    if not isinstance(config, dict):
        config = {}
    aliases = {"steps": "custom_steps", "num_inference_steps": "custom_steps",
               "cfgScale": "cfg", "guidance_scale": "cfg", "sampler": "sampler_name", "clipSkip": "clip_skip"}
    values = {aliases.get(key, key): value for key, value in config.items()}
    text = unescape(re.sub(r"<[^>]*>", "\n", str(metadata.get("description") or "")[:200000]))
    text = re.sub(r"(?m)^\s*[-*]\s+", "", text.replace("**", ""))
    for key, label in (("custom_steps", "Steps"), ("cfg", r"CFG(?:\s*Scale)?"), ("clip_skip", r"Clip\s*Skip")):
        matches = re.findall(r"(?im)^\s*" + label + r"\s*[:：]\s*(\d+(?:\.\d+)?)", text)
        if len(set(matches)) == 1 and key not in values:
            number = float(matches[0])
            values[key] = number
    # Public cards often publish these settings as literal pipeline arguments.
    # Read numbers only; never evaluate code or guess between conflicting examples.
    for key, argument in (("custom_steps", "num_inference_steps"), ("cfg", "guidance_scale")):
        matches = re.findall(r"(?m)^\s*" + argument + r"\s*=\s*([+-]?\d+(?:\.\d+)?)\s*(?:,|$)", text)
        numbers = {float(value) for value in matches}
        if len(numbers) == 1 and key not in values:
            values[key] = numbers.pop()
    # Only inspect sampling-labelled lines, never a random mention in prose.
    lines = re.findall(r"(?im)^\s*(?:Sampler(?:\s*/\s*Scheduler)?|Scheduler)\s*[:：]\s*([^\n]+)", text)
    candidates = {"sampler_name": set(), "scheduler": set()}
    for line in lines:
        normalized = line.lower().replace("sgm-uniform", "sgm_uniform").replace("res-multistep", "res_multistep").replace("er-sde", "er_sde")
        if re.search(r"\beuler\s+a\b", normalized):
            normalized = re.sub(r"\beuler\s+a\b", "euler_ancestral", normalized)
        for key, allowed in (("sampler_name", samplers), ("scheduler", schedulers)):
            found = [name for name in allowed if re.search(r"(?<![\w])" + re.escape(name) + r"(?![\w])", normalized)]
            candidates[key].update(found)
    for key, found in candidates.items():
        if len(found) == 1 and key not in values:
            values[key] = next(iter(found))
    return validated(values, samplers, schedulers)


class ModelRecommendations:
    def __init__(self, cache_dir="cache/checkpoints", path="settings/model_settings.json", offline=False):
        self.cache_dir = Path(cache_dir)
        self.path = Path(path)
        self.lock = threading.RLock()
        self.offline = offline

    def metadata(self, checkpoint):
        if not isinstance(checkpoint, str) or not checkpoint:
            return {}
        # The cache is named after the checkpoint stem; never write into models.
        try:
            data = json.loads((self.cache_dir / (Path(checkpoint).stem + ".json")).read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def load(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return {key: value for key, value in data.items() if isinstance(value, dict)} if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def identity(self, checkpoint, metadata=None):
        if metadata is None:
            metadata = self.metadata(checkpoint)
        if not metadata.get("id") and metadata.get("hf_repo_id") and self.cached_hash(metadata):
            return f"{checkpoint}:hf:{metadata['hf_repo_id']}:{self.cached_hash(metadata)}"
        return f"{checkpoint}:{metadata.get('id', 'local')}"

    @staticmethod
    def cached_hash(metadata):
        from modules.model_identity import checkpoint_hashes
        value = checkpoint_hashes(metadata).get("SHA256")
        if isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{64}", value):
            return value.lower()
        return None

    def refresh(self, checkpoint, samplers, schedulers, force=False):
        if self.offline:
            return "Offline metadata mode; retained local settings."
        metadata = self.metadata(checkpoint)
        version = metadata.get("id")
        has_version = type(version) is int and version > 0
        repository = metadata.get("hf_repo_id")
        sha256 = self.cached_hash(metadata)
        if not has_version and not (repository and sha256):
            return "No public version ID or verified Hugging Face identity available; using local metadata."
        key = self.identity(checkpoint, metadata)
        with self.lock:
            entry = self.load().get(key, {})
        ttl = 86400 if entry.get("status") == "Public version metadata refreshed." else 300
        checked = entry.get("checked", 0)
        if not isinstance(checked, (float, int)) or isinstance(checked, bool) or not 0 <= checked <= time.time() + 86400:
            checked = 0
        if not force and entry.get("schema") == RECOMMENDATION_SCHEMA and time.time() - checked < ttl:
            return entry.get("status", "Cached recommendation")
        from modules.model_sources import civitai_metadata
        remote = civitai_metadata(f"model-versions/{version}") if has_version else None
        matched = isinstance(remote, dict) and type(remote.get("id")) is int and remote["id"] == version
        recommended = extract(remote, samplers, schedulers) if matched else {}
        source = f"Civitai version {version}" if matched else None
        source_url = None
        if matched and type(remote.get("modelId")) is int and remote["modelId"] > 0:
            source_url = f"https://civitai.com/models/{remote['modelId']}?modelVersionId={version}"
            if not recommended:
                model = civitai_metadata(f"models/{remote['modelId']}")
                versions = model.get("modelVersions", []) if isinstance(model, dict) else []
                versions = [item for item in versions if isinstance(item, dict)] if isinstance(versions, list) else []
                families = {item.get("baseModel") for item in versions if isinstance(item.get("baseModel"), str)}
                if (isinstance(model, dict) and type(model.get("id")) is int and model["id"] == remote['modelId']
                        and any(item.get("id") == version for item in versions)
                        and isinstance(remote.get("baseModel"), str)
                        and families == {remote['baseModel']}):
                    general = extract(model, samplers, schedulers)
                    if general:
                        recommended = general
                        source = "Civitai creator model card (shared across versions)"
        if sha256 and not repository and not recommended:
            from modules.model_sources import huggingface_preview
            discovery = huggingface_preview(checkpoint, sha256)
            if isinstance(discovery, dict):
                repository = discovery.get("hf_repo_id")
        if repository and sha256 and not recommended:
            from modules.model_sources import huggingface_recommendations
            hf = huggingface_recommendations(repository, sha256)
            if isinstance(hf, dict):
                # Repackaged weights may link to the original creator's card.
                # Follow a bounded set of HF repositories only when each contains
                # exactly the same weights; never trust a link as model identity.
                if not extract(hf, samplers, schedulers):
                    linked = re.findall(r"https://huggingface\.co/([A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*)",
                                        str(hf.get("description") or "")[:200000])
                    parents = hf.get("source_repositories", [])
                    if isinstance(parents, list):
                        linked = [name for name in parents if isinstance(name, str)] + linked
                    candidates = [name.rstrip(".") for name in dict.fromkeys(linked)
                                  if name != repository and name.split("/")[0] not in
                                  ("docs", "spaces", "datasets", "blog", "collections")][:3]
                    for candidate in candidates:
                        creator = huggingface_recommendations(candidate, sha256)
                        if isinstance(creator, dict) and extract(creator, samplers, schedulers):
                            hf = creator
                            break
                remote = hf
                recommended = extract(hf, samplers, schedulers)
                source, source_url = hf.get("source"), hf.get("source_url")
                matched = True
        status = "Public version metadata refreshed." if matched else "Lookup unavailable; retained cached settings."
        with self.lock:
            data = self.load()
            entry = data.setdefault(key, data.pop(f"{checkpoint}:local", {}))
            entry.update(checked=time.time(), status=status, schema=RECOMMENDATION_SCHEMA)
            if matched:
                entry.update(recommended=recommended, source=source, source_url=source_url)
            save_json(self.path, data)
        return status

    def recipe(self, checkpoint, samplers, schedulers):
        details = self.details(checkpoint, samplers, schedulers)
        values = details['recommended'] | details['override']
        status = details['source']
        if not details['recommended']:
            status += ": no sampling settings published."
        if details['override']:
            status += " Saved local settings are active."
        return values, status

    def details(self, checkpoint, samplers, schedulers):
        metadata = self.metadata(checkpoint)
        with self.lock:
            data = self.load()
            entry = data.get(self.identity(checkpoint, metadata), data.get(f"{checkpoint}:local", {}))
        local = extract(metadata, samplers, schedulers)
        fetched = validated(entry.get("recommended", {}), samplers, schedulers)
        recommended = fetched if "recommended" in entry else local
        override = validated(entry.get("override", {}), samplers, schedulers)
        source = entry.get("source") or (f"Civitai version {metadata['id']}" if isinstance(metadata.get("id"), int) else "local metadata")
        url = entry.get("source_url")
        if not url and type(metadata.get("modelId")) is int and type(metadata.get("id")) is int:
            url = f"https://civitai.com/models/{metadata['modelId']}?modelVersionId={metadata['id']}"
        return dict(recommended=recommended, fetched=fetched, override=override,
                    source=source, source_url=url, network_cached="recommended" in entry,
                    status=entry.get("status", ""))

    def describe(self, checkpoint, options, samplers, schedulers, family_default=True):
        details = self.details(checkpoint, samplers, schedulers)
        source = details['source']
        url = details['source_url']
        if isinstance(url, str):
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if parsed.scheme == 'https' and parsed.hostname in ('civitai.com', 'huggingface.co'):
                source = f"[{source}]({url})"
        lines = [source]
        if details['override']:
            lines += ["**Your override is active.** Remove saved settings to use network recommendations."]
        if details['network_cached']:
            if details['fetched']:
                labels = dict(custom_steps='Steps', cfg='CFG', sampler_name='Sampler', scheduler='Scheduler', clip_skip='Clip Skip')
                fetched = ' · '.join(f"{labels[key]}: {value}" for key, value in details['fetched'].items())
                lines += [f"**Network settings:** {fetched}"]
            else:
                lines += ["**Network settings:** none published in the fetched metadata."]
        else:
            lines += ["**Network settings:** not fetched; using cached model metadata."]
        if self.offline:
            lines += ["Offline mode."]
        elif details['status'].startswith('Lookup unavailable'):
            lines += ["Refresh failed; previous settings retained."]
        table = ['| Setting | Model default | Source |', '|---|---|---|']
        for key, label in (('custom_steps','Steps'), ('cfg','CFG'), ('sampler_name','Sampler'), ('scheduler','Scheduler'), ('clip_skip','Clip Skip')):
            origin = ('Your override' if key in details['override'] else
                      'Network' if key in details['fetched'] else
                      'Cached metadata' if key in details['recommended'] else
                      'Family preset' if family_default else 'App fallback')
            table.append(f"| {label} | {options[key]} | {origin} |")
        return '\n\n'.join(lines) + '\n\n' + '\n'.join(table)

    def save_override(self, checkpoint, values, samplers, schedulers):
        clean = validated(values, samplers, schedulers)
        if set(clean) != set(FIELDS):
            raise ValueError("Override requires valid steps, CFG, sampler, scheduler and Clip Skip.")
        with self.lock:
            data = self.load()
            key = self.identity(checkpoint)
            if key not in data:
                data[key] = data.pop(f"{checkpoint}:local", {})
            data[key]["override"] = clean
            save_json(self.path, data)

    def reset_override(self, checkpoint):
        with self.lock:
            data = self.load()
            for key in (self.identity(checkpoint), f"{checkpoint}:local"):
                data.get(key, {}).pop("override", None)
            save_json(self.path, data)
