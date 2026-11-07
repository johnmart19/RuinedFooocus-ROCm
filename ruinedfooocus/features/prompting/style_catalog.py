"""Read-only style discovery shared by prompt composition and the UI."""

import csv
from dataclasses import dataclass, replace
from functools import lru_cache
import locale
from pathlib import Path

from ruinedfooocus.core.paths import data_path, resource_path

STYLE_ALIASES = {"Erniehancer": "Local chat model", "Style: Erniehancer": "Local chat model"}
PROCESSORS = {
    "None": ("{prompt}", ""),
    "Style: Pick Random": ("{prompt}", ""),
    "LoRA keywords": ("{prompt} {lora_keywords}", ""),
    "Flufferizer": ("{prompt}", ""),
    "Hyperprompt": ("{prompt}", ""),
    "Local chat model": ("{prompt}", ""),
}
GROUP_NAMES = {
    "SDXL COMFYUI PORT": "General styles",
    "ADS": "Advertising",
    "ARTSTYLE": "Art styles",
    "FUTURISTIC": "Futuristic",
    "GAME": "Games",
    "MISC": "Miscellaneous",
    "PAPERCRAFT": "Papercraft",
    "PHOTO": "Photography",
    "ARTIFY": "Artify",
}


@dataclass(frozen=True)
class Style:
    name: str
    prompt: str
    negative_prompt: str
    group: str


def is_heading(name):
    return isinstance(name, str) and name.lstrip().startswith(">>>")


def normalize_selection(selected):
    """Legacy headers are inert, including selections restored from old PNGs."""
    if isinstance(selected, str):
        selected = [selected]
    return list(
        dict.fromkeys(
            STYLE_ALIASES.get(name, name)
            for name in selected or ()
            if isinstance(name, str) and name != "None" and not is_heading(name)
        )
    )


@lru_cache(maxsize=16)
def _read_styles(path, signature, encoding):
    # Signature is part of the key so edits refresh without parsing on generation.
    if signature is None:
        return ()
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding=encoding)
    group = "Custom"
    records = []
    for row in csv.DictReader(text.splitlines(keepends=True)):
        name = row.get("name")
        if not name:
            continue
        if is_heading(name):
            heading = name.lstrip("> ").strip()
            group = GROUP_NAMES.get(heading.upper(), heading.title())
        elif name not in STYLE_ALIASES and name not in PROCESSORS:
            records.append(
                Style(name, row.get("prompt") or "", row.get("negative_prompt") or "", group)
            )
    return tuple(records)


def load_catalog(default_file=None, user_file=None):
    """User rows override bundled rows by name; neither file is ever rewritten."""
    default_file = (
        Path(default_file) if default_file else resource_path("defaults", "styles.default")
    )
    user_file = Path(user_file) if user_file else data_path("settings", "styles.csv")
    catalog = {}
    for path in (default_file, user_file):
        try:
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
        except FileNotFoundError:
            signature = None
        for style in _read_styles(path, signature, locale.getpreferredencoding(False)):
            if style.name in catalog:
                style = replace(style, group=catalog[style.name].group)
            catalog[style.name] = style
    return catalog


def style_templates(default_file=None, user_file=None):
    return PROCESSORS | {
        name: (style.prompt, style.negative_prompt)
        for name, style in load_catalog(default_file, user_file).items()
    }
