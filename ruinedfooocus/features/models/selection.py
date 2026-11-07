"""Model display names are presentation; filenames remain the selection identity."""

from collections import Counter
from pathlib import Path
import re

from ruinedfooocus.features.video.video_settings import VIDEO_FPS


def model_label(name, metadata=None):
    metadata = metadata or {}
    model = metadata.get("model")
    title = model.get("name") if isinstance(model, dict) else None
    title = metadata.get("rf_display_name") or title or metadata.get("hf_repo_id")
    if not isinstance(title, str) or not title.strip():
        return Path(name).stem.replace("_", " ")
    title = " ".join(title.strip(" ✨★☆⭐").split())
    # Publisher titles often list every merge ancestor after the primary name.
    short = re.sub(r"\s+trained$", "", title.split(" + ", 1)[0], flags=re.IGNORECASE)
    short = short if len(short) <= 80 else short[:77] + "…"
    version = metadata.get("name")
    if isinstance(version, str) and version.strip() and version.casefold() not in short.casefold():
        version = " ".join(version.replace("_", " ").split())
        separator = " " if version.lower().startswith("v") else " · "
        short += separator + version
    return short


def checkpoint_records(names, metadata=None, query=""):
    records = []
    for name in names:
        info = metadata(name) if metadata else {}
        if info.get("baseModel") in VIDEO_FPS:
            continue
        records.append({"name": name, "label": model_label(name, info)})
    counts = Counter(record["label"] for record in records)
    for record in records:
        if counts[record["label"]] > 1:
            record["label"] += f" · {record['name']}"
        record["caption"] = f"{record['label']}\n{record['name']}"
    search = (query or "").casefold().strip()
    return [r for r in records if search in (r["label"] + " " + r["name"]).casefold()]


def checkpoint_choices(names, selected=None, metadata=None):
    names = list(names)
    choices = [(r["label"], r["name"]) for r in checkpoint_records(names, metadata)]
    if selected and selected not in [value for _, value in choices]:
        status = "outside image filter" if selected in names else "not installed"
        choices.append((f"{model_label(selected)} ({status})", selected))
    return choices


def selected_checkpoint(records, index, caption):
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(records):
        raise ValueError("The model list changed. Select the checkpoint again.")
    selected = records[index]
    if selected["caption"] != caption:
        raise ValueError("The model list changed. Select the checkpoint again.")
    return selected["name"]
