"""Bundled and user assistants; model selection is independent of persona."""

import base64
import json
from pathlib import Path

from PIL import Image

from ruinedfooocus.core.paths import APP_ROOT, asset_path, content_roots, data_path, resource_path


def character_card(path):
    with Image.open(path) as image:
        encoded = image.info.get("chara")
    if not encoded:
        return None
    card = json.loads(base64.b64decode(encoded))
    data = card.get("data", card)
    if not all(key in data for key in ("name", "first_mes", "personality", "scenario")):
        raise ValueError("Unsupported character card")
    return {
        "name": data["name"],
        "greeting": data["first_mes"],
        "avatar": path,
        "system": f"Your name is {data['name']}.\nYou are: {data['personality']}\nScenario: {data['scenario']}",
        "embed": [["text", f"Summary: {data.get('description', data.get('summary', ''))}"]],
    }


def assistant_choices():
    choices = {}
    for folder in content_roots("assistants", "chatbots"):
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            try:
                if path.name == "info.json":
                    info = json.loads(path.read_text(encoding="utf-8"))
                    choices[path.parent.relative_to(folder).as_posix()] = (
                        info["name"],
                        str(path.parent),
                    )
                elif path.suffix.lower() == ".png" and not path.with_name("info.json").exists():
                    info = character_card(path)
                    if info:
                        choices[path.relative_to(folder).as_posix()] = (info["name"], str(path))
            except (OSError, ValueError, KeyError) as error:
                print(f"Cannot read assistant {path}: {error}")
    return [("General chat", "__normal__")] + sorted(
        choices.values(), key=lambda item: item[0].casefold()
    )


def assistant_info(selected):
    if selected == "__normal__":
        return {
            "name": "General chat",
            "avatar": asset_path("chat_normal.png"),
            "system": "",
            "embed": "[]",
            "chatstart": [],
        }
    path = Path(selected)
    try:
        if path.is_dir():
            info = json.loads((path / "info.json").read_text(encoding="utf-8"))
            info.setdefault("avatar", path / "avatar.png")
        else:
            info = character_card(path)
        if not info:
            raise ValueError("No character definition")
        info["embed"] = json.dumps(info.get("embed", []))
        info["chatstart"] = [{"role": "assistant", "content": info["greeting"]}]
        return info
    except (OSError, ValueError, KeyError) as error:
        print(f"Cannot load assistant {selected}: {error}")
        return {
            "name": "Unavailable assistant",
            "avatar": asset_path("error.png"),
            "system": "",
            "embed": "[]",
            "chatstart": [],
        }


def context_file(value):
    """Resolve local context without allowing files outside application/data roots."""
    candidate = Path(value)
    roots = (data_path(), APP_ROOT)
    candidates = (
        [base / candidate for base in roots] if not candidate.is_absolute() else [candidate]
    )
    if not candidate.is_absolute() and candidate.parts and candidate.parts[0] == "chatbots":
        candidates.append(resource_path("assistants", *candidate.parts[1:]))
    for path in candidates:
        path = path.resolve()
        if any(base.resolve() in path.parents for base in roots) and path.is_file():
            return path
    raise ValueError(f"Chat context file not found inside the application/data folder: {value}")
