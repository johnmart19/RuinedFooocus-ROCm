"""Locate optional Fooocus expansion weights without downloading at startup."""

from pathlib import Path
import shutil

from ruinedfooocus.core.paths import APP_ROOT, data_path

WEIGHTS = "pytorch_model.bin"
URL = "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin"


def expansion_model_path():
    writable = data_path("prompt_expansion")
    legacy = APP_ROOT / "prompt_expansion"
    return (
        legacy if not (writable / WEIGHTS).is_file() and (legacy / WEIGHTS).is_file() else writable
    )


def ensure_expansion_resources(folder):
    """Called only when expansion is selected; existing weights/configs win."""
    folder = Path(folder)
    if not (folder / WEIGHTS).is_file():
        from ruinedfooocus.core.util import load_file_from_url

        load_file_from_url(url=URL, model_dir=str(folder), file_name=WEIGHTS)
    folder.mkdir(parents=True, exist_ok=True)
    for source in (APP_ROOT / "prompt_expansion").iterdir():
        if source.suffix not in (".json", ".txt") or not source.is_file():
            continue
        destination = folder / source.name
        if destination.exists():
            continue
        try:
            target = destination.open("xb")
        except FileExistsError:
            continue
        try:
            with target, source.open("rb") as incoming:
                shutil.copyfileobj(incoming, target)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    return str(folder)
