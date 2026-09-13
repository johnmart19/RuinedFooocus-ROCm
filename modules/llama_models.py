"""Chat model choices and local-first model resolution."""

from pathlib import Path
import re
import shutil
from shared import path_manager, settings

DEFAULT_MODEL = "Qwen2.5-7B-Instruct-abliterated-v2.Q4_K_M.gguf"
GPT_OSS_MODEL = "gpt-oss-20b-MXFP4.gguf"


def local_models():
    folders = path_manager.paths["path_llm"]
    if isinstance(folders, str):
        folders = [folders]
    roots = [path_manager.get_abspath(folder) for folder in folders]
    files = {}
    for root in roots:
        for path in sorted(root.rglob("*.gguf")):
            if path.is_file() and not path.name.lower().startswith("mmproj"):
                files.setdefault(path.name, str(path.resolve()))
    hidden = set(settings.default_settings.get("llama_hidden_models", []))
    return {name: filename for name, filename in files.items() if str(Path(filename).resolve()) not in hidden}


def import_model(filename):
    """Copy a selected GGUF into the app's model folder, keeping the original."""
    path = Path(filename).resolve()
    if not path.is_file():
        raise ValueError("The selected model file is missing.")
    existing = local_models().get(path.name)
    if existing and Path(existing).resolve() != path:
        raise ValueError("A different model has this filename. Rename one file before importing.")
    if not existing:
        folders = path_manager.paths["path_llm"]
        folder = folders[0] if isinstance(folders, list) else folders
        destination = path_manager.get_abspath(folder).resolve() / path.name
        if destination != path:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise ValueError("A model with this filename already exists. Rename one file before importing.")
            temporary = destination.with_suffix(".gguf.importing")
            target = temporary.open("xb")
            try:
                with target, path.open("rb") as source:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                temporary.rename(destination)
            except OSError:
                temporary.unlink(missing_ok=True)
                raise
            path = destination
        hidden = settings.default_settings.get("llama_hidden_models", [])
        settings.default_settings["llama_hidden_models"] = [item for item in hidden if item != str(path)]
        try:
            settings.save_settings()
        except OSError:
            settings.default_settings["llama_hidden_models"] = hidden
            raise
    for model, choices in model_catalogue().items():
        for _, value in choices:
            if value in (path.name, str(path)):
                return model, value
    raise ValueError("Could not add this model to the list.")


def remove_local_model(filename, delete_file=False):
    path = Path(filename).resolve()
    if path.suffix.lower() != ".gguf" or str(path) not in {str(Path(item).resolve()) for item in local_models().values()}:
        raise ValueError("This model is no longer in the local list. Refresh the list first.")
    if delete_file:
        try:
            path.unlink()
        except PermissionError as error:
            raise ValueError("Cannot delete this file. Load another model or close apps using it, then try again.") from error
    hidden = settings.default_settings.get("llama_hidden_models", [])
    settings.default_settings["llama_hidden_models"] = list(dict.fromkeys([*hidden, str(path)])) if not delete_file else hidden
    try:
        settings.save_settings()
    except OSError as error:
        if delete_file:
            raise ValueError("The file was deleted, but the updated model list could not be saved.") from error
        raise


def model_catalogue(source=None):
    local = local_models()
    downloadable = path_manager.get_folder_list("llm")
    names = list(dict.fromkeys([DEFAULT_MODEL, GPT_OSS_MODEL]
        + downloadable + list(local)))
    catalogue = {}
    hidden_names = {Path(item).name for item in settings.default_settings.get("llama_hidden_models", [])}
    for name in names:
        if Path(name).name.lower().startswith("mmproj"):
            continue
        if name in hidden_names and name not in local:
            continue
        if source == "Local models" and name not in local:
            continue
        if source == "Downloadable" and name in local:
            continue
        match = re.fullmatch(r"(.+?)[.-]((?:I?Q\d|MXFP\d|BF16|F16|F32)[A-Za-z0-9_]*)",
                             Path(name).stem, flags=re.IGNORECASE)
        model, quant = match.groups() if match else (Path(name).stem, "Original")
        value = name if name in downloadable else local.get(name, name)
        status = "Local" if name in local else "Download"
        catalogue.setdefault(model, []).append((f"{quant.upper()} · {status}", value))
    return catalogue


def model_label(name):
    if name == "Qwen2.5-7B-Instruct-abliterated-v2":
        return "Qwen 2.5 7B (Default)"
    if name == "gpt-oss-20b":
        return "GPT-OSS 20B"
    return name.replace("-", " ").replace("_", " ")


def preferred_quant(choices):
    return next((value for label, value in choices if label.startswith("Q4_K_M ")),
                choices[0][1])


def resolve_model(name, progress=None):
    if Path(name).is_file():
        return Path(name)
    local = local_models().get(name)
    if local:
        return Path(local)
    if "vision/" + name in path_manager.DOWNLOADABLE_FILES:
        return path_manager.get_folder_file_path("vision", name, progress=progress)
    folders = path_manager.paths["path_llm"]
    folder = folders[0] if isinstance(folders, list) else folders
    return path_manager.get_folder_file_path("llm", name,
        default=path_manager.get_abspath(folder) / name, progress=progress)
