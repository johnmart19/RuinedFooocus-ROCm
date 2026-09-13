"""Private chat artifacts, separate from models and the image-browser database."""

from pathlib import Path


def chat_root():
    return Path(__file__).resolve().parents[1] / "cache" / "embeds"


def chat_folder(name):
    if name not in ("logs", "sources", "images"):
        raise ValueError("Unknown chat storage category")
    folder = chat_root() / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def migrate_legacy_chat_files():
    """Move only known chat caches; leave model/runtime downloads untouched."""
    import uuid
    from shared import path_manager
    root = Path(__file__).resolve().parents[1]
    groups = [(root / "cache" / "llama.cpp", "server-*.log", "logs"),
              (root / "cache" / "embeds", "*", "sources")]
    previous = Path(path_manager.model_paths["temp_outputs_path"]) / "chats"
    groups.extend((previous / category, "**/*", category)
                  for category in ("logs", "sources", "images"))
    for source, pattern, category in groups:
        if source.is_symlink() or not source.is_dir():
            continue
        for file in source.glob(pattern):
            if (not file.is_file() or file.is_symlink() or file.name.startswith(".")
                    or file.name == "files_for_embeddings_will_be_cached_here"):
                continue
            destination = chat_folder(category) / file.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                destination = destination.with_name(uuid.uuid4().hex + "-" + file.name)
            try:
                file.rename(destination)
            except OSError:
                print("Chat cache migration deferred for a locked file; retry after closing the old runtime.")


def clear_chat_cache():
    """Called on the serial worker after opting into shared cache deletion."""
    import shared
    from modules.llama_vision import _generated_images
    root = chat_root()
    if root.is_symlink() or root.resolve() != root.absolute():
        raise ValueError("Refusing to clear a redirected chat cache directory.")
    pipeline = shared.state.get("pipeline")
    if pipeline is not None and hasattr(pipeline, "llm"):
        pipeline.unload()
    _generated_images.clear()
    migrate_legacy_chat_files()
    removed = 0
    failed = 0
    # Resolve each individual file before deleting; do not traverse linked folders.
    import os
    for directory, folders, files in os.walk(root, followlinks=False):
        folders[:] = [name for name in folders if
            (Path(directory) / name).resolve().is_relative_to(root.resolve())
            and (Path(directory) / name).resolve() == (Path(directory) / name).absolute()
            and not (Path(directory) / name).is_symlink()]
        for name in files:
            if name == "files_for_embeddings_will_be_cached_here":
                continue
            file = Path(directory) / name
            if file.is_symlink() or not file.resolve().is_relative_to(root.resolve()):
                continue
            try:
                file.unlink()
                removed += 1
            except OSError:
                failed += 1
    return removed, failed
