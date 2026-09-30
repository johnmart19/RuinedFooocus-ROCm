"""Write configuration without truncating the last working file on failure."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def save_json(path, data):
    path = Path(path)
    text = json.dumps(data, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
