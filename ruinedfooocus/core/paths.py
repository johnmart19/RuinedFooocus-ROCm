"""Application resources and writable data, independent of the launch directory.

The default data root preserves existing clone installations. RF_DATA_DIR allows
an archive/read-only application to keep writable state somewhere else. Legacy
model path values are relative to the former modules/ directory by design.
"""

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = APP_ROOT / "ruinedfooocus"
RESOURCE_ROOT = PACKAGE_ROOT / "resources"
ASSET_ROOT = PACKAGE_ROOT / "ui" / "assets"


def data_path(*parts):
    configured = os.environ.get("RF_DATA_DIR")
    base = Path(configured).expanduser() if configured else APP_ROOT
    if not base.is_absolute():
        base = APP_ROOT / base
    return base.joinpath(*parts).resolve()


def resource_path(*parts):
    return RESOURCE_ROOT.joinpath(*parts)


def asset_path(name):
    return ASSET_ROOT / name


def legacy_model_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (data_path() / "modules" / path).resolve()


def content_roots(kind, legacy):
    """User entries override bundled entries with the same relative identity.

    New user locations take precedence over the old folder; neither is moved or
    modified by discovery. The bundled folder is always read-only to the loader.
    """
    return (resource_path(kind), data_path(legacy), data_path(kind))
