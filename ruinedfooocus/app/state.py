from ruinedfooocus.core.translation_manager import TranslationManager
from ruinedfooocus.core.settings import SettingsManager
from ruinedfooocus.features.models.performance import PerformanceSettings
from ruinedfooocus.features.models.resolutions import ResolutionSettings
from ruinedfooocus.core.path import PathManager
from ruinedfooocus.app.arguments import args
import time
import os

gradio_root = None
server_app = None
local_url = "http://127.0.0.1:7860/"
share_url = "http://127.0.0.1:7860/"

translation_manager = TranslationManager()
translation_manager.set_language(args.language)
translate = translation_manager.translate

state = {
    "preview_image": None,
    "ctrls_name": [],
    "ctrls_obj": [],
    "setting_name": [],
    "setting_obj": [],
    "cfg_items_obj": [],
    "pipeline": None,
    "last_config": 0.0,
}

wildcards = None

settings = SettingsManager()
path_manager = PathManager()
models_offline = (
    args.offline
    or os.environ.get("RF_OFFLINE") == "1"
    or settings.default_settings.get("local_model_metadata", False)
)
performance_settings = PerformanceSettings(
    cache_dir=path_manager.model_paths["cache_path"] / "checkpoints", offline=models_offline
)
resolution_settings = ResolutionSettings()
shared_cache = {}
from ruinedfooocus.features.models.model_handler import Models

models = Models(offline=models_offline)


# Call this to trigger a refresh of ui components
def update_cfg():
    state["last_config"] = str(time.time())


def add_ctrl(name, obj, configurable=False):
    state["ctrls_name"] += [name]
    state["ctrls_obj"] += [obj]
    if configurable:
        state["cfg_items_obj"] += [obj]


def get_ctrl(name):
    return state["ctrls_obj"][state["ctrls_name"].index(name)]


def add_setting(name, obj):
    state["setting_name"] += [name]
    state["setting_obj"] += [obj]
