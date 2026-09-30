"""Local checkpoint actions for PowerUp."""

import filecmp
from pathlib import Path

import gradio as gr
import shared
from safetensors import SafetensorError
from modules import checkpoint_tools as tools


def _selected(name):
    if name not in shared.models.get_names("checkpoints"):
        raise ValueError("Select an installed checkpoint first.")
    filename = shared.models.get_file_from_name("checkpoints", name)
    if filename is None:
        raise ValueError("The selected checkpoint is no longer available.")
    path = Path(filename)
    info = shared.models.get_models_by_path("checkpoints", name, fetch=False)
    return path, info


def _preview(path):
    cached = shared.models.cache_paths["checkpoints"] / path.name
    for suffix in (".jpg", ".jpeg", ".png", ".gif"):
        for candidate in (cached.with_suffix(suffix), path.with_suffix(".preview" + suffix),
                          path.with_suffix(suffix)):
            if candidate.is_file() and not filecmp.cmp(candidate, "html/warning.jpeg", shallow=False):
                return candidate


def run_action(name, action, progress):
    try:
        path, info = _selected(name)
        if action == "convert":
            result = tools.create_fp16_copy(path, shared.models.cache_paths["checkpoints"],
                                            info, _preview(path), progress)
            if result["path"]:
                destination = result["path"]
                new_name = str(Path(name).with_name(destination.name))
                names = shared.models.names["checkpoints"]
                if new_name not in names:
                    names.append(new_name)
                    names.sort(key=str.casefold)
                shared.models.revision += 1
                try:
                    cached = shared.models.cache_paths["checkpoints"] / destination.name
                    if not result["preview"]:
                        shared.models.copy_embedded_preview(destination, cached)
                except OSError as error:
                    result["message"] += f" Preview cache could not be updated: {error}"
            return result["message"]
        report = tools.inspect_checkpoint(path)
        checked = tools.check_weights(path, progress) if action == "check" else None
        return tools.format_report(report, shared.models.get_model_base(info), checked)
    except (OSError, ValueError, SafetensorError) as error:
        return f"Checkpoint tool: {error}"


def create_ui(base_model):
    with gr.Column(visible=False, elem_id="checkpoint-tools") as group:
        names = shared.models.get_names("checkpoints")
        selected = gr.Dropdown(label="Available checkpoints", choices=names,
                               value=base_model.value if base_model.value in names else None)
        selected.focus(lambda: gr.update(choices=shared.models.get_names("checkpoints")),
                       outputs=selected, api_visibility="undocumented")
        base_model.change(lambda name: gr.update(choices=shared.models.get_names("checkpoints"), value=name),
                          inputs=base_model, outputs=selected,
                          api_visibility="undocumented")
        gr.Markdown("FP16 copies keep the original. Conversion is lossy; FP8/BF16 stay unchanged.")
        with gr.Row(equal_height=True):
            inspect = gr.Button("Inspect", min_width=100)
            check = gr.Button("Check weights", min_width=100)
            convert = gr.Button("Create FP16 copy", min_width=140)
        report = gr.Textbox(label="Checkpoint report", lines=7, interactive=False)

        def inspect_selected(name, progress=gr.Progress()):
            return run_action(name, "inspect", progress)

        def check_selected(name, progress=gr.Progress()):
            return run_action(name, "check", progress)

        def convert_selected(name, progress=gr.Progress()):
            return run_action(name, "convert", progress)

        for button, callback in ((inspect, inspect_selected), (check, check_selected),
                                 (convert, convert_selected)):
            button.click(callback, inputs=selected, outputs=report,
                         concurrency_id="checkpoint_tools", concurrency_limit=1,
                         api_visibility="undocumented")
        selected.change(lambda: "", outputs=report, api_visibility="undocumented")
    return group
