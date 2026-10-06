import gradio as gr
from pathlib import Path
import json
from modules.resource_defaults import RESOURCE_DEFAULTS, RESOURCE_LABELS
from modules.settings_validation import validate_settings
from modules.sdxl_styles import load_styles
from modules.interrogate import looks

from shared import state, add_setting, performance_settings, resolution_settings, path_manager, settings, models, translate

t = translate


def obp_choices():
    names = set()
    for filename in ("random_prompt/presets/obp_presets.default", "random_prompt/userfiles/obp_presets.json"):
        path = Path(filename)
        if path.exists():
            names.update(json.loads(path.read_text(encoding="utf-8")))
    return sorted(names, key=str.casefold)


def model_choices(model_type, key, default):
    choices = list(models.names[model_type])
    if model_type == "loras":
        choices.insert(0, "None")
    selected = settings.default_settings.get(key, default)
    if selected and selected not in choices:
        choices.append((f"{selected} (not found)", selected))
    return choices


def resource_choices(key, all_files=False, selected=None):
    default = RESOURCE_DEFAULTS[key]
    choices = [(f"Automatic - {default}", "")]
    folder = "clip_vision" if key == "clip_vision" else ("clip" if key.startswith("clip_") else "vae")
    if all_files:
        choices.extend(path_manager.get_folder_list(folder))
    if selected and selected not in choices:
        choices.append(selected)
    return choices

def save_clicked(*args):
    try:
        args = validate_settings(dict(zip(state["setting_name"], args)))
    except ValueError as error:
        raise gr.Error(str(error)) from error
    ui_data = {}
    # Overwrite current settings
    for key, val in args.items():
        settings.default_settings[key] = val

        # Massage some of the data. Settings that are lists should be split
        if key in [
            "archive_folders",
            "path_checkpoints",
            "path_loras",
            "path_wildcards",
        ]:
            settings.default_settings[key] = (val or "").splitlines() if isinstance(val, str) or val is None else val

        # Remove empty keys
        if settings.default_settings[key] == None or settings.default_settings[key] == "":
            settings.default_settings.pop(key)
            continue

        # Move ui_* and path_*
        if key.startswith("ui_"):
            ui_data[key] = settings.default_settings.get(key)
            settings.default_settings.pop(key)
        if key.startswith("path_"):
            path_manager.paths[key] = settings.default_settings.get(key)
            settings.default_settings.pop(key)

    settings.set_settings_path(ui_data.get("ui_settings_name", None))
    settings.save_settings()
    from argparser import args as launch_args
    import os
    models.offline = (launch_args.offline or os.environ.get("RF_OFFLINE") == "1"
                      or settings.default_settings.get("local_model_metadata", False))
    path_manager.set_settings_path(ui_data.get("ui_settings_name", None))
    path_manager.save_paths()

    settings.name = ui_data.get("ui_settings_name") or None
    print(t("Saved new settings to {path}.", mapping={"path": settings.settings_path}))
    gr.Info(t("Saved new settings to {path}.", mapping={"path": settings.settings_path})
            + " Restart to apply startup defaults, component overrides, folders and theme.")

def create_settings():
    with gr.Blocks() as app_settings:
        gr.Markdown("Settings saved here are startup defaults. Use Main for the current generation. "
                    "Save, then restart to apply all changes.")
        with gr.Row(elem_id="settings-layout"):
            with gr.Column():
                with gr.Accordion(t("UI settings"), open=True):
                    local_metadata = gr.Checkbox(label="Local model metadata only",
                        value=settings.default_settings.get("local_model_metadata", False),
                        info="Skip online model and artwork lookups.")
                    add_setting("local_model_metadata", local_metadata)
                    with gr.Row():
                        image_number = gr.Number(label=t("Image Number"), interactive=True, minimum=1, step=1, precision=0, value=settings.default_settings.get("image_number", 1))
                        add_setting("image_number", image_number)
                        image_number_max = gr.Number(label=t("Image Number Max"), interactive=True, minimum=1, step=1, precision=0, value=settings.default_settings.get("image_number_max", 50))
                        add_setting("image_number_max", image_number_max)
                    with gr.Row():
                        seed = gr.Number(label=t("Seed"), interactive=True, minimum=-1, precision=0, value=settings.default_settings.get("seed", 0))
                        add_setting("seed", seed)
                        seed_random = gr.Checkbox(label=t("Random Seed"), interactive=True, value=settings.default_settings.get("seed_random", True))
                        add_setting("seed_random", seed_random)
                    style = gr.Dropdown(
                        label=t("Style Selection"),
                        multiselect=True,
                        container=True,
                        choices=list(load_styles().keys()),
                        value=[name for name in settings.default_settings.get("style", [])
                               if name in load_styles()],
                    )
                    add_setting("style", style)
                    prompt = gr.Textbox(label=t("Prompt"), interactive=True, value=settings.default_settings.get("prompt", ""))
                    add_setting("prompt", prompt)
                    negative_prompt = gr.Textbox(label=t("Negative Prompt"), interactive=True, value=settings.default_settings.get("negative_prompt", ""))
                    add_setting("negative_prompt", negative_prompt)
                    auto_negative_prompt = gr.Checkbox(label=t("Auto Negative Prompt"), interactive=True, value=settings.default_settings.get("auto_negative_prompt", False))
                    add_setting("auto_negative_prompt", auto_negative_prompt)
                    performance_choices = [performance_settings.MODEL_PERFORMANCE, performance_settings.CUSTOM_PERFORMANCE]
                    saved_performance = settings.default_settings.get("performance", "SDXL")
                    if saved_performance not in performance_choices:
                        saved_performance = performance_settings.MODEL_PERFORMANCE
                    performance = gr.Dropdown(
                        label=t("Performance"),
                        interactive=True,
                        choices=performance_choices,
                        value=saved_performance,
                    )
                    add_setting("performance", performance)
                    resolution = gr.Dropdown(
                        label=t("Resolution"),
                        interactive=True,
                        choices=list(resolution_settings.aspect_ratios.keys()),
                        value=settings.default_settings.get("resolution", "1152x896 (4:3)"),
                    )
                    add_setting("resolution", resolution)

                    with gr.Row():
                        lora_min = gr.Number(label=t("LoRA weight min"), interactive=True, value=settings.default_settings.get("lora_min", 0))
                        add_setting("lora_min", lora_min)
                        lora_max = gr.Number(label=t("LoRA weight max"), interactive=True, value=settings.default_settings.get("lora_max", 2))
                        add_setting("lora_max", lora_max)

                with gr.Accordion(t("Preset"), open=False):
                    preset_mode = gr.Dropdown(
                        label=t("Preset Mode"),
                        interactive=True,
                        choices=[
                            ("Full", 7),
                            ("Models and Performance", 6),
                            ("Models and Size", 5),
                            ("Performance and Size", 3),
                            ("Models only", 4),
                            ("Performance only", 2),
                            ("Size only", 1),
                        ],
                        value=settings.default_settings.get("preset_mode", 7)
                    )
                    add_setting("preset_mode", preset_mode)

                with gr.Accordion(t("Models to load at startup"), open=False):
                    base_model = gr.Dropdown(
                        label=t("Base Model"),
                        interactive=True,
                        choices=model_choices("checkpoints", "base_model", "sd_xl_base_1.0_0.9vae.safetensors"),
                        value=settings.default_settings.get("base_model", "sd_xl_base_1.0_0.9vae.safetensors"),
                    )
                    add_setting("base_model", base_model)
                    with gr.Row():
                        lora_1_model = gr.Dropdown(
                            label=t("LoRA {id} Model", mapping={'id': 1}),
                            interactive=True,
                            choices=model_choices("loras", "lora_1_model", "None"),
                            value=settings.default_settings.get("lora_1_model", "None"),
                        )
                        lora_1_weight = gr.Number(label=t("Lora {id} Weight", mapping={'id': 1}), value=settings.default_settings.get("lora_1_weight", 0.5), step=0.05)
                    with gr.Row():
                        lora_2_model = gr.Dropdown(
                            label=t("LoRA {id} Model", mapping={'id': 2}),
                            interactive=True,
                            choices=model_choices("loras", "lora_2_model", "None"),
                            value=settings.default_settings.get("lora_2_model", "None"),
                        )
                        lora_2_weight = gr.Number(label=t("Lora {id} Weight", mapping={'id': 2}), value=settings.default_settings.get("lora_2_weight", 0.5), step=0.05)
                    with gr.Row():
                        lora_3_model = gr.Dropdown(
                            label=t("LoRA {id} Model", mapping={'id': 3}),
                            interactive=True,
                            choices=model_choices("loras", "lora_3_model", "None"),
                            value=settings.default_settings.get("lora_3_model", "None"),
                        )
                        lora_3_weight = gr.Number(label=t("Lora {id} Weight", mapping={'id': 3}), value=settings.default_settings.get("lora_3_weight", 0.5), step=0.05)
                    with gr.Row():
                        lora_4_model = gr.Dropdown(
                            label=t("LoRA {id} Model", mapping={'id': 4}),
                            interactive=True,
                            choices=model_choices("loras", "lora_4_model", "None"),
                            value=settings.default_settings.get("lora_4_model", "None"),
                        )
                        lora_4_weight = gr.Number(label=t("Lora {id} Weight", mapping={'id': 4}), value=settings.default_settings.get("lora_4_weight", 0.5), step=0.05)
                    with gr.Row():
                        lora_5_model = gr.Dropdown(
                            label=t("LoRA {id} Model", mapping={'id': 5}),
                            interactive=True,
                            choices=model_choices("loras", "lora_5_model", "None"),
                            value=settings.default_settings.get("lora_5_model", "None"),
                        )
                        lora_5_weight = gr.Number(label=t("Lora {id} Weight", mapping={'id': 5}), value=settings.default_settings.get("lora_5_weight", 0.5), step=0.05)

                    add_setting("lora_1_model", lora_1_model)
                    add_setting("lora_2_model", lora_2_model)
                    add_setting("lora_3_model", lora_3_model)
                    add_setting("lora_4_model", lora_4_model)
                    add_setting("lora_5_model", lora_5_model)
                    add_setting("lora_1_weight", lora_1_weight)
                    add_setting("lora_2_weight", lora_2_weight)
                    add_setting("lora_3_weight", lora_3_weight)
                    add_setting("lora_4_weight", lora_4_weight)
                    add_setting("lora_5_weight", lora_5_weight)

            with gr.Column():
                with gr.Accordion(t("One Button Prompt"), open=False):
                    preset_choices = obp_choices()
                    saved_preset = settings.default_settings.get("OBP_preset", "Standard")
                    OBP_preset = gr.Dropdown(label=t("OBP Preset"), choices=preset_choices,
                        value=saved_preset if saved_preset in preset_choices else "Standard", interactive=True)
                    add_setting("OBP_preset", OBP_preset)
                    hint_chance = gr.Number(label=t("Hint Chance"), minimum=0, maximum=100, step=1, value=settings.default_settings.get("hint_chance", 25))
                    add_setting("hint_chance", hint_chance)

                with gr.Accordion(t("Image Browser"), open=False):
                    images_per_page = gr.Number(label=t("Images per page"), value=settings.default_settings.get("images_per_page", 100), minimum=1, maximum=1000, step=1)
                    add_setting("images_per_page", images_per_page)
                    archive_folders = gr.Code(
                        label=t("Archive Folders"),
                        interactive=True,
                        value="\n".join(settings.default_settings.get("archive_folders", [])),
                        lines=5,
                        max_lines=5
                    )
                    add_setting("archive_folders", archive_folders)
                with gr.Accordion(t("Paths"), open=False):
                    path_checkpoints = gr.Code(
                        label=t("Checkpoint Folders"),
                        interactive=True,
                        value="\n".join(path_manager.paths.get("path_checkpoints", [])),
                        lines=5,
                        max_lines=5
                    )
                    add_setting("path_checkpoints", path_checkpoints)
                    path_loras = gr.Code(
                        label=t("LoRA Folders"),
                        interactive=True,
                        value="\n".join(path_manager.paths.get("path_loras", [])),
                        lines=5,
                        max_lines=5
                    )
                    add_setting("path_loras", path_loras)
                    path_inbox = gr.Textbox(label=t("Inbox Folder"), interactive=True, placeholder="", value=path_manager.paths.get("path_inbox", "../models/inbox/"))
                    add_setting("path_inbox", path_inbox)
                    path_outputs = gr.Textbox(label=t("Output Folder"), interactive=True, placeholder="", value=path_manager.paths.get("path_outputs", "../outputs/"))
                    add_setting("path_outputs", path_outputs)
                    path_wildcards = gr.Code(
                        label=t("Wildcard Folders"),
                        interactive=True,
                        value="\n".join(path_manager.paths.get("path_wildcards", ["wildcards"])),
                        lines=5,
                        max_lines=5
                    )
                    add_setting("path_wildcards", path_wildcards)

                with gr.Accordion(t("Chatbot settings"), open=False):
                    from modules.llama_runtime_info import runtime_choices
                    llm_runtime = gr.Dropdown(label="Chat runtime", choices=runtime_choices(),
                                              value=settings.default_settings.get("llm_runtime", "llama.cpp"))
                    add_setting("llm_runtime", llm_runtime)
                    runtime_signature = gr.State(str(runtime_choices()))

                    def refresh_runtime_label(previous):
                        choices = runtime_choices()
                        signature = str(choices)
                        if signature == previous:
                            return gr.skip(), gr.skip()
                        return gr.update(choices=choices), signature

                    gr.Timer(2).tick(refresh_runtime_label, inputs=runtime_signature,
                        outputs=[llm_runtime, runtime_signature], queue=False, api_visibility='undocumented')
                    from modules.llama_installer import backend_choices
                    available_backends = backend_choices()
                    saved_backend = settings.default_settings.get("llama_backend", "Auto")
                    llama_backend = gr.Dropdown(label="llama.cpp backend", choices=available_backends,
                        value=saved_backend if saved_backend in available_backends else "Auto",
                        info="Save settings, then load or chat to switch. LLAMA_SERVER overrides this selection.",
                        visible=llm_runtime.value == "llama.cpp")
                    add_setting("llama_backend", llama_backend)
                    llm_runtime.change(lambda runtime: gr.update(visible=runtime == "llama.cpp"),
                        inputs=llm_runtime, outputs=llama_backend, api_visibility='undocumented')
                    from modules.llama_build import build_capabilities, build_cuda_runtime
                    can_build_cuda = bool(build_capabilities())
                    with gr.Accordion("Build local CUDA runtime", open=False, visible=can_build_cuda):
                        gr.Markdown("Builds the pinned llama.cpp for Linux/WSL and this NVIDIA GPU. "
                                    "CUDA/CMake dependencies are installed in a separate cache environment. "
                                    "Requires system build tools (Ubuntu: `sudo apt install build-essential git python3-venv`). "
                                    "Compilation may take several minutes and use several GB of disk space.")
                        build_runtime = gr.Button("Install dependencies and build", size="sm")
                        build_status = gr.Textbox(label="Build status", lines=6, max_lines=12,
                                                  interactive=False, visible=False)

                        def build_native_runtime():
                            yield gr.update(interactive=False), gr.update(value="Preparing buildвЂ¦", visible=True), gr.skip()
                            try:
                                for message in build_cuda_runtime():
                                    yield gr.skip(), gr.update(value=message), gr.skip()
                            except Exception as error:
                                yield gr.update(interactive=True), gr.update(value=str(error)), gr.skip()
                            else:
                                yield gr.update(interactive=True), gr.skip(), gr.update(choices=backend_choices())

                        build_runtime.click(build_native_runtime, outputs=[build_runtime, build_status, llama_backend],
                            concurrency_id="native-runtime-build", concurrency_limit=1, api_visibility='private',
                            show_progress="hidden")
                    with gr.Accordion("Installed chat runtimes", open=False):
                        runtime_details = gr.Markdown("Click Refresh to inspect installed versions and devices.")
                        runtime_refresh = gr.Button("Refresh", size="sm")
                        gr.Markdown("Detected devices are separate from PyTorch. CPU mode and GPU-layer settings still apply.")
                        from modules.llama_runtime_info import runtime_info
                        runtime_refresh.click(runtime_info, outputs=runtime_details, api_visibility='undocumented')
                    curr_localfile = settings.default_settings.get("llama_localfile", None)
                    llama_localfile = gr.Textbox(label="Custom GGUF path", interactive=True,
                        info="Optional local file. Chat model and quantization selection is in Chat bots.",
                        value=curr_localfile)
                    add_setting("llama_localfile", llama_localfile)
                    llama_server_args = gr.Textbox(label="Extra llama.cpp arguments", placeholder="Optional, e.g. --reasoning off",
                                                   value=settings.default_settings.get("llama_server_args", ""))
                    add_setting("llama_server_args", llama_server_args)
                    llm_n_predict = gr.Number(label="Chat output token limit", info="Maximum output tokens; -1 means no fixed limit.", interactive=True, value=settings.default_settings.get("llm_n_predict", -1), minimum=-1, step=1)
                    add_setting("llm_n_predict", llm_n_predict)
                    llm_n_ctx = gr.Number(label="Chat context tokens", info="Maximum conversation context; larger values use more memory. 0 uses the runtime default.", interactive=True, value=settings.default_settings.get("llm_n_ctx", 8192), minimum=0, step=1)
                    add_setting("llm_n_ctx", llm_n_ctx)
                    llm_n_gpu_layers = gr.Number(label="Chat GPU layers", info="-1: automatic VRAM fitting in llama.cpp, full offload in xllamacpp. 0: CPU weights.", interactive=True, value=settings.default_settings.get("llm_n_gpu_layers", -1), minimum=-1, step=1)
                    add_setting("llm_n_gpu_layers", llm_n_gpu_layers)
                    llm_chat_history = gr.Number(label="Messages kept in chat history", info="0 keeps only the current message.", interactive=True, value=settings.default_settings.get("llm_chat_history", 7), minimum=0, step=1)
                    add_setting("llm_chat_history", llm_chat_history)
                    enable_llm_tools = gr.Checkbox(label=t("Enable image generation"), value=settings.default_settings.get("enable_llm_tools", False))
                    add_setting("enable_llm_tools", enable_llm_tools)
                    llm_hp_max_tokens = gr.Number(label="Prompt enhancer token limit", interactive=True, value=settings.default_settings.get("llm_hp_max_tokens", 256), minimum=1, step=1)
                    add_setting("llm_hp_max_tokens", llm_hp_max_tokens)

                with gr.Accordion(t("Other"), open=False):
                    video_fps = gr.Number(label="Fallback video frame rate", info="Used when the model recipe or Main video controls do not specify a frame rate.", interactive=True, value=settings.default_settings.get("video_fps", 30.0), minimum=0.01, step=0.01)
                    add_setting("video_fps", video_fps)
                    interrogator = gr.Dropdown(label=t("Default Interrogator"), info="Model used to describe an uploaded image when no prompt is supplied.", interactive=True, choices=list(looks.keys()), value=settings.default_settings.get("interrogator", "florence"),)
                    add_setting("interrogator", interrogator)
                    save_metadata = gr.Checkbox(label=t("Save Metadata"), value=settings.default_settings.get("save_metadata", True))
                    add_setting("save_metadata", save_metadata)

                    meta_comment = gr.Textbox(
                        label=t("Metadata Comment (Text added to the metadata)"),
                        interactive=True,
                        value=settings.default_settings.get("meta_comment", ""),
                        lines=5,
                        max_lines=5
                    )
                    add_setting("meta_comment", meta_comment)

                    update_interval = gr.Number(label=t("WebUI update interval"), interactive=True, value=settings.default_settings.get("update_interval", 0.1), minimum=0.01, step=0.01)
                    add_setting("update_interval", update_interval)
                    theme = gr.Textbox(label=t("Theme"), interactive=True, info="Leave blank for the built-in theme; otherwise enter a Gradio theme ID. Restart required.", value=settings.default_settings.get("theme", None))
                    add_setting("theme", theme)

            with gr.Column():
                gr.Markdown("### Model components")
                gr.Markdown("Automatic uses the filename shown and downloads missing catalog files when needed. "
                            "Overrides are advanced: a different architecture may fail to load. Restart after saving.")
                show_all_resources = gr.Checkbox(label="Show all component files (advanced)", value=False,
                    info="Includes installed files and download catalog entries; not all files are compatible with each model.")
                reset_components = gr.Button("Use automatic model components", size="sm")
                resource_controls = []
                resource_keys = []
                for prefix, title in (("clip_", "Text and vision encoder overrides"), ("vae_", "VAE overrides")):
                    with gr.Accordion(title, open=False):
                        for key in sorted(k for k in RESOURCE_DEFAULTS if k.startswith(prefix) and k != "clip_gemma2_it"):
                            selected = settings.default_settings.get(key) or ""
                            component = gr.Dropdown(label=RESOURCE_LABELS.get(key, key.removeprefix(prefix).replace("_", " ")),
                                choices=resource_choices(key, selected=selected), value=selected, interactive=True)
                            add_setting(key, component)
                            resource_keys.append(key)
                            resource_controls.append(component)
                def toggle_resource_choices(show_all, *values):
                    return [gr.update(choices=resource_choices(key, show_all, value))
                            for key, value in zip(resource_keys, values)]
                show_all_resources.change(toggle_resource_choices,
                    inputs=[show_all_resources] + resource_controls, outputs=resource_controls,
                    api_visibility="undocumented")
                shift_controls = []
                with gr.Accordion("Sampling shift overrides", open=False):
                    gr.Markdown("Leave blank for the model's normal sampling shift. Changes apply after restart.")
                    for key, label, default in (
                        ("auraflow_shift", "AuraFlow", 1.73),
                        ("lumina2_shift", "Lumina 2 / Z Image", 3.0),
                        ("hidream_shift", "HiDream", 3.0),
                        ("sd3_shift", "SD3", 3.0),
                        ("qwen_image_shift", "Qwen Image", 3.1),
                        ("newbieimage_shift", "NewBieImage", 6.0),
                        ("anima_image_shift", "Anima", None),
                    ):
                        component = gr.Textbox(label=label, placeholder="Automatic",
                            value=str(settings.default_settings[key]) if settings.default_settings.get(key) is not None else "",
                            info=f"Automatic: {default}" if default is not None else "Automatic: native model sampling")
                        add_setting(key, component)
                        shift_controls.append(component)
                reset_components.click(lambda: [gr.update(value="")] * len(resource_controls)
                    + [gr.update(value="")] * len(shift_controls),
                    outputs=resource_controls + shift_controls, api_visibility="undocumented")
                gr.Markdown("Use automatic model components resets overrides in this form. Click Save to keep the change.")

        with gr.Row(), gr.Group():
            ui_settings_name = gr.Text(
                label=t("Name"),
                interactive=True,
                placeholder=t("Optional"),
                value=settings.name,
            )
            add_setting("ui_settings_name", ui_settings_name)
            save_btn = gr.Button(t("Save"))

# Deal with this later
#            output = gr.Textbox(label="Status")
#            download_file = gr.File(label="Download File")
#        with gr.Row():
#            upload = gr.File(label="Load settings file (optional)", file_count="single", type="filepath")
#            download_btn = gr.Button("Download current settings")


        save_btn.click(
            fn=save_clicked,
            api_visibility='undocumented',
            inputs=state["setting_obj"],
        )

        # These files are checked in launch.py to trigger a --force-reinstall
        def trigger_reinstall_all():
            Path('reinstall').touch()
            gr.Info("Application Python packages and Torch will be reinstalled on the next online restart, using this Python environment. Overrides freezetorch.")
        def trigger_reinstall_torch():
            Path('reinstalltorch').touch()
            gr.Info("Torch will be reinstalled for the selected GPU runtime on the next online restart. Overrides freezetorch.")

        with gr.Group(), gr.Row():
            reinstall_all_btn = gr.Button(t("Trigger reinstall of all python modules"))
            reinstall_torch_btn = gr.Button(t("Trigger reinstall of torch"))

        reinstall_all_btn.click(
            fn=trigger_reinstall_all,
            api_visibility='undocumented',
        )
        reinstall_torch_btn.click(
            fn=trigger_reinstall_torch,
            api_visibility='undocumented',
        )

    return app_settings


