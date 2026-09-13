import gradio as gr
from pathlib import Path
from modules.sdxl_styles import load_styles
from modules.interrogate import looks

from shared import state, add_setting, performance_settings, resolution_settings, path_manager, settings, models, translate

t = translate


def model_choices(model_type, key, default):
    choices = list(models.names[model_type])
    choices += [name for name in path_manager.get_folder_list(model_type) if name not in choices]
    if model_type == "loras":
        choices.insert(0, "None")
    selected = settings.default_settings.get(key, default)
    if selected and selected not in choices:
        choices.append((f"{selected} (not found)", selected))
    return choices

def save_clicked(*args):
    ui_data = {}
    # Overwrite current settings
    for key, val in zip(state["setting_name"], args):
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

    print(t("Saved new settings to {path}.", mapping={"path": settings.settings_path}))
    gr.Info(t("Saved new settings to {path}.", mapping={"path": settings.settings_path}))

def create_settings():
    with gr.Blocks() as app_settings:
        with gr.Row():
            with gr.Column():
                gr.Markdown(t("# UI settings"))
                local_metadata = gr.Checkbox(label="Local model metadata only",
                    value=settings.default_settings.get("local_model_metadata", False),
                    info="Skip online model and artwork lookups.")
                add_setting("local_model_metadata", local_metadata)
                with gr.Row():
                    image_number = gr.Number(label=t("Image Number"), interactive=True, value=settings.default_settings.get("image_number", 1))
                    add_setting("image_number", image_number)
                    image_number_max = gr.Number(label=t("Image Number Max"), interactive=True, value=settings.default_settings.get("image_number_max", 50))
                    add_setting("image_number_max", image_number_max)
                with gr.Row():
                    seed = gr.Number(label=t("Seed"), interactive=True, value=settings.default_settings.get("seed", -1))
                    add_setting("seed", seed)
                    seed_random = gr.Checkbox(label=t("Random Seed"), interactive=True, value=settings.default_settings.get("seed_random", True))
                    add_setting("seed_random", seed_random)
                style = gr.Dropdown(
                    label=t("Style Selection"),
                    multiselect=True,
                    container=True,
                    choices=list(load_styles().keys()),
                    value=list(
                        set(settings.default_settings.get("style", [])) &
                        set(load_styles().keys())
                    ),
                )
                add_setting("style", style)
                prompt = gr.Textbox(label=t("Prompt"), interactive=True, value=settings.default_settings.get("prompt", ""))
                add_setting("prompt", prompt)
                negative_prompt = gr.Textbox(label=t("Negative Prompt"), interactive=True, value=settings.default_settings.get("negative_prompt", ""))
                add_setting("negative_prompt", negative_prompt)
                auto_negative_prompt = gr.Checkbox(label=t("Auto Negative Prompt"), interactive=True, value=settings.default_settings.get("auto_negative_prompt", False))
                add_setting("auto_negative_prompt", auto_negative_prompt)
                performance_choices = list(performance_settings.performance_options) + [performance_settings.CUSTOM_PERFORMANCE]
                saved_performance = settings.default_settings.get("performance", "SDXL")
                if saved_performance not in performance_choices:
                    saved_performance = performance_settings.CUSTOM_PERFORMANCE
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
                    value=settings.default_settings.get("resolution", "1344x768 (16:9)"),
                )
                add_setting("resolution", resolution)

                with gr.Row():
                    lora_min = gr.Number(label=t("LoRA weight min"), interactive=True, value=settings.default_settings.get("lora_min", 0))
                    add_setting("lora_min", lora_min)
                    lora_max = gr.Number(label=t("LoRA weight max"), interactive=True, value=settings.default_settings.get("lora_max", 2))
                    add_setting("lora_max", lora_max)

                gr.Markdown(t("# Preset"))
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

                gr.Markdown(t("# Models to load at startup"))
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
                    lora_1_weight = gr.Number(label=t("Lora {id} Weight", mapping={'id': 1}), value=settings.default_settings.get("lora_1_weight", 1.0), step=0.05)
                with gr.Row():
                    lora_2_model = gr.Dropdown(
                        label=t("LoRA {id} Model", mapping={'id': 2}),
                        interactive=True,
                        choices=model_choices("loras", "lora_2_model", "None"),
                        value=settings.default_settings.get("lora_2_model", "None"),
                    )
                    lora_2_weight = gr.Number(label=t("Lora {id} Weight", mapping={'id': 2}), value=settings.default_settings.get("lora_2_weight", 1.0), step=0.05)
                with gr.Row():
                    lora_3_model = gr.Dropdown(
                        label=t("LoRA {id} Model", mapping={'id': 3}),
                        interactive=True,
                        choices=model_choices("loras", "lora_3_model", "None"),
                        value=settings.default_settings.get("lora_3_model", "None"),
                    )
                    lora_3_weight = gr.Number(label=t("Lora {id} Weight", mapping={'id': 3}), value=settings.default_settings.get("lora_3_weight", 1.0), step=0.05)
                with gr.Row():
                    lora_4_model = gr.Dropdown(
                        label=t("LoRA {id} Model", mapping={'id': 4}),
                        interactive=True,
                        choices=model_choices("loras", "lora_4_model", "None"),
                        value=settings.default_settings.get("lora_4_model", "None"),
                    )
                    lora_4_weight = gr.Number(label=t("Lora {id} Weight", mapping={'id': 4}), value=settings.default_settings.get("lora_4_weight", 1.0), step=0.05)
                with gr.Row():
                    lora_5_model = gr.Dropdown(
                        label=t("LoRA {id} Model", mapping={'id': 5}),
                        interactive=True,
                        choices=model_choices("loras", "lora_5_model", "None"),
                        value=settings.default_settings.get("lora_5_model", "None"),
                    )
                    lora_5_weight = gr.Number(label=t("Lora {id} Weight", mapping={'id': 5}), value=settings.default_settings.get("lora_5_weight", 1.0), step=0.05)

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
                gr.Markdown(t("# One Button Prompt"))
                OBP_preset = gr.Textbox(label=t("OBP Preset"), value=settings.default_settings.get("OBP_preset", "Standard"))
                add_setting("OBP_preset", OBP_preset)
                hint_chance = gr.Number(label=t("Hint Chance"), value=settings.default_settings.get("hint_chance", 25))
                add_setting("hint_chance", hint_chance)
                
                gr.Markdown(t("# Image Browser"))
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
                gr.Markdown(t("# Paths"))
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

                gr.Markdown(t("# Chatbot settings"))
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
                llama_backend = gr.Dropdown(label="llama.cpp backend", choices=backend_choices(),
                    value=settings.default_settings.get("llama_backend", "Auto"),
                    info="Save settings, then load or chat to switch. LLAMA_SERVER overrides this selection.",
                    visible=llm_runtime.value == "llama.cpp")
                add_setting("llama_backend", llama_backend)
                llm_runtime.change(lambda runtime: gr.update(visible=runtime == "llama.cpp"),
                    inputs=llm_runtime, outputs=llama_backend, api_visibility='undocumented')
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
                llm_n_predict = gr.Number(label="n_predict", info="Maximum output tokens; -1 means no fixed limit.", interactive=True, value=settings.default_settings.get("llm_n_predict") or -1, minimum=-1, step=1)
                add_setting("llm_n_predict", llm_n_predict)
                llm_n_ctx = gr.Number(label="n_ctx", interactive=True, value=settings.default_settings.get("llm_n_ctx", 8192), minimum=0, step=1)
                add_setting("llm_n_ctx", llm_n_ctx)
                llm_n_gpu_layers = gr.Number(label="n_gpu_layers", info="-1 offloads all layers to GPU; 0 keeps model weights on CPU.", interactive=True, value=settings.default_settings.get("llm_n_gpu_layers", -1), minimum=-1, step=1)
                add_setting("llm_n_gpu_layers", llm_n_gpu_layers)
                llm_chat_history = gr.Number(label="chat_history", info="0 keeps only the current message.", interactive=True, placeholder=7, value=settings.default_settings.get("llm_chat_history", None), minimum=0, step=1)
                add_setting("llm_chat_history", llm_chat_history)
                enable_llm_tools = gr.Checkbox(label=t("Enable image generation"), value=settings.default_settings.get("enable_llm_tools", False))
                add_setting("enable_llm_tools", enable_llm_tools)
                llm_hp_max_tokens = gr.Number(label="max_tokens (Hyperprompting/Llamas)", interactive=True, placeholder=256, value=settings.default_settings.get("llm_hp_max_tokens", None), minimum=0, step=1)
                add_setting("llm_hp_max_tokens", llm_hp_max_tokens)

                gr.Markdown(t("# Other"))
                video_fps = gr.Number(label="video_fps", interactive=True, value=settings.default_settings.get("video_fps", 30.0), minimum=0.01, step=0.01)
                add_setting("video_fps", video_fps)
                interrogator = gr.Dropdown(label=t("Default Interrogator"), interactive=True, choices=list(looks.keys()), value=settings.default_settings.get("interrogator", None),)
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
                theme = gr.Textbox(label=t("Theme"), interactive=True, value=settings.default_settings.get("theme", None))
                add_setting("theme", theme)

            with gr.Column():
                gr.Markdown(t("# Clip"))
                clip_aura = gr.Dropdown(label="clip_aura (AuraFlow)", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_aura", None),)
                add_setting("clip_aura", clip_aura)
                clip_g = gr.Dropdown(label="clip_g (HiDream, SD3, SDXL)", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_g", None),)
                add_setting("clip_g", clip_g)
                clip_jina = gr.Dropdown(label="clip_jina (NewBieImage)", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_jina", None),)
                add_setting("clip_jina", clip_jina)
                clip_gemma = gr.Dropdown(label="clip_gemma", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_gemma", None),)
                add_setting("clip_gemma", clip_gemma)
                clip_l = gr.Dropdown(label="clip_l (SD1.5, Flux, HiDream, SD3, SDXL)", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_l", None),)
                add_setting("clip_l", clip_l)
                clip_llama = gr.Dropdown(label="clip_llama (HiDream)", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_llama", None),)
                add_setting("clip_llama", clip_llama)
                clip_llava = gr.Dropdown(label="clip_llava", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_llava", None),)
                add_setting("clip_llava", clip_llava)
                clip_ministral3 = gr.Dropdown(label="clip_ministral3 (ErnieImage)", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_ministral3", None),)
                add_setting("clip_ministral3", clip_ministral3)
                clip_mistral3 = gr.Dropdown(label="clip_mistral3 (Flux2)", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_mistral3", None),)
                add_setting("clip_mistral3", clip_mistral3)
                clip_qwen25 = gr.Dropdown(label="clip_qwen25 (QwenImage)", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_qwen25", None),)
                add_setting("clip_qwen25", clip_qwen25)
                clip_qwen3_06b = gr.Dropdown(label="clip_qwen3_06b (Anima)", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_qwen3_06b", None),)
                add_setting("clip_qwen3_06b", clip_qwen3_06b)
                clip_qwen3_4b = gr.Dropdown(label="clip_qwen3_4b (Flux2 Klein 4B, ZImage)", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_qwen3_4b", None),)
                add_setting("clip_qwen3_4b", clip_qwen3_4b)
                clip_qwen3_8b = gr.Dropdown(label="clip_qwen3_8b (Flux2 Klein )", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_qwen3_8b", None),)
                add_setting("clip_qwen3_8b", clip_qwen3_8b)
                clip_qwen3vl_8b = gr.Dropdown(label="clip_qwen3vl_8b (Ideogram4)", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_qwen3vl_8b", None),)
                add_setting("clip_qwen3vl_8b", clip_qwen3vl_8b)
                clip_qwen3vl_8b_scaled = gr.Dropdown(label="clip_qwen3vl_8b_scaled (Boogu-Image)", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_qwen3vl_8b_scaled", None),)
                add_setting("clip_qwen3vl_8b_scaled", clip_qwen3vl_8b_scaled)
                clip_oldt5 = gr.Dropdown(label="clip_oldt5 (CosmosPredict2)", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_oldt5", None),)
                add_setting("clip_oldt5", clip_oldt5)
                clip_t5 = gr.Dropdown(label="clip_t5 (Flux, HiDream, PixArt, SD3)", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_t5", None),)
                add_setting("clip_t5", clip_t5)
                clip_umt5 = gr.Dropdown(label="clip_umt5", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_umt5", None),)
                add_setting("clip_umt5", clip_umt5)
                clip_vision = gr.Dropdown(label="clip_vision", interactive=True, choices=[None]+path_manager.get_folder_list("clip_vision"), value=settings.default_settings.get("clip_vision", None),)
                add_setting("clip_vision", clip_vision)

                with gr.Accordion("Additional text encoders", open=False):
                    for key, label in {
                        "clip_gemma2_it_elm": "PixelDiT Gemma 2",
                        "clip_ernie_enhancer": "Ernie prompt enhancer",
                        "clip_gemma3": "NewBieImage Gemma 3",
                        "clip_qwen3vl_4b": "Mage-Flow Qwen3 VL",
                        "clip_qwen3vl_4b_scaled": "Krea 2 Qwen3 VL",
                        "clip_gemma3_12b": "LTX 2 / 2.3 Gemma 3",
                        "clip_ltx23_text_proj": "LTX 2.3 text projection",
                        "clip_ltx2_dev": "LTX 2 Dev connector",
                        "clip_ltx2_distilled": "LTX 2 Distilled connector",
                        "clip_gemma4_12b": "LTX 2.5 Gemma 4",
                        "clip_qwen3vl_32b": "MiniMax H3 Qwen3 VL",
                    }.items():
                        component = gr.Dropdown(label=label, interactive=True,
                            choices=[None] + path_manager.get_folder_list("clip"),
                            value=settings.default_settings.get(key))
                        add_setting(key, component)

                gr.Markdown(t("# Shift"))
                auraflow_shift = gr.Textbox(label="AuraFlow shift", interactive=True, placeholder=1.73, value=settings.default_settings.get("auraflow_shift", None))
                add_setting("auraflow_shift", auraflow_shift)
                lumina2_shift = gr.Textbox(label="Lumina2 shift", interactive=True, placeholder=3.0, value=settings.default_settings.get("lumina2_shift", None))
                add_setting("lumina2_shift", lumina2_shift)
                hidream_shift = gr.Textbox(label="HiDream shift", interactive=True, placeholder=3.0, value=settings.default_settings.get("hidream_shift", None))
                add_setting("hidream_shift", hidream_shift)
                sd3_shift = gr.Textbox(label="SD3 shift", interactive=True, placeholder=3.0, value=settings.default_settings.get("sd3_shift", None))
                add_setting("sd3_shift", sd3_shift)
                qwen_image_shift = gr.Textbox(label="Qwen Image shift", interactive=True, placeholder=3.1, value=settings.default_settings.get("qwen_image_shift", None))
                add_setting("qwen_image_shift", qwen_image_shift)

                gr.Markdown(t("# VAE"))
                vae_auraflow = gr.Dropdown(label="AuraFlow VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_auraflow", None),)
                add_setting("vae_auraflow", vae_auraflow)
                vae_flux = gr.Dropdown(label="Flux VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_flux", None),)
                add_setting("vae_flux", vae_flux)
                vae_flux2 = gr.Dropdown(label="Flux2 VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_flux2", None),)
                add_setting("vae_flux2", vae_flux2)
                vae_hunyuan_video = gr.Dropdown(label="Hunyuan Video VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_hunyuan_video", None),)
                add_setting("vae_hunyuan_video", vae_hunyuan_video)
                vae_lumina2 = gr.Dropdown(label="Lumina2 VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_lumina2", None),)
                add_setting("vae_lumina2", vae_lumina2)
                vae_qwen_image = gr.Dropdown(label="Qwen Image VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_qwen_image", None),)
                add_setting("vae_qwen_image", vae_qwen_image)
                vae_pixart = gr.Dropdown(label="PixArt VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_pixart", None),)
                add_setting("vae_pixart", vae_pixart)
                vae_sd = gr.Dropdown(label="SD1.5 VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_sd", None),)
                add_setting("vae_sd", vae_sd)
                vae_sd3 = gr.Dropdown(label="SD3 VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_sd3", None),)
                add_setting("vae_sd3", vae_sd3)
                vae_sdxl = gr.Dropdown(label="SDXL/Pony/Illustrious VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_sdxl", None),)
                add_setting("vae_sdxl", vae_sdxl)
                vae_wan = gr.Dropdown(label="WAN 2.1 VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_wan", None),)
                add_setting("vae_wan", vae_wan)
                vae_wan_22 = gr.Dropdown(label="WAN 2.2 VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_wan_22", None),)
                add_setting("vae_wan_22", vae_wan_22)

                with gr.Accordion("Additional VAEs", open=False):
                    for key, label in {
                        "vae_mage_flow": "Mage-Flow",
                        "vae_ltxv": "LTX Video",
                        "vae_ltxv23_audio": "LTX 2.3 audio",
                        "vae_ltxv2_audio": "LTX 2 audio",
                        "vae_ltxv2_video": "LTX 2 video",
                        "vae_ltxv23_video": "LTX 2.3 video",
                        "vae_ltxv25_audio": "LTX 2.5 audio",
                        "vae_ltxv25_video": "LTX 2.5 video",
                        "vae_minimax_h3_audio": "MiniMax H3 audio",
                        "vae_minimax_h3_video": "MiniMax H3 video",
                    }.items():
                        component = gr.Dropdown(label=label, interactive=True,
                            choices=[None] + path_manager.get_folder_list("vae"),
                            value=settings.default_settings.get(key))
                        add_setting(key, component)

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


