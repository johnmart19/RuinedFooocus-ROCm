import gradio as gr
from shared import path_manager, settings
import modules.async_worker as worker
from pathlib import Path
import json
from PIL import Image
import base64
import re
from modules.llama_models import (DEFAULT_MODEL, model_catalogue, model_label, preferred_quant,
                                 import_model, local_models, remove_local_model)
from modules.llama_file_picker import select_gguf
from modules.chat_code import python_runner


def display_history(history, show_reasoning):
    """Keep full history in session state; thoughts are an optional display detail."""
    messages = []
    for message in history:
        if message["role"] == "assistant" and isinstance(message["content"], str):
            # Close partial streamed thoughts so Gradio keeps them collapsed, too.
            replacement = (lambda m: f"<{m[1]}>{m[2]}</{m[1]}>") if show_reasoning else ""
            text = re.sub(r"<(think|thinking)>(.*?)(?:</\1>|$)", replacement,
                          message["content"], flags=re.DOTALL).strip()
            if not text:
                continue
            message = dict(message, content=text)
        messages.append(message)
    return messages


def create_chat(image_controls=None):
    image_controls = image_controls or {}
    def info_from_char(file):
        with Image.open(str(file)) as i:
            i.getexif()
            if 'chara' in i.info:
                d = i.info.get('chara', None)
            else:
                return None

        b = base64.b64decode(d) 
        j = json.loads(b)
        spec= j.get('spec', '')

        if spec == 'chara_card_v3':
            name = j['data']['name']
            greeting = j['data']['first_mes']
            avatar = file
            personality = j['data']['personality']
            scenario = j['data']['scenario']
            summary = j['data']['description']
        elif spec == 'chara_card_v2':
            name = j['data']['name']
            greeting = j['data']['first_mes']
            avatar = file
            personality = j['data']['personality']
            scenario = j['data']['scenario']
            summary = j['data']['description']
        elif all(key in j for key in ('name', 'first_mes', 'personality', 'scenario')):
            # chara_card_v1
            name = j['name']
            greeting = j['first_mes']
            avatar = file
            personality = j['personality']
            scenario = j['scenario']
            summary = j.get('summary', '')
        else:
            print(f"WARNING: Can not parse {file}, skipping.")
            return None

        info = {
            "name": name,
            "greeting": greeting,
            "avatar": avatar,
            "system": f"Your name is {name}.\nYou are: {personality}\nScenario: {scenario}",
            "embed": json.dumps([["text", f"Summary: {summary}"]]),
        }
        return info

    def llama_get_assistants():
        names = []
        folder_path = Path("chatbots")
        for path in folder_path.rglob("*"):
            if path.is_dir():
                try:
                    with open(path / "info.json" , "r", encoding='utf-8') as f:
                        info = json.load(f)
                    names.append((info["name"], str(path)))
                except Exception as e:
                    print(f"ERROR: in folder {path}: {e}")
                    pass
            else:
                # Ignore png's that has a info.json in the same folder
                if str(Path(path).suffix).lower() == ".png" and not Path(Path(path).parent / "info.json").exists():
                    # Try as aichar card
                    try:
                        character = info_from_char(path)
                        if character is not None:
                            names.append((character.get('name', '???'), str(path)))
                    except Exception as e:
                        print(f"ERROR: in character card {path}: {e}")
                        pass

        names.sort(key=lambda x: x[0].casefold())
        return [("Normal", "__normal__")] + names

    def gr_llama_get_assistants(source, model):
        return {
            llama_assistants: gr.update(
                choices=llama_get_assistants(),
            ),
            **select_source(source, model),
        }

    def _llama_select_assistant(dropdown):
        if dropdown == "__normal__":
            return {"name": "Normal", "avatar": None, "system": "",
                    "embed": "[]", "chatstart": []}
        character = Path(dropdown)
        try:
            if character.is_dir():
                with open(character / "info.json", "r", encoding='utf-8') as f:
                    info = json.load(f)
                    if "avatar" not in info:
                        info["avatar"] = character / "avatar.png"
                    if "embed" in info:
                        info["embed"] = json.dumps(info["embed"])
                    else:
                        info["embed"] = json.dumps([])
            else: 
                info = info_from_char(character)

        except Exception as e:
            print(f"ERROR: {dropdown}: {e}")
            info = {
                "name": "Error",
                "greeting": "Error!",
                "avatar": "html/error.png",
                "system": "Everything is broken.",
                "embed": json.dumps([]),
            }
            pass
        info["chatstart"] = [{"role": "assistant", "content": info["greeting"]}]
        return info

    def llama_select_assistant(dropdown):
        info = _llama_select_assistant(dropdown)
        return {
            llama_chat: gr.update(value=info["chatstart"]),
            llama_history: info["chatstart"],
            llama_msg: gr.update(value="", placeholder=info.get("input_hint", "Message")),
            llama_avatar: gr.update(
                value=info["avatar"],
                label=info["name"],
                visible=info["avatar"] is not None,
            ),
            llama_system: gr.update(value=info["system"]),
            llama_embed: gr.update(value=info["embed"])
        }


    selected_file = settings.default_settings.get("llama_localfile") or DEFAULT_MODEL
    default_model = next((name for name, choices in model_catalogue().items()
        if any(Path(value).name == Path(selected_file).name for _, value in choices)),
        "Qwen2.5-7B-Instruct-abliterated-v2")
    default_source = "Local models" if default_model in model_catalogue("Local models") else "Downloadable"
    catalogue = model_catalogue(default_source)
    if default_model not in catalogue:
        default_model = next(iter(catalogue), None)
    default_choices = catalogue.get(default_model, [])
    html_dir = Path(__file__).resolve().parents[2] / "html"
    with gr.Blocks() as app_llama_chat:
        with gr.Row(elem_id="chat-layout"):
            with gr.Column(scale=3, min_width=0, elem_id="chat-conversation"), gr.Group():
                default_bot = str(Path("chatbots/rf_support_troll"))
                llama_history = gr.State(_llama_select_assistant(default_bot)["chatstart"])
                llama_chat = gr.Chatbot(
                    label="",
                    show_label=False,
                    height=600,
                    resizable=True,
                    elem_id="chat-messages",
                    reasoning_tags=[("<think>", "</think>"), ("<thinking>", "</thinking>")],
                    buttons=['copy_all'],
                    value=_llama_select_assistant(default_bot)["chatstart"],
                )
                llama_msg = gr.Textbox(
                    show_label=False,
                )
                clear_cached_chat = gr.Checkbox(value=False,
                    label="Also delete cached chat data when clearing",
                    info="Deletes all local chat caches, including chatbot images, and unloads the chat model. Models and settings are kept.")
                llama_sent = gr.Textbox(visible='hidden')
                with gr.Accordion("Run Python", open=True, visible=False) as python_panel:
                    load_code = gr.Button("Load code from chat")
                    code_runner = gr.HTML("", js_on_load=(Path(__file__).resolve().parents[2]
                        / "html" / "chat_python_theme.js").read_text(encoding="utf-8"))
            gr.HTML('<div role="separator" tabindex="0" aria-label="Resize chat" '
                    'aria-orientation="vertical" aria-valuemin="35" aria-valuemax="85" '
                    'aria-valuenow="72"></div>', elem_id="chat-divider", min_width=12, scale=0,
                    js_on_load=(Path(__file__).resolve().parents[2] / "html" / "chat_splitter.js").read_text(encoding="utf-8"))
            with gr.Column(scale=2, min_width=0, elem_id="chat-controls"), gr.Group():
                llama_avatar = gr.Image(
                    value=_llama_select_assistant(default_bot)["avatar"],
                    label=_llama_select_assistant(default_bot)["name"],
                    height=400,
                    width=400,
                    show_label=True,
                )
                with gr.Row():
                    llama_reload = gr.Button(value="↻")
                    llama_assistants = gr.Dropdown(
                        choices=llama_get_assistants(),
                        value=default_bot,
                        label="Chatbot",
                        show_label=True,
                        interactive=True,
                        scale=7,
                        buttons=[llama_reload],
                    )
                llama_active_model = gr.State(selected_file)
                llama_source = gr.Radio([("Local", "Local models"), ("Cloud", "Downloadable")],
                    value=default_source, label="Model source", show_label=False, elem_id="chat-model-source")
                llama_model = gr.Dropdown(
                    label="Model", choices=[(model_label(name), name) for name in catalogue],
                    value=default_model, interactive=True, visible=default_source == "Downloadable",
                )
                local_picker = gr.HTML(
                    {"choices": [(model_label(name), name) for name in model_catalogue("Local models")],
                     "selected": default_model}, visible=default_source == "Local models",
                    elem_id="chat-local-models",
                    html_template=(html_dir / "chat_local_models.html").read_text(encoding="utf-8"),
                    js_on_load=(html_dir / "chat_local_models.js").read_text(encoding="utf-8"),
                    css_template="""
                      label { display: block; margin: 0 0 8px; font-size: var(--block-label-text-size); }
                      details { margin: 0; padding: 0; }
                      summary { cursor: pointer; list-style: none; padding: 6px 12px; min-height: 42px;
                        margin: 0; box-sizing: border-box; line-height: 28px; border-radius: var(--input-radius);
                        background: var(--input-background-fill); border: 1px solid var(--border-color-primary); }
                      summary::after { content: '▾'; float: right; }
                      .model-options { max-height: 280px; overflow-y: auto; margin-top: 4px;
                        border: 1px solid var(--border-color-primary); border-radius: var(--input-radius);
                        background: var(--input-background-fill); box-shadow: 0 4px 12px #0003; }
                      .model-options > div { display: flex; align-items: stretch; margin: 0;
                        border-bottom: 1px solid var(--border-color-primary); }
                      .model-options > div:last-child { border-bottom: 0; }
                      .model-options > div:has([aria-pressed="true"]) {
                        background: color-mix(in srgb, var(--color-accent) 20%, var(--input-background-fill)); }
                      .model-options > div:hover { background: var(--button-secondary-background-fill-hover); }
                      button { background: transparent; color: var(--body-text-color);
                        cursor: pointer; border: 0; padding: 9px; font: inherit; margin: 0; border-radius: 0; }
                      button:focus-visible { outline: 2px solid var(--color-accent); outline-offset: -2px; }
                      .choose-model[aria-pressed="true"] { color: var(--color-accent); font-weight: 600; }
                      .choose-model { flex: 1; min-width: 0; text-align: left; overflow-wrap: anywhere; }
                      .remove-model { flex: 0 0 34px; font-size: 20px; }
                    """)
                llama_quant = gr.Dropdown(label="Quantization", interactive=True, visible=len(default_choices) > 1,
                    choices=default_choices, value=preferred_quant(default_choices) if default_choices else None)
                with gr.Row(elem_id="chat-model-actions"):
                    llama_use = gr.Button("Load",
                                          size="md", scale=1, min_width=0)
                    llama_import = gr.Button("Import GGUF", size="md", scale=1, min_width=0)
                llama_model_status = gr.Markdown("Load downloads the selected model if needed.")
                llama_download = gr.HTML(visible=False)
                show_reasoning = gr.Checkbox(label="Show reasoning", value=False,
                    info="Only shown when the model returns reasoning.")
                from modules.llama_vision import model_has_vision, vision_models
                from shared import path_manager
                enable_vision = gr.Checkbox(label="Enable Model Vision", value=False)
                vision_include_system = gr.Checkbox(label="Include system prompt", value=False,
                    visible=False, info="Include the chatbot's character instructions as image review context.")
                vision_model = gr.Dropdown(label="Select Model", visible=False,
                    choices=[(model_label(Path(name).stem), name) for name in vision_models(path_manager)],
                    value=None, info="Used to inspect images; downloaded when needed.")
                def vision_controls(enabled, model):
                    return gr.update(visible=enabled and not model_has_vision(model))
                enable_vision.change(vision_controls, inputs=[enable_vision, llama_active_model],
                    outputs=vision_model, api_visibility='undocumented')
                enable_vision.change(lambda enabled: gr.update(visible=enabled), inputs=enable_vision,
                    outputs=vision_include_system, api_visibility='undocumented')
                llama_active_model.change(vision_controls, inputs=[enable_vision, llama_active_model],
                    outputs=vision_model, api_visibility='undocumented')
                enable_python = gr.Checkbox(label="Enable Python runner", value=False)
                pending_remove = gr.State(None)
                remove_dialog = gr.HTML("", elem_id="chat-model-confirm",
                    html_template=(html_dir / "chat_model_remove.html").read_text(encoding="utf-8"),
                    js_on_load=(html_dir / "chat_model_remove.js").read_text(encoding="utf-8"),
                    css_template="""
                      dialog { max-width: min(560px, 90vw); max-height: 85vh; overflow: auto;
                        margin: auto; padding: 24px; border-radius: 8px;
                        background: var(--block-background-fill); color: var(--body-text-color);
                        border: 1px solid var(--border-color-primary); }
                      dialog::backdrop { background: #0009; }
                      .model-path { overflow-wrap: anywhere; font-family: monospace; }
                      select { max-width: 100%; padding: 8px; background: var(--input-background-fill);
                        color: var(--body-text-color); border: 1px solid var(--border-color-primary); }
                      .actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
                      button { padding: 8px 12px; border-radius: 6px; cursor: pointer;
                        background: var(--button-secondary-background-fill); color: var(--body-text-color); }
                      button[data-action="delete"] { background: #b91c1c; color: white; }
                    """)
                llama_system = gr.Textbox(
                    visible='hidden',
                    value=_llama_select_assistant(default_bot)["system"],
                )
                llama_embed = gr.Textbox(
                    visible='hidden',
                    value=_llama_select_assistant(default_bot)["embed"],
                )

        def llama_get_text(message):
            return "", message

        load_code.click(python_runner, inputs=[llama_history], outputs=[code_runner],
                        api_visibility='undocumented')
        enable_python.change(
            lambda enabled, history: (gr.update(visible=enabled), python_runner(history) if enabled else ""),
            inputs=[enable_python, llama_history], outputs=[python_panel, code_runner],
            api_visibility='undocumented')

        def llama_respond(message, system, embed, chat_history, model, show_thinking, vision_enabled, review_model, include_system, *image_values):
            chat_history = list(chat_history)
            chat_history.append({"role": "user", "content": message})

            gen_data = {
                "task_type": "llama",
                "vision_enabled": vision_enabled,
                "vision_include_system": include_system,
                "vision_model": review_model,
                "model": model,
                "system": system,
                "embed": embed,
                "history": chat_history,
                "image_settings": dict(zip(image_controls, image_values)),
            }

            # Add work
            task_id = worker.add_task(gen_data.copy())

            # Wait for result
            finished = False
            while not finished:
                flag, product = worker.task_result(task_id)
                if flag == "preview":
                    yield {llama_chat: display_history(product, show_thinking), llama_history: product,
                           llama_model_status: "Thinking…", llama_download: gr.update(visible=False)}
                elif flag == "download":
                    yield download_update(product)
                elif flag == "results":
                    finished = True

            chat_history.append({"role": "assistant", "content": product})
            yield {llama_chat: display_history(chat_history, show_thinking), llama_history: chat_history,
                   llama_model_status: "", llama_download: gr.update(visible=False)}

        def download_update(progress):
            received, total, speed = progress
            if received is None:
                return {llama_download: gr.update(visible=False), llama_model_status: "Loading model…"}
            amount = f"{received / 1000000:.1f} MB"
            value = ""
            if total:
                percent = min(100, received * 100 / total)
                value = f'value="{percent:.1f}"'
                amount = f"{percent:.0f}%"
            return {llama_model_status: f"Downloading: {amount} · Speed: {speed / 1000000:.1f} MB/s",
                    llama_download: gr.update(visible=True, value=
                        f'<progress max="100" {value} style="width:100%;accent-color:var(--color-accent)"></progress>')}

        def select_model(model, source, selected_quant=None):
            choices = model_catalogue(source).get(model, [])
            value = selected_quant if selected_quant else preferred_quant(choices) if choices else None
            return {
                llama_quant: gr.update(choices=choices, value=value, visible=len(choices) > 1),
                llama_model_status: "Click Load to load this model." if choices else "Choose a model or import a local GGUF.",
                llama_msg: gr.update(interactive=False),
                llama_use: gr.update(value="Load",
                                     interactive=bool(choices)),
            }

        def select_source(source, model=None):
            catalogue = model_catalogue(source)
            if model not in catalogue:
                model = next(iter(catalogue), None)
            choices = [(model_label(name), name) for name in catalogue]
            return {llama_model: gr.update(choices=choices, value=model, visible=source == "Downloadable"),
                    local_picker: gr.update(value={"choices": choices, "selected": model}, visible=source == "Local models"),
                    **select_model(model, source)}

        def import_gguf():
            try:
                path = select_gguf()
                if not path:
                    return {llama_model_status: gr.skip()}
                model, quant = import_model(path)
            except (OSError, ValueError, RuntimeError) as error:
                return {llama_model_status: str(error)}
            return {llama_source: "Local models", **select_source("Local models", model),
                    **select_model(model, "Local models", quant)}

        def use_model(model, selection, source, action):
            if action == "Unload":
                yield {llama_model_status: "Unloading model…", llama_msg: gr.update(interactive=False),
                       llama_use: gr.update(interactive=False)}
                task_id = worker.add_task({"task_type": "llama", "unload": True})
                while True:
                    flag, result = worker.task_result(task_id)
                    if flag == "results":
                        break
                if isinstance(result, dict) and result.get("unloaded"):
                    yield {llama_active_model: None, llama_model_status: "Model unloaded.",
                           llama_use: gr.update(value="Load", interactive=bool(model))}
                else:
                    yield {llama_model_status: str(result), llama_use: gr.update(interactive=True)}
                return
            if not model:
                yield {llama_model_status: "Choose a model and quantization first."}
                return
            if Path(model).is_absolute() and not Path(model).is_file():
                yield {llama_model_status: "The model file is missing. Import it again from its current location."}
                return
            if model not in [value for _, value in model_catalogue(source).get(selection, [])]:
                yield {llama_model_status: "Choose an available quantization, then click Load."}
                return
            yield {llama_model_status: "Downloading / loading model…", llama_msg: gr.update(interactive=False),
                   llama_use: gr.update(value="Loading…", interactive=False)}
            task_id = worker.add_task({"task_type": "llama", "model": model, "load_only": True})
            while True:
                flag, result = worker.task_result(task_id)
                if flag == "download":
                    yield download_update(result)
                if flag == "results":
                    break
            yield {llama_download: gr.update(visible=False)}
            if isinstance(result, dict) and result.get("ready"):
                yield {llama_source: "Local models", **select_source("Local models", selection),
                       **select_model(selection, "Local models", model), llama_active_model: model,
                       llama_model_status: f"Ready: {Path(model).name}", llama_msg: gr.update(interactive=True),
                       llama_use: gr.update(value="Unload", interactive=True)}
            else:
                yield {llama_model_status: str(result), llama_use: gr.update(value="Load", interactive=True)}

        model_outputs = [llama_model, local_picker, llama_quant, llama_model_status, llama_msg, llama_use]

        def local_model_action(source, current_quant, event: gr.EventData):
            data = event._data or {}
            model = data.get("model")
            choices = model_catalogue("Local models").get(model, [])
            if source != "Local models" or not choices:
                return {llama_model_status: "This model is no longer available. Refresh the list."}
            if data.get("action") == "select":
                return select_source("Local models", model)
            if data.get("action") != "remove":
                return {llama_model_status: gr.skip()}
            files = local_models()
            entries = [(label, str(Path(files.get(value, value)).resolve())) for label, value in choices]
            paths = [path for _, path in entries]
            current_path = str(Path(files.get(current_quant, current_quant)).resolve()) if current_quant else None
            return {pending_remove: paths,
                    remove_dialog: {"files": entries, "selected": current_path if current_path in paths else paths[0]}}

        def confirm_remove(paths, selection, quant, event: gr.EventData):
            data = event._data or {}
            action, index = data.get("action"), data.get("file_index")
            result = {pending_remove: None, remove_dialog: ""}
            if action not in ("hide", "delete") or type(index) is not int or not 0 <= index < len(paths or []):
                return result
            path = paths[index]
            try:
                remove_local_model(path, delete_file=action == "delete")
            except (OSError, ValueError) as error:
                return {**result, llama_model_status: str(error)}
            updates = select_source("Local models", selection)
            choices = model_catalogue("Local models").get(selection, [])
            if quant in [value for _, value in choices]:
                updates[llama_quant] = gr.update(choices=choices, value=quant, visible=len(choices) > 1)
                updates[llama_msg] = gr.skip()
                updates[llama_use] = gr.skip()
            return {**result, **updates,
                    llama_model_status: "File deleted." if action == "delete" else "Removed from list. The file was kept."}

        local_picker.click(local_model_action, inputs=[llama_source, llama_quant],
            outputs=[pending_remove, remove_dialog, *model_outputs], api_visibility='undocumented')
        remove_dialog.click(confirm_remove, inputs=[pending_remove, llama_model, llama_quant],
            outputs=[pending_remove, remove_dialog, *model_outputs], api_visibility='undocumented')
        llama_import.click(import_gguf, outputs=[llama_source, *model_outputs], api_visibility='undocumented')
        llama_source.input(select_source, inputs=[llama_source, llama_model],
                           outputs=model_outputs, api_visibility='undocumented')
        llama_model.input(select_model, inputs=[llama_model, llama_source],
                          outputs=model_outputs, api_visibility='undocumented')
        llama_quant.input(lambda quant: ("Click Load to load this quantization." if quant else "Choose a quantization.",
                                        gr.update(interactive=False), gr.update(value="Load", interactive=bool(quant))),
            inputs=[llama_quant], outputs=[llama_model_status, llama_msg, llama_use], api_visibility='undocumented')
        llama_use.click(use_model, inputs=[llama_quant, llama_model, llama_source, llama_use],
            outputs=[llama_active_model, llama_source, llama_download, *model_outputs], api_visibility='undocumented',
            show_progress="hidden")

        llama_msg.submit(
            fn=llama_get_text,
            api_visibility='undocumented',
            inputs=[llama_msg],
            outputs=[llama_msg, llama_sent]
        ).then(
            fn=llama_respond,
            api_visibility='undocumented',
            inputs=[llama_sent, llama_system, llama_embed, llama_history, llama_active_model, show_reasoning, enable_vision, vision_model, vision_include_system,
                    *image_controls.values()],
            outputs=[llama_chat, llama_history, llama_model_status, llama_download],
            show_progress="hidden"
        )

        llama_assistants.select(
            fn=llama_select_assistant,
            api_visibility='undocumented',
            inputs=[llama_assistants],
            outputs=[llama_chat, llama_msg, llama_avatar, llama_system, llama_embed, llama_history]
        )
        show_reasoning.change(display_history, inputs=[llama_history, show_reasoning],
                              outputs=[llama_chat], api_visibility='undocumented')
        def clear_chat(delete_cache):
            result = {llama_history: [], code_runner: ""}
            if delete_cache:
                from modules.api_runtime import submit
                from modules.chat_storage import clear_chat_cache
                job = submit(lambda job, task_id: clear_chat_cache())
                kind, value = job.events.get(timeout=600)
                if kind != "result":
                    raise gr.Error(str(value))
                removed, failed = value
                result.update({llama_use: gr.update(value="Load"),
                    llama_model_status: f"Cleared {removed} cached files." +
                        (f" {failed} locked files remain; close other sessions and retry." if failed else ""),
                    clear_cached_chat: False})
            return result

        llama_chat.clear(clear_chat, inputs=[clear_cached_chat],
            outputs=[llama_history, code_runner, llama_use, llama_model_status, clear_cached_chat],
            api_visibility='undocumented')
        llama_reload.click(
            fn=gr_llama_get_assistants,
            inputs=[llama_source, llama_model],
            api_visibility='undocumented',
            outputs=[llama_assistants, *model_outputs]
        )

    return app_llama_chat
