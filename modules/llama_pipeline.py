import re
import gc
from modules.llama_server import Server as NativeServer
from modules.llama_models import DEFAULT_MODEL, resolve_model
from modules.llama_vision import projector_path, review_messages, remember_image, attach_latest_image, model_has_vision, has_generated_image, attach_image_feedback
from txtai import Embeddings
from modules.util import TimeIt
from pathlib import Path
from modules.util import url_to_filename, load_file_from_url
from shared import path_manager, settings, local_url
import json
import xmltodict
import modules.async_worker as worker

def llama_names():
        names = []
        folder_path = Path("llamas")
        for path in folder_path.rglob("*"):
            if path.suffix.lower() in [".txt"]:
                f = open(path, "r", encoding='utf-8')
                name = f.readline().strip()
                names.append((name, str(path)))
        names.sort(key=lambda x: x[0].casefold())
        return names

def run_llama(system_file, prompt, *, instruction=None):
        name = None
        sys_pat = r"system:.*\n\n"
        system = re.match(sys_pat, prompt, flags=re.M|re.I)
        if instruction is not None:
            name = "Image prompt"
            system_prompt = instruction
        elif system is not None: # Llama system-prompt provided in the ui-prompt
            name = "Llama"
            system_prompt = re.sub(r"^[^:]*: *", "", system.group(0), flags=re.M|re.I)
            prompt = re.sub(sys_pat, "", prompt)
        else:
            try:
                file = open(system_file, "r", encoding='utf-8')
                name = name if name is not None else file.readline().strip()
                system_prompt = file.read().strip()
            except:
                print(f"LLAMA ERROR: Could not open file {system_file}")
                return prompt

        llama = pipeline()
        llama.load_base_model()

        with TimeIt(""):
            print(f"# {name}: (Thinking...)")
            try:

                ret = llama.llm.handle_chat_completions(
                    {
                        "max_tokens": settings.default_settings.get("llm_hp_max_tokens", 256),
                        "messages": [{"role": "system", "content": system_prompt},
                                     {"role": "user", "content": prompt}],
                    }
                )
                if instruction is not None:
                    from modules.prompt_enhancement import image_prompt_result
                    res = image_prompt_result(ret['choices'][0], prompt)
                else:
                    res = ret['choices'][0]['message']['content'] or prompt
            except Exception as e:
                print(f"LLAMA ERROR: {e}")
                res = prompt


        llama.unload()

        return res

class pipeline:
    pipeline_type = ["llama"]

    llm = None
    embeddings = None
    embeddings_hash = ""
    model_settings = None
    runtime_name = None
    runtime_label = None
    vision_projector = None

    def current_model_settings(self, model=None):
        keys = ("llm_runtime", "llama_backend", "llama_localfile", "llama_server_args", "llm_n_ctx",
                "llm_n_predict", "llm_n_gpu_layers", "llama_mmproj")
        return (model,) + tuple(settings.default_settings.get(key) for key in keys)

    def parse_gen_data(self, gen_data):
        return gen_data

    def unload(self):
        if isinstance(self.llm, NativeServer):
            self.llm.close()
        self.llm = None
        self.runtime_name = None
        self.runtime_label = None
        self.vision_projector = None
        self.model_settings = None
        self.embeddings = None
        self.embeddings_hash = ""
        gc.collect()

    def load_base_model(self, model=None, progress=None):
        self.unload()
        localfile = model or settings.default_settings.get("llama_localfile") or DEFAULT_MODEL
        llm_path = resolve_model(localfile, progress=progress)
        projector = projector_path(settings.default_settings, llm_path, progress=progress)
        self.vision_projector = projector
        if progress:
            progress(None, None, None)
        with TimeIt("Load LLM"):
            print(f"Loading {localfile}")
            from argparser import args

            if settings.default_settings.get("llm_runtime", "llama.cpp") == "llama.cpp":
                import torch
                runtime_settings = settings.default_settings.copy()
                runtime_settings["llama_mmproj"] = str(projector) if projector else "None"
                if args.directml is not None and runtime_settings.get("llama_backend") != "CPU":
                    runtime_settings["llama_backend"] = "Vulkan"
                gpu = not args.cpu and (args.directml is not None or
                    runtime_settings.get("llama_backend", "Auto") not in ("Auto", "CPU") or
                    torch.cuda.is_available() or torch.backends.mps.is_available())
                self.llm = NativeServer(llm_path, runtime_settings, gpu=gpu)
                self.runtime_label = self.llm.runtime_label
            else:
                import xllamacpp as xlc
                params = xlc.CommonParams()
                params.prompt = ""
                params.model.path = str(llm_path)
                if projector:
                    params.mmproj.path = str(projector)
                params.n_predict = int(settings.default_settings.get("llm_n_predict") or -1)
                params.n_ctx = int(settings.default_settings.get("llm_n_ctx", 8192))
                params.n_gpu_layers = 0 if args.cpu else int(settings.default_settings.get("llm_n_gpu_layers", -1))
                if args.cpu:
                    params.no_kv_offload = True
                    params.no_op_offload = True
                    params.mmproj_use_gpu = False
                params.ctx_shift = True
                params.cpuparams.n_threads = 4
                params.cpuparams_batch.n_threads = 2
                params.endpoint_metrics = False
                params.use_jinja = True

                self.llm = xlc.Server(params)
                try:
                    devices = xlc.get_device_info()
                    self.runtime_label = "/".join(dict.fromkeys(
                        re.sub(r"\d+$", "", device["name"]) for device in devices
                        if device["name"] != "CPU")) or "CPU"
                except Exception:
                    self.runtime_label = "unknown"
                if params.n_gpu_layers == 0 and self.runtime_label != "CPU":
                    self.runtime_label += "; CPU weights"

        self.runtime_name = settings.default_settings.get("llm_runtime", "llama.cpp")
        self.model_settings = self.current_model_settings(model)
        self.embeddings = None

    def index_sources(self, sources):
        from modules.chat_storage import chat_folder, migrate_legacy_chat_files
        migrate_legacy_chat_files()
        if not isinstance(sources, list):
            raise ValueError("Chat context must be a list of text or URL sources.")
        documents = []
        for source in sources:
            if (not isinstance(source, (list, tuple)) or len(source) != 2
                    or source[0] not in ("text", "url") or not isinstance(source[1], str)):
                raise ValueError("Each chat context source must be [text, content] or [url, address].")
            kind, content = source
            if kind == "url":
                filename = load_file_from_url(content, model_dir=str(chat_folder("sources")),
                    progress=True, file_name=url_to_filename(content))
                content = Path(filename).read_text(encoding="utf-8")
            documents.extend(part.strip() for part in re.split(r"\n\s*\n|\n(?=#)", content)
                             if part.strip())
        embeddings = None
        if documents:
            from argparser import args
            embeddings = Embeddings(content=True, **({"device": "cpu"} if args.cpu or args.directml is not None else {}))
            embeddings.index(documents)
        # Publish only a complete index; a failed download can be retried.
        self.embeddings = embeddings
        self.embeddings_hash = str(sources)

    def complete_text(self, messages):
        parts = []
        def collect(chunk):
            for choice in chunk.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    parts.append(content)
        self.llm.handle_chat_completions({"stream": True, "messages": messages}, collect)
        text = "".join(parts).strip()
        if not text:
            raise RuntimeError("The model returned no review text.")
        return text

    def inspect_image(self, messages, reviewer, original, gen_data):
        def progress(received, total, speed):
            worker.add_result(gen_data["task_id"], "download", (received, total, speed))
        try:
            self.load_base_model(reviewer, progress=progress)
            return self.complete_text(messages)
        finally:
            # The reviewer is a temporary helper, never the conversation owner.
            if reviewer != original or self.llm is None:
                self.load_base_model(original, progress=progress)

    def process(self, gen_data):

        if gen_data.get("unload"):
            self.unload()
            return {"unloaded": True}

        if not gen_data.get("load_only"):
            worker.add_result(
                gen_data["task_id"], "preview",
                gen_data["history"] + [{"role": "assistant", "content": "🤔"}]
            )

        vision_enabled = gen_data.get("vision_enabled", False)
        original_model = gen_data.get("model") or settings.default_settings.get("llama_localfile") or DEFAULT_MODEL
        review_model = original_model if model_has_vision(original_model) else gen_data.get("vision_model")
        if vision_enabled and not review_model:
            return "Select a vision model below Enable Model Vision to inspect generated images."

        if self.llm is None or self.model_settings != self.current_model_settings(gen_data.get("model")):
            def download_progress(received, total, speed):
                worker.add_result(gen_data["task_id"], "download", (received, total, speed))
            self.load_base_model(gen_data.get("model"),
                progress=download_progress)

        if gen_data.get("load_only"):
            return {"ready": True}

        embed = json.loads(gen_data.get("embed") or "[]")
        if self.embeddings_hash != str(embed) or (embed and self.embeddings is None):
            self.index_sources(embed)

        system_prompt = gen_data["system"]

        h = gen_data["history"]

        if self.embeddings:
            q = h[-1]["content"]
            context = "This some context that will help you answer the question:\n"
            for data in self.embeddings.search(q, limit=3):
                #if data["score"] >= 0.5:
                context += data["text"] + "\n\n"
            system_prompt += context

        if settings.default_settings.get("enable_llm_tools", False):
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "generate_image",
                        "description": "Create and show an image using the app's selected image model.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string", "description": "The prompt for the image"},
                            },
                            "required": ["prompt"],
                        },
                    },
                },
            ]
            tool_prompt = (
                "\nYou can create images with the built-in generate_image tool. "
                "When the user asks you to draw, create, or generate a picture, illustration, or wallpaper, "
                "write a descriptive visual prompt and call generate_image. The user does not need to name the tool. "
                "A direct request to generate is authorization: call the tool without asking for confirmation. "
                "Call it directly rather than describing a tool call or claiming you cannot make images. "
                "The app shows the resulting image in chat. Do not mention the internal tool name in your reply. "
                "If the user asks only for a written image prompt or advice, answer in text without generating.\n"
            )
        else:
            tools = None
            tool_prompt = ""
        if vision_enabled:
            tool_prompt += ("\nA previously generated image or visual observations may be supplied for review. "
                "Use the image or reviewer observations to correct the prompt. "
                "If the user asks to generate or regenerate, including 'generate a new image based on feedback', "
                "revise the prompt and call the available image tool immediately. This request already approves generation; "
                "do not ask for another approval or stop at a written prompt. "
                "If the user asks only for feedback, edits to the prompt, or advice without requesting generation, "
                "provide the revised prompt and wait for their next instruction. "
                "Review feedback alone never authorizes another generation. "
                "Do not claim to see an image unless one is attached.\n")
        chat = []
        if system_prompt or tool_prompt:
            chat.append({"role": "system", "content": system_prompt + tool_prompt})
        # Zero disables older history, but must still send the current message.
        history_len = max(1, int(settings.default_settings.get("llm_chat_history", 7)))
        history_len = -min(history_len, len(h))

        def clean_content(text):
            return re.sub('!\\[Image\\]\\([^(]*\\)', '', text) # Remove Image-markdown from LLM input
        for idx in range(history_len, 0):
            c = json.loads(json.dumps(h[idx])) # Thread safe Deep copy
            if isinstance(c['content'], list) and all(item.get('type') == 'text' for item in c['content']):
                c['content'] = '\n'.join(item.get('text', '') for item in c['content'])
            if isinstance(c['content'], str):
                c['content'] = clean_content(c['content'])
                if c['role'] == 'assistant' and c['content'].startswith('<think>'):
                    reasoning, separator, content = c['content'][7:].partition('</think>')
                    if separator:
                        c.update(reasoning_content=reasoning.strip(), content=content.lstrip())
            else: 
                try:
                    c['content'][0]['text'] = clean_content(c['content'][0]['text'])
                except Exception as e:
                    print(f"LLM: ({e}): {c}")
            chat.append(c)

        if vision_enabled:
            attach_image_feedback(chat, h)

        if vision_enabled and review_model == original_model and has_generated_image(h):
            attach_latest_image(chat, h)

        print(f"Thinking...")
        with TimeIt("LLM thinking"):
            result = []

            result = {
                "text": "",
                "reasoning": "",
                "finish_reason": None,
                "tool": {
                    "function": None,
                    "arguments": "",
                }
            }

            def response_text():
                if result['reasoning']:
                    return f"<think>{result['reasoning']}</think>\n\n{result['text']}"
                return result['text']

            def callback(chunk):
                if len(chunk.get('choices', [])) == 0:
                    return
                result['finish_reason'] = chunk['choices'][0].get('finish_reason') or result['finish_reason']
                try:
                    delta = chunk['choices'][0]['delta']
                except:
                    print(f"ERROR: No delta? {chunk}")
                    return

                result['reasoning'] += delta.get('reasoning_content') or ''
                result['text'] += delta.get('content') or ''
                if delta.get('reasoning_content') or delta.get('content'):
                    worker.add_result(
                        gen_data["task_id"], "preview",
                        gen_data["history"] + [{"role": "assistant", "content": response_text()}]
                    )

                if 'tool_calls' in delta:
                    tool_call = next((call.get('function', {}) for call in delta['tool_calls']
                                      if call.get('index', 0) == 0), {})
                    if 'name' in tool_call:
                        result['tool']['function'] = tool_call['name']
                    result['tool']['arguments'] += tool_call.get('arguments', '')

            self.llm.handle_chat_completions(
                {
                    "stream": True,
                    "messages": chat,
                    "tools": tools,
                },
                lambda d: callback(d),
            )

            if result['finish_reason'] == 'length':
                result['text'] += "\n\n[Response reached the token limit. Increase n_predict or context size in Settings.]"
            elif not result['text'].strip() and result['tool']['function'] is None:
                result['text'] = "[The model returned no answer. Try rephrasing your message.]"
            text = response_text()

            if settings.default_settings.get("enable_llm_tools", False) and result['tool']['function'] is not None:
                try:
                    task_id = -1

                    args = json.loads(result['tool']['arguments'])
                    if result['tool']['function'] != 'generate_image':
                        raise ValueError("Unknown image tool.")
                    prompt = args['prompt']

                    tmp_data = {
                        'task_type': "tool_call",
                        'task_id': task_id,
                        'silent': True,
                        'prompt': prompt,
                        'negative': "",
                        'loras': [
                            ("", f"{settings.default_settings.get('lora_1_weight', 1.0)} - {settings.default_settings.get('lora_1_model', 'None')}"),
                            ("", f"{settings.default_settings.get('lora_2_weight', 1.0)} - {settings.default_settings.get('lora_2_model', 'None')}"),
                            ("", f"{settings.default_settings.get('lora_3_weight', 1.0)} - {settings.default_settings.get('lora_3_model', 'None')}"),
                            ("", f"{settings.default_settings.get('lora_4_weight', 1.0)} - {settings.default_settings.get('lora_4_model', 'None')}"),
                            ("", f"{settings.default_settings.get('lora_5_weight', 1.0)} - {settings.default_settings.get('lora_5_model', 'None')}"),
                        ],
                        'style_selection': settings.default_settings['style'],
                        'seed': -1,
                        'base_model_name': settings.default_settings['base_model'],
                        'performance_selection': settings.default_settings['performance'],
                        'aspect_ratios_selection': settings.default_settings["resolution"],
                        'cn_selection': None,
                        'cn_type': None,
                        'silent': True,
                        'image_number': 1,
                        'generate_forever': False,
                    }
                    tmp_data.update(gen_data.get("image_settings", {}))

                    import shared
                    checkpoint = shared.models.get_file("checkpoints", tmp_data["base_model_name"])
                    if checkpoint is None or not checkpoint.is_file():
                        raise ValueError("Select an available checkpoint in Main before generating an image.")
                    review_enabled = vision_enabled
                    has_vision = bool(review_model)
                    self.unload()

                    info_txt = "(Generating image...)"
                    tmp_text = text + "\n" + info_txt

                    worker.add_result(
                        gen_data["task_id"],
                        "preview",
                        gen_data["history"] + [{"role": "assistant", "content": tmp_text}]
                    )

                    tmp_data["_chat_output"] = True
                    results = worker._process(tmp_data.copy())
                    if not results:
                        raise RuntimeError("Image generation produced no image. Check the checkpoint and application log.")
                    file = results[0]
                    filename = str(file.relative_to(file.cwd()).as_posix())
                    url = "gradio_api/file=" + re.sub(r'[^/]+/\.\./', '', filename)
                    remember_image(url, file)
                    markdown = f"\n*{prompt}*\n\n![Image]({url})\n"

                    text += "\n" + markdown
                    if review_enabled:
                        if not has_vision:
                            text += "\nImage review requires a vision model and its matching projector in Chatbot settings."
                        else:
                            worker.add_result(gen_data["task_id"], "preview",
                                h + [{"role": "assistant", "content": text + "\nReviewing image..."}])
                            try:
                                # Release diffusion VRAM before loading the vision model again.
                                from comfy import model_management
                                model_management.unload_all_models()
                                shared.state["pipeline"] = self
                                request = next((message["content"] for message in reversed(h)
                                                if message["role"] == "user"), prompt)
                                review_system = gen_data.get("system", "") if gen_data.get("vision_include_system", False) else ""
                                observations = self.inspect_image(review_messages(file, request, prompt, review_system),
                                    review_model, original_model, gen_data)
                                text += "\n\n**Vision model feedback:**\n\n" + observations
                            except Exception as error:
                                text += f"\n\nImage review failed: {error}"


                except Exception as e:
                    import traceback
                    print(f"ERROR:")
                    traceback.print_exc()
                    text += f"\n\nImage generation failed: {e}"

        return text
