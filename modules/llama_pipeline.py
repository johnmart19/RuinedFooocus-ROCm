import re
import gc
from modules.llama_server import Server as NativeServer
from modules.llama_models import DEFAULT_MODEL, resolve_model
from modules.llama_vision import (
    projector_path, review_messages, remember_image, attach_latest_image,
    model_has_vision, attach_image_feedback, transient_vision_messages, image_tool_history,
)
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

def run_llama(system_file, prompt):
        name = None
        sys_pat = r"system:.*\n\n"
        system = re.match(sys_pat, prompt, flags=re.M|re.I)
        if system is not None: # Llama system-prompt provided in the ui-prompt
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
            print(f"# System:\n{system_prompt.strip()}\n")
            print(f"# User:\n{prompt.strip()}\n")
            print(f"# {name}: (Thinking...)")
            try:

                ret = llama.llm.handle_chat_completions(
                    {
                        "max_tokens": settings.default_settings.get("llm_hp_max_tokens", 256),
                        "messages": [{"role": "system", "content": system_prompt},
                                     {"role": "user", "content": prompt}],
                    }
                )
                res = ret['choices'][0]['text']
            except Exception as e:
                print(f"LLAMA ERROR: {e}")
                res = prompt

            print(f"{res.strip()}\n")

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
                runtime_settings = settings.default_settings.copy()
                runtime_settings["llama_mmproj"] = str(projector) if projector else "None"
                if args.directml is not None and runtime_settings.get("llama_backend") != "CPU":
                    runtime_settings["llama_backend"] = "Vulkan"
                # llama.cpp probes its own devices; a CPU PyTorch wheel does not
                # mean Vulkan is unavailable (notably in WSL and DirectML setups).
                gpu = not args.cpu and runtime_settings.get("llama_backend", "Auto") != "CPU"
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

    def index_source(self, source):
        if self.embeddings == None:
            self.embeddings = Embeddings(content=True)
            self.embeddings.initindex(reindex=True)

        match source[0]:

            case "url":
                print(f"Read {source[1]}")
                filename = load_file_from_url(
                    source[1],
                    model_dir="cache/embeds",
                    progress=True,
                    file_name=url_to_filename(source[1]),
                )
                file = open(filename, "r", encoding='utf-8')
                data = file.read()
                file.close()

                if source[1].endswith(".md"):
                    data = data.split("\n#")
                elif source[1].endswith(".txt"):
                    data = data.split("\n\n")

            case "text":
                data = source[1]

            case _:
                print("WARNING: Unknown embedding type {source[0]}")
                return

        if data:
            self.embeddings.upsert(data)


    def process(self, gen_data):

        worker.add_result(
            gen_data["task_id"],
            "preview",
            gen_data["history"] + [{"role": "assistant", "content": "🤔"}]
        )

        if self.llm == None:
            self.load_base_model()

        # load embeds?
        # FIXME should dump the entire gen_data["embed"] to index_source() and have it sort it out
        embed = json.loads(gen_data['embed'])
        if self.embeddings_hash != str(embed):
            self.embeddings_hash = str(embed)
            self.embeddings = None
        if embed:
            if not self.embeddings: # If chatbot has embeddings to index, check that we have them.
                for source in embed:
                    self.index_source(source)
        else:
            self.embeddings = None

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
                        "description": "Generates an image from a prompt.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string", "description": "The prompt for the image"},
                            },
                        },
                    },
                },
            ]
            tool_prompt = "\nUse the tool when you intend to generate an image. You must make sure you use the correct format. The image will be shown to the user.\n"
        else:
            tools = None
            tool_prompt = ""
        chat = [{"role": "system", "content": system_prompt + tool_prompt}]
        history_len = settings.default_settings.get("llm_chat_history", 7) # Keep the some of the last messages in the discussion.
        history_len = -history_len if len(h) > history_len else -len(h)

        def clean_content(text):
            return re.sub('!\\[Image\\]\\([^(]*\\)', '', text) # Remove Image-markdown from LLM input
        for idx in range(history_len, 0):
            c = json.loads(json.dumps(h[idx])) # Thread safe Deep copy
            if isinstance(c['content'], str):
                c['content'] = clean_content(c['content'])
            else: 
                try:
                    c['content'][0]['text'] = clean_content(c['content'][0]['text'])
                except Exception as e:
                    print(f"LLM: ({e}): {c}")
            chat.append(c)

        print(f"Thinking...")
        with TimeIt("LLM thinking"):
            result = []

            result = {
                "text": "",
                "tool": {
                    "function": None,
                    "arguments": "",
                }
            }

            def callback(chunk):
                if len(chunk.get('choices', [])) == 0:
                    return
                try:
                    delta = chunk['choices'][0]['delta']
                except:
                    print(f"ERROR: No delta? {chunk}")
                    return

                if 'content' in delta:
                    text = delta['content']

                    if text is not None:
                        #print(text, end="")
                        result['text'] += text
                        worker.add_result(
                            gen_data["task_id"],
                            "preview",
                            gen_data["history"] + [{"role": "assistant", "content": result['text']}]
                        )

                if 'tool_calls' in delta:
                    tool_call = delta['tool_calls'][0]['function'] # Simply assume we only have a single tool call
                    if 'name' in tool_call:
                        result['tool']['function'] = tool_call['name']
                    result['tool']['arguments'] += tool_call['arguments']

            self.llm.handle_chat_completions(
                {
                    "stream": True,
                    "messages": chat,
                    "tools": tools,
                },
                lambda d: callback(d),
            )

            text = result['text']

            call = None
            tool_error = f"![Error](gradio_api/file=html/error.png)"
            if settings.default_settings.get("enable_llm_tools", False) and result['tool']['function'] is not None:
                try:
                    task_id = -1

                    args = json.loads(result['tool']['arguments'])
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
                    }

                    # unload llm model from memory?
                    # TODO: make this selectable for people with more ram/vram that is socialy acceptable
                    del self.llm
                    self.llm = None

                    info_txt = "(Generating image...)"
                    tmp_text = text + "\n" + info_txt

                    worker.add_result(
                        gen_data["task_id"],
                        "preview",
                        gen_data["history"] + [{"role": "assistant", "content": tmp_text}]
                    )

                    results = worker._process(tmp_data.copy())
                    file = results[0]
                    filename = str(file.relative_to(file.cwd()).as_posix())
                    url = "gradio_api/file=" + re.sub(r'[^/]+/\.\./', '', filename)
                    markdown = f"\n*{prompt}*\n\n![Image]({url})\n"

                    text += "\n" + markdown

                except Exception as e:
                    import traceback
                    print(f"ERROR:")
                    traceback.print_exc()
                    text += f"Error: {e}\n\n"
                    text += f"Call: {call}\n\n"
                    text += "Looks like I made a mistake. I really need to make sure I use the correct format. Do you want me to try again?"

        return text
