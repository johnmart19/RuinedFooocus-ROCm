import requests
import filecmp
import base64
import hashlib
import shutil
import os
import cv2
import json
import threading
import time
from pathlib import Path
import numpy as np

from shared import translate as t

class Models:
    def civit_update_worker(self, model_type, folder_paths):
        if model_type in self.civit_workers:
            return
        self.civit_workers.append(str(model_type))
        try:
            self.cache_paths[model_type].mkdir(parents=True, exist_ok=True)
            self._update_models(model_type, folder_paths)
        except Exception as error:
            print(f"Model update failed for {model_type}: {error}")
        finally:
            from shared import shared_cache
            for key in list(shared_cache):
                if isinstance(key, Path) and key.parent == self.cache_paths[model_type]:
                    shared_cache.pop(key, None)
            self.ready[model_type] = True
            self.civit_workers.remove(str(model_type))
            self.revision += 1

    def _update_models(self, model_type, folder_paths):
        from shared import path_manager

        self.ready[model_type] = False
        updated = 0

        # Quick list
        self.names[model_type] = []
        for folder in folder_paths:
            for path in folder.rglob("*"):
                if path.suffix.lower() in self.EXTENSIONS:
                    # Add to model names
                    self.names[model_type].append(str(path.relative_to(folder)))

        # Return a sorted list, prepend names with 0 if they are in a folder or 1
        # if it is a plain file. This will sort folders above files in the dropdown
        self.names[model_type] = sorted(
            self.names[model_type],
            key=lambda x: (
                f"0{x.casefold()}"
                if not str(Path(x).parent) == "."
                else f"1{x.casefold()}"
            ),
        )
        self.ready[model_type] = True

        if model_type == "inbox" and self.names["inbox"]:
            checkpoints = path_manager.model_paths["modelfile_path"]
            checkpoints = checkpoints[0] if isinstance(checkpoints, list) else checkpoints
            loras = path_manager.model_paths["lorafile_path"]
            loras = loras[0] if isinstance(loras, list) else loras
            folders = {
                "DoRA": (loras, self.cache_paths["loras"]),
                "LoCon": (loras, self.cache_paths["loras"]),
                "LORA": (loras, self.cache_paths["loras"]),
                "Checkpoint": (checkpoints, self.cache_paths["checkpoints"]),
            }

        # Go through and check previews
        for folder in folder_paths:
            for path in folder.rglob("*"):
                if path.suffix.lower() in self.EXTENSIONS:
                    # get file name, add cache path change suffix
                    cache_file = Path(self.cache_paths[model_type] / path.name)
                    model_data = self.get_models_by_path(model_type, path, fetch=True)

                    suffixes = [".jpeg", ".jpg", ".png", ".gif"]
                    has_preview = False
                    for suffix in suffixes:
                        thumbcheck = cache_file.with_suffix(suffix)
                        if thumbcheck.is_file():
                            # Older versions cached the warning image as a real preview.
                            if filecmp.cmp(thumbcheck, "html/warning.jpeg", shallow=False):
                                thumbcheck.unlink()
                            else:
                                has_preview = True
                                break

                    if not has_preview:
                        for suffix in suffixes:
                            for local in (path.with_suffix(suffix), path.with_suffix(".preview" + suffix)):
                                if local.is_file():
                                    shutil.copyfile(local, cache_file.with_suffix(suffix))
                                    has_preview = True
                                    break
                            if has_preview:
                                break

                    if not has_preview:
                        has_preview = self.copy_embedded_preview(path, cache_file)

                    if not has_preview and not self.offline:
                        self.get_image(model_data, thumbcheck)
                        if not any(cache_file.with_suffix(suffix).is_file() for suffix in suffixes):
                            from modules.model_sources import huggingface_preview
                            files = model_data.get("files") or [{}]
                            hash = files[0].get("hashes", {}).get("SHA256") or self.model_sha256(path)
                            metadata = self.read_safetensors_header(path).get("__metadata__", {})
                            source = (f"https://huggingface.co/{model_data['hf_repo_id']}"
                                      if model_data.get("hf_repo_id") else metadata.get("modelspec.source", ""))
                            fallback = huggingface_preview(path, hash, source)
                            if fallback:
                                model_data = model_data | fallback
                                cache_file.with_suffix(".json").write_text(json.dumps(model_data, indent=2), encoding="utf-8")
                                self.get_image(model_data, thumbcheck)
                            else:
                                print(f"No matching downloadable artwork found for {path.name}.")
                        updated += 1
                        time.sleep(1)

                    txtcheck = cache_file.with_suffix(".txt")
                    if model_type == "loras" and model_data.get("trainedWords") and not txtcheck.exists():
                        print(
                            t(
                                "Get LoRA keywords for {name} ({base} - {type})",
                                mapping= {
                                    'name': Path(path).name,
                                    'base': self.get_model_base(model_data),
                                    'type': self.get_model_type(model_data)
                                }
                            )
                        )
                        keywords = self.get_keywords(model_data)
                        with open(txtcheck, "w") as f:
                            f.write(", ".join(keywords))
                        updated += 1

                    if model_type == "inbox" and not self.offline:
                        name = str(path.relative_to(folder))
                        filename = path
                        baseModel = self.get_model_base(model_data)
                        destination, cache = folders.get(self.get_model_type(model_data), [None, None])
                        if destination is None or baseModel is None:
                            print(t('Skipping {name} not sure what {type} is.', mapping={'name': str(name), 'type': self.get_model_type(model_data)}))
                            updated += 1
                            continue
                        # Never overwrite an existing model or its cached metadata.
                        dest = Path(destination) / baseModel / name
                        if not dest.resolve().is_relative_to(Path(destination).resolve()):
                            print(f"WARNING: Invalid destination for {name}. Leaving it in Inbox.")
                            continue
                        cache_file = self.cache_paths[model_type] / path.name
                        suffixes = [".json", ".txt", ".jpeg", ".jpg", ".png", ".gif"]
                        cached = [(cache_file.with_suffix(suffix),
                                   (Path(cache) / path.name).with_suffix(suffix))
                                  for suffix in suffixes]
                        if dest.exists() or any(target.exists() for _, target in cached):
                            print(f"WARNING: {name} already exists at the destination. Leaving it in Inbox.")
                            continue
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        Path(cache).mkdir(parents=True, exist_ok=True)
                        shutil.move(filename, dest)
                        for source, target in cached:
                            if source.is_file():
                                shutil.move(source, target)
                        print(t("Moved {name} to {dest}", mapping={'name': name, 'dest': dest}))
                        updated += 1

                    else:
                        # If this isn't the inbox, store some info about the "live" model
                        try:
                            self.model_hash[model_type][model_data["files"][0]["hashes"]["SHA256"]] = str(path)
                        except:
                            # Some models seem to not have sha256 hashes, just ignore those.
                            pass

        if updated > 0:
            print(t("CivitAI update for {type} done.", mapping={'type': model_type}))

    def get_names(self, model_type):
        while not self.ready[model_type]:
            # Wait until we have read all the filenames
            time.sleep(0.2)
        return self.names[model_type]

    def get_file(self, model_type, name):
        # Search the folders for the model
        try:
            for folder in self.model_dirs[model_type]:
                file = Path(folder) / name
                if file.is_file():
                    return file
        except:
            pass
        return None

    def update_all_models(self):
        for model_type in ["checkpoints", "loras", "inbox"]:
            threading.Thread(
                target=self.civit_update_worker,
                args=(
                    model_type,
                    self.model_dirs[model_type],
                ),
                daemon=True,
            ).start()



    def __init__(self, offline=False):
        from shared import path_manager, settings

        self.offline = offline
        self.civit_workers = []
        self.revision = 0

        self.ready = {
            "checkpoints": False,
            "loras": False,
            "inbox": False,
        }
        self.names = {
            "checkpoints": [],
            "loras": [],
            "inbox": [],
        }
        self.model_hash = {
            "checkpoints": {},
            "loras": {},
            "inbox": {},
        }
        checkpoints = path_manager.model_paths["modelfile_path"]
        checkpoints = checkpoints if isinstance(checkpoints, list) else [checkpoints]
        loras = path_manager.model_paths["lorafile_path"]
        loras = loras if isinstance(loras, list) else [loras]
        inbox = path_manager.model_paths["inbox_path"]
        inbox = inbox if isinstance(inbox, list) else [inbox]
        self.model_dirs = {
            "checkpoints": checkpoints,
            "loras": loras,
            "inbox": inbox,
        }
        self.cache_paths = {
            "checkpoints": Path(path_manager.model_paths["cache_path"] / "checkpoints"),
            "loras": Path(path_manager.model_paths["cache_path"] / "loras"),
            "inbox": Path(path_manager.model_paths["cache_path"] / "inbox"),
        }

        self.base_url = "https://civitai.com/api/v1/"
        self.headers = {"Content-Type": "application/json"}
        self.session = requests.Session()
        self.EXTENSIONS = [".pth", ".ckpt", ".bin", ".safetensors", ".gguf"]

        self.update_all_models()

    def get_file_from_hash(self, model_type, hash):
        return self.model_hash.get(model_type, {}).get(hash, None)

    def get_file_from_name(self, model_type, model_name):
        for folder in self.model_dirs[model_type]:
            path = Path(folder) / model_name
            if path.is_file():
                return path
        return None

    def model_sha256(self, filename):
        print(t("Hashing {filename}", mapping={'filename': filename}))
        blksize = 1024 * 1024
        hash_sha256 = hashlib.sha256()
        try:
            with open(filename, 'rb') as f:
                for chunk in iter(lambda: f.read(blksize), b""):
                    hash_sha256.update(chunk)
            f.close()
            ret = hash_sha256.hexdigest().upper()
        except Exception as e:
            print(f"model_sha256(): Failed reading {filename}")
            print(f"Error: {e}")
            ret = None
        return ret

    def search_civitai_with_hash(self, hash):
        if self.offline:
            return None
        from modules.model_sources import civitai_metadata
        return civitai_metadata(f"model-versions/by-hash/{hash}") or {
            "files": [{"hashes": {"SHA256": hash}}]
        }

    def get_models_by_path(self, model_type, path, fetch=False):
        path = Path(path)
        if not path.is_file():
            path = self.get_file(model_type, path)
        if path is None:
            return {}
        if path.suffix == ".merge":
            return {"baseModel": "Merge"}

        json_path = (self.cache_paths[model_type] / path.name).with_suffix(".json")
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}

        # Optimized checkpoints have a different hash. Keep downloaded metadata
        # beside them as <checkpoint>.civitai.info to retain the original identity.
        local_metadata = bool(data.get("rf_source"))
        if not data.get("id"):
            try:
                local = json.loads(path.with_suffix(".civitai.info").read_text(encoding="utf-8"))
                if isinstance(local, dict) and local.get("baseModel"):
                    data = local
                    local_metadata = True
            except (OSError, ValueError):
                pass

        # Retry incomplete metadata during background refresh, never on UI selection.
        needs_metadata = not (data.get("id") or data.get("hf_repo_id")) or not data.get("images")
        if fetch and not self.offline and needs_metadata and (not local_metadata or data.get("id")):
            files = data.get("files") or [{}]
            hash = files[0].get("hashes", {}).get("SHA256")
            if not hash and not data.get("id"):
                hash = self.model_sha256(path)
            if hash or data.get("id"):
                from modules.model_sources import civitai_metadata
                remote = (civitai_metadata(f"model-versions/{data['id']}") if data.get("id")
                          else self.search_civitai_with_hash(hash))
                if remote:
                    data = data | remote
                if hash:
                    data.setdefault("files", [{"hashes": {"SHA256": hash}}])
                json_path.parent.mkdir(parents=True, exist_ok=True)
                json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        from modules.video_settings import VIDEO_FPS
        base = self.detect_checkpoint_base(path)
        if base and (base in VIDEO_FPS or data.get("baseModel") in (None, "", "Unknown")):
            data = data | {"baseModel": base}
        return data

    @staticmethod
    def read_safetensors_header(path):
        if path.suffix.lower() != ".safetensors":
            return {}
        try:
            with path.open("rb") as handle:
                size = int.from_bytes(handle.read(8), "little")
                if not 0 < size <= 16 * 1024 * 1024:
                    return {}
                header = json.loads(handle.read(size))
                return header if isinstance(header, dict) else {}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def detect_checkpoint_base(path):
        from modules.video_settings import detect_video_model
        if path.suffix.lower() == ".gguf":
            import gguf
            try:
                reader = gguf.GGUFReader(str(path))
                try:
                    tensors = {t.name: {"shape": list(reversed(t.shape.tolist()))} for t in reader.tensors}
                    video = detect_video_model(tensors)
                    if video:
                        return video
                    field = reader.get_field("general.architecture")
                    architecture = field.contents() if field else None
                    if architecture == "ltxv":
                        for tensor in reader.tensors:
                            if tensor.name.endswith("keyframes_abs_pos_embedding") and 4096 in tensor.shape:
                                return "LTXV 2.5"
                        if any(t.name.endswith("transformer_blocks.0.attn1.to_gate_logits.weight") for t in reader.tensors):
                            return "LTXV 2.3"
                        return "LTXV2" if any(t.name.startswith("audio_") for t in reader.tensors) else "LTXV"
                    return {"wan": "Wan Video", "hyvid": "Hunyuan Video",
                            "minimax_h3": "MiniMaxH3"}.get(architecture)
                finally:
                    reader.data._mmap.close()
            except (OSError, ValueError, IndexError):
                return None
        # Inspect tensor names, not the filename or optimizer's family label.
        tensors = Models.read_safetensors_header(path)
        video = detect_video_model(tensors)
        if video:
            return video
        if ("conditioner.embedders.1.model.text_projection" in tensors
                and "model.diffusion_model.input_blocks.0.0.weight" in tensors):
            return "SDXL 1.0"
        prefix = "model.diffusion_model."
        if (prefix + "double_blocks.0.img_attn.qkv.weight" in tensors
                and prefix + "single_blocks.0.linear1.weight" in tensors
                and tensors.get(prefix + "img_in.weight", {}).get("shape") == [3072, 64]):
            return "Flux.1 D" if prefix + "guidance_in.in_layer.weight" in tensors else "Flux.1 S"
        return None

    @staticmethod
    def copy_embedded_preview(model_path, cache_path):
        metadata = Models.read_safetensors_header(model_path).get("__metadata__", {})
        thumbnail = metadata.get("modelspec.thumbnail", "")
        if not isinstance(thumbnail, str) or not thumbnail.startswith("data:image/"):
            return False
        try:
            data = base64.b64decode(thumbnail.split(",", 1)[1], validate=True)
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                return cv2.imwrite(str(cache_path.with_suffix(".jpeg")), image)
        except (ValueError, IndexError, cv2.error):
            pass
        return False


    def get_model_path(self, model_type, name, hash=None, default=None):
        from shared import path_manager, settings

        # Look through folders for the filename
        filename = self.get_file(model_type, name)

        # Try looking for model using the hash
        if filename is None and hash is not None:
            filename = self.get_file_from_hash(model_type, hash)
            if filename is not None:
                print(f"INFO: Found {filename} from hash")

        # Download the selected catalog entry for checkpoints and LoRAs.
        # Do not replace a missing custom model with an unrelated default.
        if filename is None and model_type in ("checkpoints", "loras"):
            from argparser import args
            if not args.offline and os.environ.get("RF_OFFLINE") != "1":
                filename = path_manager.get_folder_file_path(model_type, name)
                if filename is not None:
                    threading.Thread(target=self.civit_update_worker,
                        args=(model_type, self.model_dirs[model_type]), daemon=True).start()

        # If we don't have a filename, get the default.
        if filename is None and default is not None:
            name = path_manager.get_folder_file_path(
                model_type,
                default,
            )
            filename = self.get_file(model_type, name)
            if filename is not None:
                print(f"INFO: Using {filename} (default)")

        if filename is None:
            print(f"Could not find file: {name}")
            if hash is not None and hash != "None":
                print(f"    SHA256: {hash}")
                data = self.search_civitai_with_hash(hash)
                if data is not None and data.get('modelId', None) is not None:
                    print(f"    Download link: https://civitai.com/models/{data.get('modelId')}")
                print(f"    Search here: https://civitaiarchive.com/sha256/{hash}")

        return filename


    def get_keywords(self, model):
        keywords = model.get("trainedWords", [""])
        return keywords

    def get_model_base(self, model):
        return (model or {}).get("baseModel", "Unknown")

    def get_model_type(self, model):
        res = (model or {}).get("model", None)
        if res is not None:
            res = res.get("type", "Unknown")
        else:
            res = "Unknown"
        return res

    def get_image(self, model, path):
        from shared import settings

        if "baseModel" in model and model["baseModel"] == "Merge":
            return

        import imageio.v3 as iio
        opts = settings.default_settings.get("model_preview", "").split(",")
        caption = "caption" in opts
        nogifzoom = "nogifzoom" in opts
        zoom = "zoom" in opts

        def make_thumbnail(image, text, zoom=False, caption=False):
            max = 166  # Max width or height

            if image is None:
                return None

            if zoom:
                oh = image.shape[0]
                ow = image.shape[1]
                scale = max / oh if oh > ow else max / ow
                image = cv2.resize(
                    image,
                    dsize=(int(ow * scale), int(oh * scale)),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            if caption:
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.35
                thickness = 1

                org = (3, 10)
                color = (25, 15, 11) # BGR
                image = cv2.putText(
                    image,
                    text,
                    org,
                    font,
                    fontScale,
                    color,
                    thickness*2,
                    cv2.LINE_AA
                )
                org = (3, 10)
                color = (255, 215, 185) # BGR
                image = cv2.putText(
                    image,
                    text,
                    org,
                    font,
                    fontScale,
                    color,
                    thickness,
                    cv2.LINE_AA
                )

            return image

        path = path.with_suffix(".jpeg")
        caption_text = f"{path.with_suffix('').name}"

        image_url = None
        for preview in model.get("images", [{}]):
            url = preview.get("url")
            format = preview.get("type")
            if url:
                print(t("Updating preview for {text}.", mapping={'text': caption_text}))
                image_url = url
                try:
                    response = self.session.get(image_url, timeout=(5, 20))
                except requests.exceptions.RequestException as error:
                    print(f"Preview download failed for {caption_text}: {error}")
                    continue
                if response.status_code != 200:
                    print(f"WARNING: get_image() for {caption_text} - {response.status_code} : {response.reason}")
                    continue
                image = np.asarray(bytearray(response.content), dtype="uint8") 
                out = make_thumbnail(cv2.imdecode(image, cv2.IMREAD_COLOR), caption_text, caption=caption, zoom=zoom)
                if out is not None:
                    out = cv2.imencode('.jpg', out)[1] 
                else:
                    out = response.content
                with open(path, "wb") as file:
                    file.write(out)

                try:
                    fps = iio.immeta(path).get("fps", False)
                except Exception as e:
                    print(f"WARNING: Could not decode preview for {caption_text}: {e}")
                    path.unlink(missing_ok=True)
                    continue
                if format == "video" and fps:
                    tmp_path = f"{path}.tmp"
                    shutil.move(path, tmp_path)
                    video = iio.imiter(tmp_path)
                    video_out = []
                    for i in video:
                        out = make_thumbnail(i, caption_text, caption=caption, zoom=not nogifzoom)
                        if out is None:
                            out = i
                        video_out.append(out)
                    iio.imwrite(
                        str(path.with_suffix(".gif")), video_out, fps=fps, loop=0
                    )
                    os.remove(tmp_path)
                break
