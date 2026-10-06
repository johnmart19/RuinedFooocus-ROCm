import requests
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
from modules.config_io import save_json

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

                    from modules.model_previews import SUFFIXES, VERSION, is_warning, valid_preview
                    suffixes = list(SUFFIXES)
                    has_preview = False
                    for suffix in suffixes:
                        thumbcheck = cache_file.with_suffix(suffix)
                        if thumbcheck.is_file() and valid_preview(thumbcheck) and not is_warning(thumbcheck):
                            # Retain valid artwork if its online refresh fails.
                            has_preview = True
                            break

                    # Rebuild legacy low-resolution/animated network thumbnails
                    # once, preserving the old valid file if fetching fails.
                    marker = cache_file.with_suffix(".preview.json")
                    try:
                        preview_state = json.loads(marker.read_text())
                        preview_state = preview_state if isinstance(preview_state, dict) else {}
                    except (OSError, ValueError):
                        preview_state = {}
                    real_artwork = any(isinstance(item, dict) and item.get("url") and
                                       "cdn-thumbnails.huggingface.co/social-thumbnails/" not in item["url"]
                                       for item in (model_data.get("images") or []))
                    replace_generated = preview_state.get("generated") and real_artwork
                    if has_preview and not self.offline and (preview_state.get("version") != VERSION or replace_generated) and model_data.get("images"):
                        from PIL import Image
                        with Image.open(thumbcheck) as image:
                            legacy = replace_generated or max(image.size) <= 166 or getattr(image, "n_frames", 1) > 1
                        local_artwork = any(path.with_suffix(suffix).is_file() or path.with_suffix(".preview" + suffix).is_file() for suffix in suffixes)
                        if legacy and not local_artwork:
                            self.get_image(model_data, cache_file)

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
                        if not any(valid_preview(cache_file.with_suffix(suffix)) and not is_warning(cache_file.with_suffix(suffix)) for suffix in suffixes):
                            from modules.model_sources import huggingface_preview
                            from modules.model_identity import checkpoint_hashes
                            hash = checkpoint_hashes(model_data).get("SHA256") or self.model_sha256(path)
                            metadata = self.read_safetensors_header(path).get("__metadata__", {})
                            source = (f"https://huggingface.co/{model_data['hf_repo_id']}"
                                      if model_data.get("hf_repo_id") else metadata.get("modelspec.source", ""))
                            fallback = huggingface_preview(path, hash, source)
                            if fallback:
                                model_data = model_data | fallback
                                save_json(cache_file.with_suffix(".json"), model_data)
                                self.get_image(model_data, thumbcheck)
                            else:
                                print(f"No published artwork found for {path.name}; add a local .preview image to supply artwork.")
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
                            from modules.model_identity import checkpoint_hashes
                            self.model_hash[model_type][checkpoint_hashes(model_data)["SHA256"]] = str(path)
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
        filename = Path(filename)
        try:
            stat = filename.stat()
        except OSError as error:
            print(f"model_sha256(): Failed reading {filename}: {error}")
            return None
        signature = [str(filename.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        key = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
        cache = self.cache_paths["checkpoints"].parent / "hashes" / (key + ".json")
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if (isinstance(cached, dict) and cached.get("signature") == signature
                    and isinstance(cached.get("sha256"), str)
                    and len(cached["sha256"]) == 64
                    and all(char in "0123456789ABCDEF" for char in cached["sha256"])):
                return cached["sha256"]
        except (OSError, ValueError):
            pass
        print(t("Hashing {filename}", mapping={'filename': filename}))
        blksize = 1024 * 1024
        hash_sha256 = hashlib.sha256()
        try:
            with open(filename, 'rb') as f:
                for chunk in iter(lambda: f.read(blksize), b""):
                    hash_sha256.update(chunk)
            f.close()
            ret = hash_sha256.hexdigest().upper()
            # Cache locally, never beside the potentially read-only model. Only
            # publish a hash if the file remained unchanged while being read.
            after = filename.stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                try:
                    save_json(cache, {"signature": signature, "sha256": ret})
                except OSError as error:
                    print(f"Could not cache model hash: {error}")
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

        stat = path.stat()
        signature = [str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        previous_signature = data.get("_rf_file_signature")
        if previous_signature is not None and previous_signature != signature:
            # A checkpoint replaced under the same name needs fresh identity.
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

        # Record the actual local file hash separately from all release artifacts.
        # model_sha256 reuses the file-signature cache; never hash on UI selection.
        if fetch and not self.offline and not data.get("_rf_local_sha256"):
            local_hash = self.model_sha256(path)
            if local_hash:
                data["_rf_local_sha256"] = local_hash
                save_json(json_path, data)

        # Retry incomplete metadata during background refresh, never on UI selection.
        needs_metadata = not (data.get("id") or data.get("hf_repo_id")) or not data.get("images")
        if fetch and not self.offline and needs_metadata and (not local_metadata or data.get("id")):
            from modules.model_identity import checkpoint_hashes
            hash = checkpoint_hashes(data).get("SHA256")
            if not hash and not data.get("id"):
                hash = self.model_sha256(path)
            if hash or data.get("id"):
                from modules.model_sources import civitai_metadata
                remote = (civitai_metadata(f"model-versions/{data['id']}") if data.get("id")
                          else self.search_civitai_with_hash(hash))
                if remote:
                    data = data | remote
                if hash and not data.get("id") and not data.get("hf_repo_id"):
                    # Identity discovery must not depend on needing new artwork.
                    from modules.model_sources import huggingface_preview
                    header = self.read_safetensors_header(path).get("__metadata__", {})
                    source = header.get("modelspec.source", "") if isinstance(header, dict) else ""
                    fallback = huggingface_preview(path, hash, source)
                    if fallback:
                        data = data | fallback
                if hash:
                    # An existing files entry may lack hashes. setdefault on the
                    # outer key alone loses the computed hash and rehashes on
                    # every background refresh when lookup returns no metadata.
                    data["_rf_local_sha256"] = hash
                data["_rf_file_signature"] = signature
                save_json(json_path, data)

        if fetch and data and data.get("_rf_file_signature") != signature:
            # Stamp legacy metadata once without rehashing an unchanged model.
            data["_rf_file_signature"] = signature
            save_json(json_path, data)

        from modules.video_settings import VIDEO_FPS
        base = self.detect_checkpoint_base(path)
        if base and (base in VIDEO_FPS or base in ("Flux2Klein4B", "Flux2Klein9B")
                     or data.get("baseModel") in (None, "", "Unknown", "Other", "Lumina2")):
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
                    family = {"wan": "Wan Video", "hyvid": "Hunyuan Video",
                              "minimax_h3": "MiniMaxH3"}.get(architecture)
                    if family:
                        return family
                    from modules.model_architecture import detect_tensor_architecture
                    return detect_tensor_architecture(tensors)
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
        for anima_prefix in (prefix, "net.", ""):
            if (anima_prefix + "blocks.0.mlp.layer1.weight" in tensors
                    and anima_prefix + "llm_adapter.blocks.0.cross_attn.q_proj.weight" in tensors):
                return "Anima"
        if (prefix + "double_blocks.0.img_attn.qkv.weight" in tensors
                and prefix + "single_blocks.0.linear1.weight" in tensors
                and tensors.get(prefix + "img_in.weight", {}).get("shape") == [3072, 64]):
            return "Flux.1 D" if prefix + "guidance_in.in_layer.weight" in tensors else "Flux.1 S"
        try:
            from modules.model_architecture import detect_tensor_architecture
            return detect_tensor_architecture(tensors)
        except ImportError:
            return None  # Backend is not installed during first-run setup.

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
        from modules.model_previews import write_preview

        if model.get("baseModel") == "Merge":
            return False
        caption = Path(path).stem if "caption" in settings.default_settings.get("model_preview", "").split(",") else ""
        previews = model.get("images") or []
        if not isinstance(previews, list):
            return False
        for preview in previews[:5]:
            if not isinstance(preview, dict) or not preview.get("url") or "cdn-thumbnails.huggingface.co/social-thumbnails/" in preview["url"]:
                continue
            try:
                with self.session.get(preview["url"], timeout=(5, 20), stream=True) as response:
                    response.raise_for_status()
                    data = bytearray()
                    started = time.monotonic()
                    for chunk in response.iter_content(65536):
                        if len(data) + len(chunk) > 64 * 1024 * 1024 or time.monotonic() - started > 60:
                            raise ValueError("Preview exceeds the download size or time limit")
                        data.extend(chunk)
                write_preview(bytes(data), path, caption, video=preview.get("type") == "video")
                return True
            except Exception as error:
                print(f"Could not update preview for {Path(path).stem}: {error}")
        return False
