import os
import re
import random

from modules.sdxl_styles import apply_style, styles as available_styles, STYLE_ALIASES
from modules.prompt_scheduling import prompt_switch_per_step
from random_prompt.build_dynamic_prompt import (
    build_dynamic_prompt,
    build_dynamic_negative,
)
from shared import path_manager


def process_metadata(gen_data):
    from modules.image_metadata import read_metadata, render_values
    meta = read_metadata(gen_data['prompt'])
    if meta is not None:
        if not isinstance(meta['prompt'], str):
            raise ValueError('Image metadata needs a text prompt.')
        from modules.image_guidance import restore_guidance_metadata
        from shared import models
        checkpoint = meta.get('base_model_name', gen_data.get('base_model_name'))
        family = 'Unknown'
        if checkpoint:
            info = models.get_models_by_path('checkpoints', checkpoint, fetch=False)
            family = models.get_model_base(info)
            path = models.get_file('checkpoints', checkpoint)
            if path:
                family = models.detect_checkpoint_base(path) or family
        restored = restore_guidance_metadata(meta, family=family)
        gen_data.update(render_values(meta))
        gen_data['_metadata_prompt'] = True
        # Canonical validated flat/grouped guidance takes precedence. This also
        # restores the recorded effective Auto shift instead of a later default.
        gen_data.update(restored)
        gen_data["style_selection"] = []
        gen_data["performance_selection"] = "Own settings"
        if meta.get("width") and meta.get("height"):
            gen_data["width"] = gen_data["custom_width"] = meta.get("generation_width", meta["width"])
            gen_data["height"] = gen_data["custom_height"] = meta.get("generation_height", meta["height"])
            gen_data["aspect_ratios_selection"] = "Custom..."
        gen_data["automatic_upscale"] = meta.get("automatic_upscale", "Off")
        gen_data["automatic_upscale_target"] = "Generated image"

    if meta is not None and "steps" in meta:
        gen_data["custom_steps"] = int(meta["steps"])
    return gen_data


def get_promptlist(gen_data):
    from modules.image_metadata import read_metadata
    if gen_data.get('_metadata_prompt') or read_metadata(gen_data['prompt']) is not None:
        return [gen_data['prompt']]
    return gen_data["prompt"].split("---")


def process_wildcards(wildcard_text, directory=None, *, _stack=()):
    if directory is None:
        directory = path_manager.paths.get("path_wildcards", ["wildcards"])
    placeholders = re.findall(r"__([\w-]+?(?::(?:(?!__).)+?)?)__", wildcard_text)
    if not placeholders:
        return wildcard_text
    for placeholder in placeholders:
        if placeholder in _stack:
            raise ValueError("Recursive wildcard: " + " -> ".join((*_stack, placeholder)))

    placeholder_choices = {}  # Store random choices for each placeholder
    official_directory = "wildcards_official"
    directories = [directory] if isinstance(directory, (str, os.PathLike)) else list(directory)
    if official_directory not in directories:
        directories.append(official_directory)

    for placeholder in placeholders:
        # Skip onebuttonprompt wildcards for now; handled below
        if placeholder.startswith("onebutton"):
            continue

        elif placeholder not in placeholder_choices:
            found = False
            for dir in directories:
                for root, dirs, files in os.walk(dir):
                    if f"{placeholder}.txt" in files:
                        file_path = os.path.join(root, f"{placeholder}.txt")
                        with open(file_path, encoding="utf-8-sig") as f:
                            words = [
                                word.strip()
                                for word in f.read().splitlines()
                                if not word.startswith("#")
                            ]
                        if not words:
                            raise ValueError(f"Wildcard file has no choices: {file_path}")
                        placeholder_choices[placeholder] = words
                        found = True
                        break
                if found == True:
                    break

            if not found:
                print(
                    f"Error: Could not find file {placeholder}.txt in {directory} or its subdirectories."
                )
                placeholder_choices[placeholder] = None

    for placeholder in placeholders:
        random_choice = ""

        # Some one button prompt specials
        if placeholder.startswith("onebutton"):
            subjectoverride = ""
            placeholdersplit = placeholder.split(":", 1)
            if len(placeholdersplit) > 1:
                subjectoverride = placeholdersplit[1]

            if placeholder.startswith("onebuttonprompt"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    givensubject=subjectoverride,
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonsubject"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonhumanoid"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="human - all",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonmale"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="human - all",
                    gender="male",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonfemale"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="human - all",
                    gender="female",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonanimal"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="animal - all",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonobject"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="object",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonlandscape"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="landscape - all",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonconcept"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="concept - all",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonartist"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    onlyartists=True,
                    artists=subjectoverride or "all",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebutton1girl"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="human - all",
                    gender="female",
                    advancedprompting=False,
                    base_model="Anime Model",
                )
            elif placeholder.startswith("onebutton1boy"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="human - all",
                    gender="male",
                    advancedprompting=False,
                    base_model="Anime Model",
                )
            elif placeholder.startswith("onebuttonfurry"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="animal - all",
                    advancedprompting=False,
                    base_model="Anime Model",
                )
            # failover
            else:
                random_choice = build_dynamic_prompt(
                    insanitylevel=3,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    advancedprompting=False,
                    base_model="SDXL",
                )

        # Regular wildcards
        else:
            if placeholder_choices[placeholder] is None:
                continue
            random_choice = random.choice(placeholder_choices[placeholder])

        if "__" in random_choice:
            random_choice = process_wildcards(random_choice, directory, _stack=(*_stack, placeholder))
        wildcard_text = wildcard_text.replace(f"__{placeholder}__", random_choice, 1)

    return wildcard_text


def use_random_prompt(gen_data, prompt):
    from modules.image_metadata import read_metadata
    if gen_data.get('_metadata_prompt') or read_metadata(prompt) is not None:
        return False
    if "obp_assume_direct_control" not in gen_data:
        return False
    # An empty image-conditioned prompt is intentional, not a request for OBP.
    image_conditioned = ("input_image" in gen_data and gen_data["input_image"] is not None) or (
        "inpaint_toggle" in gen_data and gen_data["inpaint_toggle"]
    )
    if image_conditioned:
        return gen_data.get("obp_image_prompt", False)
    return gen_data["obp_assume_direct_control"] or not prompt.strip()


def process_prompt(style, prompt, negative, gen_data=None):
    if gen_data is None:
        gen_data = {}
    if gen_data.get('_metadata_prompt'):
        return prompt, negative
    enhanced = False
    def enhance_once(text, model_type="", preset_focus=""):
        nonlocal enhanced
        if enhanced:
            return text
        from modules.prompt_enhancement import enhance_prompt
        def check_cancelled():
            from modules.async_worker import check_interrupt
            check_interrupt(gen_data)
        result = enhance_prompt(text, model_type, preset_focus, check_cancelled=check_cancelled)
        enhanced = True
        return result
    if use_random_prompt(gen_data, prompt):
        prompt = build_dynamic_prompt(
            insanitylevel=gen_data["obp_insanitylevel"],
            forcesubject=gen_data["obp_subject"],
            artists=gen_data["obp_artist"],
            subtypeobject=gen_data["obp_chosensubjectsubtypeobject"],
            subtypehumanoid=gen_data["obp_chosensubjectsubtypehumanoid"],
            subtypeconcept=gen_data["obp_chosensubjectsubtypeconcept"],
            gender=gen_data["obp_chosengender"],
            imagetype=gen_data["obp_imagetype"],
            imagemodechance=gen_data["obp_imagemodechance"],
            givensubject=gen_data["obp_givensubject"],
            smartsubject=gen_data["obp_smartsubject"],
            overrideoutfit=gen_data["obp_givenoutfit"],
            prefixprompt=gen_data["obp_prefixprompt"],
            suffixprompt=gen_data["obp_suffixprompt"],
            giventypeofimage=gen_data["obp_giventypeofimage"],
            antivalues=gen_data["obp_antistring"],
            OBP_preset=gen_data["OBP_preset"],
            advancedprompting=False,
            base_model=gen_data["OBP_modeltype"],
            prompt_enhancer=gen_data["OBP_promptenhance"],
            enhancer=enhance_once,
        )

    # styles
    pattern = re.compile(r"<style:([^>]+)>")
    # Dropdown selections are staged for "Send Style to prompt". Generation
    # must not invisibly apply them again, including saved/default selections.
    styles = (style or []).copy() if "task_type" in gen_data and gen_data["task_type"] == "tool_call" else []
    def inline_style(match):
        name = STYLE_ALIASES.get(match.group(1), match.group(1))
        key = f"Style: {name}"
        if key not in available_styles:
            key = name
        if key in available_styles:
            styles.append(key)
            return ""
        return match.group(0)
    prompt = pattern.sub(inline_style, prompt)
    if "lora_keywords" in gen_data:
        keywords = gen_data["lora_keywords"]
    else:
        keywords = ""
    p_txt, n_txt = apply_style(styles, prompt, negative, keywords, enhancer=enhance_once)

    # wildcards
    if "__" in p_txt:
        p_txt = process_wildcards(p_txt)

    # apply auto negative prompt if enabled
    if "auto_negative" in gen_data and gen_data["auto_negative"] == True:
        n_txt = build_dynamic_negative(
            positive_prompt=p_txt, existing_negative_prompt=n_txt, base_model="SDXL"
        )
    return p_txt, n_txt


def parse_loras(prompt, negative):
    pattern = re.compile(r"<lora:([^<>]+):([+-]?(?:\d+(?:\.\d*)?|\.\d+))>")
    loras = []
    for match in re.finditer(pattern, prompt):
        loras.append({"name": f"{match.group(1)}.safetensors", "weight": float(match.group(2)), "hash": None})
    for match in re.finditer(pattern, negative):
        loras.append({"name": f"{match.group(1)}.safetensors", "weight": float(match.group(2)), "hash": None})
    return loras, re.sub(pattern, "", prompt), re.sub(pattern, "", negative)
