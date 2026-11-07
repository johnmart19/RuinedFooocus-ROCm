import random
import re
import csv
import locale
from os.path import exists
from csv import DictReader
from pathlib import Path
from modules.prompt_expansion import PromptExpansion
from modules.prompt_enhancement import enhance_prompt
from random_prompt import build_dynamic_prompt

DEFAULT_STYLES_FILE = Path("settings/styles.default")
STYLES_FILE = Path("settings/styles.csv")

STYLE_ALIASES = {"Erniehancer": "Local chat model", "Style: Erniehancer": "Local chat model"}

prompt_expansion = PromptExpansion()


def load_styles():
    default_styles = []
    styles = []
    styles_encoding = "utf-8-sig"

    with open(DEFAULT_STYLES_FILE, encoding="utf-8-sig", newline="") as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            default_styles.append(row)

    if exists(STYLES_FILE):
        try:
            with open(STYLES_FILE, encoding=styles_encoding, newline="") as f:
                styles = list(csv.reader(f))
        except UnicodeDecodeError:
            # Earlier Windows releases wrote custom styles in the system locale.
            styles_encoding = locale.getpreferredencoding(False)
            with open(STYLES_FILE, encoding=styles_encoding, newline="") as f:
                styles = list(csv.reader(f))

    # Add any missing default styles
    changed = False
    for row in default_styles:
        if row not in styles:
            styles.append(row)
            changed = True

    if changed:
        styles_encoding = "utf-8-sig"
        with open(STYLES_FILE, "w", encoding="utf-8", newline='') as f:
            csv_writer = csv.writer(f)
            for row in styles:
                csv_writer.writerow(row)


    with STYLES_FILE.open("r", encoding=styles_encoding, newline="") as f:
        reader = DictReader(f)
        styles = list(reader)

    default_style = {"name": "None", "prompt": "{prompt}", "negative_prompt": ""}
    random_style = {
        "name": "Style: Pick Random",
        "prompt": "{prompt}",
        "negative_prompt": "",
    }
    lora_keywords_style = {
        "name": "LoRA keywords",
        "prompt": "{prompt} {lora_keywords}",
        "negative_prompt": "",
    }
    flufferizer_style = {
        "name": "Flufferizer",
        "prompt": "{prompt}",
        "negative_prompt": "",
    }
    hyperprompt_style = {
        "name": "Hyperprompt",
        "prompt": "{prompt}",
        "negative_prompt": "",
    }
    local_chat_style = {
        "name": "Local chat model",
        "prompt": "{prompt}",
        "negative_prompt": "",
    }

    styles = [s for s in styles if s["name"] not in STYLE_ALIASES]
    styles.insert(0, local_chat_style)
    styles.insert(0, hyperprompt_style)
    styles.insert(0, flufferizer_style)
    styles.insert(0, lora_keywords_style)
    styles.insert(0, random_style)
    styles.insert(0, default_style)

    return {s["name"]: (s["prompt"], s["negative_prompt"]) for s in styles}


def apply_style(style, prompt, negative_prompt, lora_keywords, *, enhancer=None):
    output_prompt = ""
    negative_parts = []
    temp_style_prompt = prompt
    bFlufferizer = False
    bHyperprompt = False
    bLocalChat = False
    artifylist = []
    artifystylelist = []
    index = 0

    if not style:
        return prompt, negative_prompt

    while "Style: Pick Random" in style:
        style[style.index("Style: Pick Random")] = random.choice(allstyles)

    for s in style.copy():
        _s = s.upper().strip()
        if _s in map(str.upper, ["Flufferizer", "Style: Flufferizer"]):
            bFlufferizer = True
            del style[style.index(s)]

        if _s in map(str.upper, ["Hyperprompt", "Style: Hyperprompt"]):
            bHyperprompt = True
            del style[style.index(s)]

        if _s in map(str.upper, ["Local chat model", "Style: Local chat model", *STYLE_ALIASES]):
            bLocalChat = True
            del style[style.index(s)]

        if _s in map(str.upper, ["LoRA keywords", "Style: LoRA keywords"]):
            style[style.index(s)] = "LoRA keywords" # Make sure it has the correct name

    if bHyperprompt:
        prompt = build_dynamic_prompt.one_button_superprompt(prompt=prompt)
        temp_style_prompt = prompt
        print("Hypered prompt: " + prompt)

    while index < len(style):
        if style[index].startswith("Artify"):
            artifylist.append(style[index].replace("Artify: ", ""))
            artifystylelist.append(style[index])
        index += 1

    style = [x for x in style if x not in artifystylelist]

    for s in style:
        p, n = styles.get(s, default_style)
        if (s == "Style: sai-cinematic" and p and p.startswith("cinematic film still {prompt}")
                and re.search(r"\banime\b", prompt, re.IGNORECASE)):
            p = p.replace("cinematic film still", "cinematic anime illustration", 1)
            # Do not make the style reject the medium explicitly requested in the prompt.
            n = ", ".join(tag.strip() for tag in (n or "").split(",")
                          if tag.strip().lower() not in {"", "anime", "cartoon", "graphic", "painting", "crayon"})
        if p is not None:
            template = (p + ", ").replace(", ,", ", ")
            def substitute(match):
                if match.group(0) == "{prompt}":
                    return temp_style_prompt
                return "{lora_keywords}" if lora_keywords is None else lora_keywords
            temp_style_prompt = re.sub(r"\{prompt\}|\{lora_keywords\}", substitute, template)
            output_prompt = temp_style_prompt
        if n:
            negative_parts.append(n)

    # prep outputprompt for use in Flufferize and Artify
    if output_prompt == "":
        output_prompt = prompt

    for artist in artifylist:
        output_prompt = build_dynamic_prompt.artify_prompt(prompt=output_prompt, artists=artist)

    if bFlufferizer:
        output_prompt = prompt_expansion.expand_prompt(output_prompt)

    if bLocalChat:
        output_prompt = (enhancer or enhance_prompt)(output_prompt)

    return output_prompt, ", ".join(negative_parts + ([negative_prompt] if negative_prompt else []))


styles = load_styles()
default_style = styles["None"]
allstyles = [x for x in load_styles() if x.startswith("Style")]
allstyles.remove("Style: Pick Random")
