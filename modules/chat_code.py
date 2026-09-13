"""Prepare a browser-only Python runner from fenced assistant code."""

import html
import json
import re
from pathlib import Path


def python_blocks(history):
    for message in reversed(history or []):
        if message.get("role") != "assistant" or not isinstance(message.get("content"), str):
            continue
        text = re.sub(r"<(think|thinking)>.*?(?:</\1>|$)", "", message["content"], flags=re.S)
        blocks = re.findall(r"^```(?:python|py)\s*\n(.*?)^```\s*$", text, re.M | re.S | re.I)
        if blocks:
            return blocks
    return []


def python_runner(history):
    blocks = python_blocks(history)
    template = (Path(__file__).parent.parent / "html" / "chat_python.html").read_text(encoding="utf-8")
    data = json.dumps(blocks or [""], ensure_ascii=True)
    # Model output is data, never HTML or JavaScript source.
    document = template.replace("__CODE_BLOCKS__", data.replace("<", "\\u003c"))
    return ('<iframe title="Python runner" sandbox="allow-scripts" '
            'style="width:100%;height:620px;border:0" srcdoc="'
            + html.escape(document, quote=True) + '"></iframe>')
