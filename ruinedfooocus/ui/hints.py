from ruinedfooocus.core.paths import resource_path
import random
import glob


def get_hint():
    hintfiles = resource_path("hints").glob("*.txt")
    hints = []
    for hintfile in hintfiles:
        lines = open(hintfile, encoding='utf8').read().splitlines()
        hints += lines

    hint = f"**LPT:** *{random.choice(hints)}*"
    return hint
