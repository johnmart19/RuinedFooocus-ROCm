"""Audit native image tokenizers on CPU, without weights, downloads or inference.

Run with the app's Python: python tools/validate_prompt_tokenizers.py --output outputs/prompt-tokenizers
Checks empty, one-character, one-word, Unicode and 16K/64K/128K-character inputs.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def prompt_cases():
    sentence = ('A red fox beside a blue teapot on a wooden table. The scene has soft daylight, '
                'precise detail, balanced composition and natural colors. ')
    tail = ' The final detail is a violet umbrella marked END_SENTINEL_94721.'
    cases = {'empty': '', 'one-character': 'f', 'one-word': 'fox', 'unicode': '\u732b'}
    for size, label in ((16000, 'large-16k'), (64000, 'ultra-64k'), (128000, 'ultra-128k')):
        cases[label] = (sentence * (size // len(sentence) + 1))[:size] + tail
    return cases


def native_tokenizers():
    sys.path.insert(0, str(ROOT / 'repositories/ComfyUI'))
    from comfy.cli_args import args
    args.cpu = True
    from comfy.sdxl_clip import SDXLTokenizer
    from comfy.text_encoders.krea2 import Krea2Tokenizer
    from comfy.text_encoders.anima import AnimaTokenizer
    from comfy.text_encoders.moss_image import MOSSImageTokenizer
    from comfy.text_encoders.qwen_image import QwenImageTokenizer
    from comfy.text_encoders.mage_flow import MageFlowTokenizer
    return {'SDXL': SDXLTokenizer, 'Krea2': Krea2Tokenizer, 'Anima': AnimaTokenizer,
            'MOSS': MOSSImageTokenizer, 'Qwen edit': QwenImageTokenizer, 'Mage edit': MageFlowTokenizer}


def audit(factory, family, cases):
    try:
        tokenizer = factory()
    except Exception as error:
        return [{'family': family, 'error': str(error), 'passed': False}]
    results = []
    for case, prompt in cases.items():
        started = time.monotonic()
        row = {'family': family, 'case': case, 'characters': len(prompt)}
        try:
            tokens = tokenizer.tokenize_with_weights(prompt)
            counts = {key: [len(chunk) for chunk in chunks] for key, chunks in tokens.items()}
            tail_retained = None
            if 'END_SENTINEL_94721' in prompt:
                changed = tokenizer.tokenize_with_weights(prompt.replace('END_SENTINEL_94721', 'TAIL_REPLACED_68439'))
                tail_retained = tokens != changed
            row.update(tokens={key: sum(sizes) for key, sizes in counts.items()},
                       chunks={key: len(sizes) for key, sizes in counts.items()}, tail_retained=tail_retained)
            row['passed'] = bool(counts) and all(sum(sizes) > 0 for sizes in counts.values()) and tail_retained is not False
        except Exception as error:
            row.update(error=str(error), passed=False)
        row['seconds'] = round(time.monotonic() - started, 3)
        results.append(row)
        print(json.dumps(row), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/prompt-length-tokenizers')
    options = parser.parse_args()
    cases = prompt_cases()
    results = []
    for family, factory in native_tokenizers().items():
        results.extend(audit(factory, family, cases))
    report = {'comfy_commit': subprocess.check_output(
        ['git', '-C', str(ROOT / 'repositories/ComfyUI'), 'rev-parse', 'HEAD'], text=True).strip(),
        'results': results, 'passed': all(row['passed'] for row in results)}
    options.output.mkdir(parents=True, exist_ok=True)
    (options.output / 'native-tokenizers.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (options.output / 'prompts.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    return not report['passed']


if __name__ == '__main__':
    raise SystemExit(main())
