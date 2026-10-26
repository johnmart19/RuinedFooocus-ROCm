"""Explicit MiniMax recovered-8B production functional run, not native parity.

Run only when the GPU is idle; this invokes real inference at Model Defaults
(currently 20 steps), 39 frames and the canonical 768x768 canvas. The caller must
also record the process exit code: a JSON report precedes native Py_Exit.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, default=ROOT/'outputs/feature-validation-2026-10-25/cases/video-minimax-baseline-t2v.json')
    parser.add_argument('--report', type=Path, default=ROOT/'outputs/feature-validation-2026-10-25/video-minimax-recovered-t2v-functional.json')
    options = parser.parse_args()
    record = {'scope': 'Production recovered-8B T2V functional execution only',
              'native_equivalence_claim': False, 'quality_claim': False,
              'external_exit_requires_verification': True, 'functional_passed': False,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    started = time.perf_counter()
    artifacts = None
    try:
        case = json.loads(options.case.read_text(encoding='utf-8'))
        from tools.validate_comfy_workflows import initialize, cleanup_runtime, provenance
        initialize(case.get('runtime_args', []))
        import torch
        import comfy.model_management as memory
        import shared
        from tools.comfy_video_app import generate
        kwargs = dict(case['app']['kwargs'])
        kwargs.update(family='minimax', source_image=None, recovered_encoder=True,
                      length=39, width=768, height=768, fps=24., lora=None)
        path = Path(kwargs['diffusion_model']).resolve(strict=True)
        names = shared.models.get_names('checkpoints')
        model_name = next(name for name in names if shared.models.get_file('checkpoints', name).resolve() == path)
        defaults = shared.performance_settings.get_perf_options('Model Defaults', model_name, 'MiniMaxH3')
        kwargs.update(steps=defaults['custom_steps'], cfg=defaults['cfg'],
                      sampler=defaults['sampler_name'], scheduler=defaults['scheduler'])
        kwargs['output_video'] = str(options.report.with_suffix('.mp4').resolve())
        record.update(kwargs=kwargs, model_defaults={key: defaults[key] for key in shared.performance_settings.default_settings},
                      provenance=provenance())
        with torch.inference_mode():
            artifacts = generate(**kwargs)
        selected = artifacts['encoder_selection']
        assert selected['requested_recovered'] and selected['actual_recovered'], selected
        frames = artifacts['output'].detach().cpu().float()
        audio = artifacts['audio']
        assert audio is not None, 'Missing decoded audio'
        waveform = audio['waveform'].detach().cpu().float()
        video_finite = bool(torch.isfinite(frames).all())
        audio_finite = bool(torch.isfinite(waveform).all())
        assert frames.numel() and waveform.numel() and video_finite and audio_finite
        record.update(functional_passed=True, encoder_selection=selected,
            settings=artifacts['settings'],
            video={'shape': list(frames.shape), 'finite': video_finite,
                   'min': float(frames.min()), 'max': float(frames.max())},
            audio={'shape': list(waveform.shape), 'finite': audio_finite,
                   'sample_rate': audio['sample_rate'], 'rms': float(waveform.square().mean().sqrt())})
        artifacts = None
        del frames, waveform, audio
        record['runtime_cleanup'] = cleanup_runtime(memory, torch, gc)
    except Exception:
        record.update(functional_passed=False, error=traceback.format_exc())
    finally:
        artifacts = None
        gc.collect()
        record['seconds'] = round(time.perf_counter() - started, 1)
        options.report.parent.mkdir(parents=True, exist_ok=True)
        options.report.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print(json.dumps({'report': str(options.report), 'functional_passed': record['functional_passed']}), flush=True)
    return 0 if record['functional_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
