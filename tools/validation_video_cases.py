"""Write offline app/native video comparison recipes; never load models or run a GPU.

The four-step tier checks execution only. Baselines compare the application's
selected stock-node policy, not the model vendor's complete quality workflow.
LTX-2.5 distilled always retains its required eight plus three step schedule.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def recipes():
    # name, adapter family, checkpoint, encoder, VAE, frames, width, height,
    # baseline steps, CFG, sampler, fps, supports image conditioning
    rows = [
        ('ltx-legacy', 'ltx_legacy', 'ltxv-2b-0.9.6-dev-04-25.safetensors',
         't5-v1_1-xxl-encoder-Q3_K_S.gguf', 'LTX-Video-0.9.6-VAE-BF16.safetensors', 25, 512, 320, 40, 3, 'euler', 24, True),
        ('wan21', 'wan21', 'wan2.1_t2v_1.3B_bf16.safetensors',
         'umt5_xxl_fp8_e4m3fn_scaled.safetensors', 'wan_2.1_vae.safetensors', 33, 512, 320, 20, 6, 'uni_pc', 16, False),
        ('wan22-5b', 'wan22_5b', 'wan2.2_ti2v_5B_fp16.safetensors',
         'umt5_xxl_fp8_e4m3fn_scaled.safetensors', 'wan2.2_vae.safetensors', 33, 640, 384, 20, 5, 'uni_pc', 24, True),
        ('hunyuan', 'hunyuan', 'hunyuan-video-t2v-720p-Q4_K_M.gguf',
         'clip_l.safetensors', 'hunyuan_video_vae_bf16.safetensors', 25, 640, 384, 20, 6, 'euler', 24, False),
        ('wan21-i2v', 'wan21', 'wan2.1-i2v-14b-480p-Q4_K_M.gguf',
         'umt5_xxl_fp8_e4m3fn_scaled.safetensors', 'wan_2.1_vae.safetensors', 33, 512, 320, 20, 6, 'uni_pc', 16, True),
        ('hunyuan-i2v', 'hunyuan', 'hunyuan-video-i2v-720p-Q4_K_M.gguf',
         'clip_l.safetensors', 'hunyuan_video_vae_bf16.safetensors', 25, 640, 384, 20, 6, 'euler', 24, True),
        ('ltx2', 'ltx_av', 'ltx-2-19b-dev-Q4_K_M.gguf',
         'gemma-3-12b-it-Q4_0.gguf', 'LTX2_video_vae_bf16.safetensors', 25, 768, 512, 20, 4, 'euler_ancestral', 24, True),
        ('ltx23', 'ltx_av', 'ltx-2.3-22b-dev-Q4_K_M.gguf',
         'gemma-3-12b-it-Q4_0.gguf', 'LTX23_video_vae_bf16.safetensors', 25, 768, 512, 30, 3, 'euler', 24, True),
        ('ltx25', 'ltx_av', 'ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors',
         'gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors', 'ltx-2.5-video-vae-bf16.safetensors', 25, 768, 512, 8, 1, 'euler_ancestral', 24, True),
        ('minimax', 'minimax', 'minimax_h3_fl2va_pruned_fp8_U16G.gguf',
         'qwen3vl_32b_minimax_h3_int8_convrot.safetensors', 'minimax_h3_video_vae_fp16.safetensors', 39, 768, 768, 20, 1, 'res_multistep', 24, True),
    ]
    return rows


def write_cases(output_root, source_image):
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    index = {}
    for path in (ROOT / 'models').rglob('*'):
        if path.is_file() and '.cache' not in path.parts:
            index.setdefault(path.name, []).append(path)
    missing = set()
    pinned = json.loads((ROOT / 'tools/validation_assets.json').read_text(encoding='utf-8'))['assets']
    destinations = {Path(row['file']).name: row['destination'] for row in pinned}

    def asset(name, category, family=''):
        matches = index.get(name, [])
        if len(matches) > 1:
            raise ValueError(f'Ambiguous local asset {name}: {matches}')
        if matches:
            return str(matches[0].resolve())
        missing.add(name)
        return str(ROOT / destinations[name]) if name in destinations else str(ROOT / 'models' / category / family / name)

    manifest = {'purpose': 'Execution smoke and same-component application-policy comparison; no quality claim',
                'cases': [], 'missing_assets': []}
    for row in recipes():
        name, family, model, encoder, vae, length, width, height, steps, cfg, sampler, fps, i2v = row
        kwargs = dict(family=family, diffusion_model=asset(model, 'checkpoints', name),
                      text_encoder=asset(encoder, 'clip'), vae_model=asset(vae, 'vae'),
                      positive='A small red sailboat gently moving across a calm lake, steady camera, soft daylight.',
                      negative='' if family in ('hunyuan', 'minimax') or name == 'ltx25' else 'blur, distortion, flickering',
                      seed=12345, steps=steps, cfg=cfg, sampler=sampler, scheduler='simple',
                      width=width, height=height, length=length, fps=fps)
        entry = 'generate' if family in ('ltx_av', 'minimax') else 'legacy'
        if family == 'hunyuan':
            kwargs['text_encoder2'] = asset('llava_llama3_fp8_scaled.safetensors', 'clip')
        if name in ('wan21-i2v', 'hunyuan-i2v'):
            kwargs['clip_vision_model'] = asset('clip_vision_h_fp8_e4m3fn.safetensors' if name == 'wan21-i2v'
                                               else 'llava_llama3_vision.safetensors', 'clip_vision')
        if family in ('ltx_av', 'minimax'):
            audio = {'ltx2': 'LTX2_audio_vae_bf16.safetensors', 'ltx23': 'LTX23_audio_vae_bf16.safetensors',
                     'ltx25': 'ltx-2.5-audio-vae-bf16.safetensors', 'minimax': 'minimax_h3_audio_vae_fp32.safetensors'}[name]
            kwargs['audio_vae_model'] = asset(audio, 'vae')
        if name in ('ltx2', 'ltx23'):
            kwargs['text_encoder2'] = asset('ltx-2-19b-embeddings_connector_dev_bf16.safetensors'
                                           if name == 'ltx2' else 'ltx-2.3_text_projection_bf16.safetensors', 'clip')
        if name == 'ltx25':
            kwargs.update(distilled=True, refinement_upscaler=asset(
                'ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors', 'latent_upscale_models'))
        for tier in ('smoke', 'baseline'):
            modes = ('i2v',) if name.endswith('-i2v') else ('t2v',) if tier == 'smoke' or not i2v else ('t2v', 'i2v')
            for mode in modes:
                case_name = f'video-{name}-{tier}-{mode}'
                values = dict(kwargs)
                if tier == 'smoke' and name != 'ltx25':
                    values['steps'] = 4
                if mode == 'i2v':
                    values['source_image'] = str(Path(source_image).resolve())
                spec = {'runtime_args': ['--default-device', '0', '--use-quad-cross-attention', '--preview-method', 'none'],
                        'atol': 0.0001, 'rtol': 0.0001,
                        'validation_scope': {'tier': tier, 'quality_claim': False,
                            'policy': 'Matched application stock-node baseline; no STG/modality guidance',
                            'mandatory_distilled_two_stage': name == 'ltx25',
                            'canonical_minimax_canvas': name == 'minimax'},
                        'app': {'entrypoint': f'tools.comfy_video_app:{entry}', 'kwargs': dict(values)},
                        'native': {'entrypoint': f'tools.comfy_video_reference:{entry}', 'kwargs': dict(values)}}
                for side in ('app', 'native'):
                    spec[side]['kwargs']['output_video'] = str(output_root.parent / (case_name + '-' + side + '.mp4'))
                path = output_root / (case_name + '.json')
                path.write_text(json.dumps(spec, indent=2) + '\n')
                manifest['cases'].append({'name': case_name, 'tier': tier, 'spec': str(path)})
    manifest['missing_assets'] = sorted(missing)
    (output_root / 'video-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default=str(ROOT / 'outputs/feature-validation-2026-10-25/cases'))
    parser.add_argument('--source-image', default=str(ROOT / 'outputs/feature-validation-2026-10-25/source.png'))
    options = parser.parse_args()
    result = write_cases(options.output_root, options.source_image)
    print(json.dumps({'cases': len(result['cases']), 'missing_assets': result['missing_assets']}, indent=2))
