"""Observational adapters around the actual production video pipelines.

Signatures mirror comfy_video_reference.generate/legacy. Component choices are
set before pipeline import and restored afterwards. No sampler return values or
production generation policies are changed. Use a fresh process per recipe.
"""
from contextlib import ExitStack
from pathlib import Path
import shutil
from unittest.mock import patch
from tools.comfy_feature_app import _values, _local, _catalog, _input


def _observe(owner, data, av, output_video):
    import comfy.samplers
    from nodes import VAEDecode, VAEDecodeTiled
    from comfy_extras.nodes_audio import VAEDecodeAudio
    observed, raw = {}, {}
    original_conds = comfy.samplers.CFGGuider.inner_set_conds
    original_sample = comfy.samplers.CFGGuider.sample
    original_decode = VAEDecode.decode
    original_tiled = VAEDecodeTiled.decode
    original_audio = VAEDecodeAudio.execute
    audio_vae = getattr(owner.model_base_patched, 'audio_vae', None)

    def conds(self, values):
        raw[id(self)] = dict(values)
        return original_conds(self, values)

    def sample(self, noise, latent_image, sampler, sigmas, *args, **kwargs):
        observed['conditioning'] = dict(raw[id(self)])
        observed['conditioning'].setdefault('negative', None)
        observed['sigmas'] = sigmas
        return original_sample(self, noise, latent_image, sampler, sigmas, *args, **kwargs)

    def decode(self, vae, samples, *args, **kwargs):
        images = original_decode(self, vae, samples, *args, **kwargs)
        observed['video_latent'] = samples
        observed['output'] = images[0]
        return images

    def tiled(self, vae, samples, *args, **kwargs):
        images = original_tiled(self, vae, samples, *args, **kwargs)
        observed['video_latent'] = samples
        observed['output'] = images[0]
        return images

    def audio(cls, vae, samples):
        sound = original_audio(vae=vae, samples=samples)
        if vae is audio_vae:
            observed['audio_latent'] = samples
            observed['audio'] = sound[0]
        return sound

    with ExitStack() as stack:
        stack.enter_context(patch.object(comfy.samplers.CFGGuider, 'inner_set_conds', conds))
        stack.enter_context(patch.object(comfy.samplers.CFGGuider, 'sample', sample))
        stack.enter_context(patch.object(VAEDecode, 'decode', decode))
        stack.enter_context(patch.object(VAEDecodeTiled, 'decode', tiled))
        stack.enter_context(patch.object(VAEDecodeAudio, 'execute', classmethod(audio)))
        artifacts = owner.process(gen_data=data, callback=None)
    if output_video:
        # Production AV returns its GIF; its actual MP4 is the same stem.
        candidates = [Path(value).with_suffix('.mp4') for value in artifacts if isinstance(value, (str, Path))]
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            raise RuntimeError('Production pipeline did not emit an MP4 artifact')
        target = Path(output_video).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target:
            shutil.copy2(source, target)
    observed['latent'] = ({'video': observed.pop('video_latent')['samples'], 'audio': observed.pop('audio_latent', {}).get('samples')}
                          if av else observed.pop('video_latent')['samples'])
    if av:
        observed.setdefault('audio', None)
    return observed


def _run(family, diffusion_model, text_encoder, vae_model, audio_vae_model=None,
         text_encoder2=None, positive='a landscape', negative='', seed=1, steps=20,
         cfg=4.0, sampler='euler', scheduler='normal', width=512, height=320,
         length=9, fps=24.0, source_image=None, distilled=False, weight_dtype='default',
         lora=None, lora_strength=1.0, output_video=None, clip_vision_model=None,
         sampling_shift=None, refinement_upscaler=None, gguf_loader="upstream", recovered_encoder=False):
    import shared
    if gguf_loader not in ('upstream', 'shared_compact'):
        raise ValueError('gguf_loader describes the native reference policy, upstream or shared_compact')
    # The app always retains its production loader; this option describes the
    # paired native run and must never replace the production loader here.
    if weight_dtype != 'default' or sampling_shift is not None:
        raise ValueError('Production video uses default weight dtype and fixed family shift; omit overrides')
    if family == 'hunyuan' and text_encoder2 is None:
        raise ValueError('Hunyuan requires its CLIP-L and LLaVA text encoders')
    model = _local(diffusion_model)
    text, vae = str(_local(text_encoder)), str(_local(vae_model))
    updates = {'minimax_recovered_encoder': _recovered_selection(family, recovered_encoder)}
    module = {'ltx_legacy': 'ltx_video_pipeline', 'wan21': 'wan_video_pipeline',
              'wan22_5b': 'wan_video_pipeline', 'hunyuan': 'hunyuan_video_pipeline',
              'ltx_av': 'video_pipeline', 'minimax': 'video_pipeline'}[family]
    if family == 'ltx_legacy':
        updates.update(clip_t5=text, vae_ltxv=vae)
    elif family in ('wan21', 'wan22_5b'):
        updates.update(clip_umt5=text, vae_wan=vae, vae_wan_22=vae)
        if clip_vision_model:
            updates['clip_vision'] = str(_local(clip_vision_model))
    elif family == 'hunyuan':
        updates.update(clip_l=text, clip_llava=str(_local(text_encoder2)), vae_hunyuan_video=vae)
        if clip_vision_model:
            updates['clip_hunyuan_vision'] = str(_local(clip_vision_model))
    else:
        audio = str(_local(audio_vae_model))
        updates.update({key: text for key in ('clip_gemma3_12b', 'clip_gemma4_12b', 'clip_qwen3vl_32b')})
        updates.update({key: vae for key in ('vae_ltxv2_video', 'vae_ltxv23_video', 'vae_ltxv25_video', 'vae_minimax_h3_video')})
        updates.update({key: audio for key in ('vae_ltxv2_audio', 'vae_ltxv23_audio', 'vae_ltxv25_audio', 'vae_minimax_h3_audio')})
        if text_encoder2:
            updates.update({key: str(_local(text_encoder2)) for key in ('clip_ltx23_text_proj', 'clip_ltx2_distilled', 'clip_ltx2_dev')})
    files = {'checkpoints': model}
    if lora:
        files['loras'] = _local(lora)
    if refinement_upscaler:
        asset = _local(refinement_upscaler)
    with ExitStack() as runtime:
        runtime.enter_context(_values(shared.settings.default_settings, updates))
        runtime.enter_context(_catalog(files))
        if refinement_upscaler:
            original_get = shared.path_manager.get_folder_file_path
            def local_upscaler(folder, filename, *args, **kwargs):
                if folder == 'latent_upscalers' and filename == asset.name:
                    return asset
                return original_get(folder, filename, *args, **kwargs)
            runtime.enter_context(patch.object(shared.path_manager, 'get_folder_file_path', local_upscaler))
        import importlib
        # Match production bootstrap: worker imports the pipeline registry before
        # individual pipelines import video_preview and its worker dependency.
        importlib.import_module('modules.async_worker')
        app = importlib.import_module('modules.' + module)
        owner = app.pipeline()
        owner.load_base_model(str(model), unet_only=True)
        owner.load_loras([{'name': str(files['loras']), 'weight': lora_strength}] if lora else [])
        if family == 'ltx_av':
            from modules.video_settings import is_ltx25_distilled
            production_distilled = (owner.model_info['unet_type'] == 'LTXAV2.5' and
                                    is_ltx25_distilled('LTXV 2.5', str(model)))
            if bool(distilled) != production_distilled:
                raise ValueError('distilled must match the production model filename/family policy')
            if refinement_upscaler and not production_distilled:
                raise ValueError('Production spatial refinement requires LTX 2.5 distilled')
        data = {'input_image': _input(source_image, width, height) if source_image else None,
                'positive_prompt': positive, 'negative_prompt': negative, 'seed': seed,
                'steps': steps, 'cfg': cfg, 'sampler_name': sampler, 'scheduler': scheduler,
                'width': width, 'height': height, 'base_model_name': str(model),
                'task_id': 'comfy-video-validation', 'loras': [], 'image_number': length,
                'video_fps': fps, 'video_duration': (length - 1) / fps if family in ('ltx_legacy', 'wan21', 'wan22_5b', 'hunyuan') else length / fps,
                'show_preview': False}
        data = owner.parse_gen_data(data)
        artifacts = _observe(owner, data, family in ('ltx_av', 'minimax'), output_video)
        if recovered_encoder:
            artifacts['encoder_selection'] = {
                'requested_recovered': True,
                'actual_recovered': bool(getattr(owner.model_base_patched.clip, '_rf_minimax_recovered', False)),
                'scope': 'Production functional execution only; no independent 32B parity claim'}
    artifacts['settings'] = {'seed': seed, 'steps': int(artifacts['sigmas'].shape[0]) - 1,
        'cfg': 1.0 if family == 'minimax' else cfg, 'sampler': sampler,
        'scheduler': ('LTXDistilledRefinement' if len(artifacts['sigmas']) == 4 and distilled else 'LTXDistilled' if distilled else 'LTXVScheduler') if family in ('ltx_legacy', 'ltx_av') else scheduler,
        'width': data['width'], 'height': data['height'], 'length': int(artifacts['output'].shape[0]),
        'fps': fps, 'lora_strength': lora_strength if lora else None}
    if family not in ('ltx_av', 'minimax'):
        from tools.comfy_video_reference import _conditioning_artifact
        artifacts['conditioning'] = _conditioning_artifact(artifacts['conditioning'])
        artifacts['settings'].update(family=family, sampling_shift=sampling_shift)
        # Native legacy reports the caller scheduler despite its LTX scheduler node.
        artifacts['settings']['scheduler'] = scheduler
    return artifacts


def generate(family, diffusion_model, text_encoder, vae_model, audio_vae_model,
             text_encoder2=None, positive='a landscape', negative='', seed=1,
             steps=20, cfg=4.0, sampler='euler', scheduler='normal', width=512,
             height=320, length=9, fps=24.0, source_image=None, distilled=False,
             weight_dtype='default', lora=None, lora_strength=1.0, output_video=None,
             refinement_upscaler=None, gguf_loader='upstream', recovered_encoder=False):
    return _run(**locals())


def legacy(family, diffusion_model, text_encoder, vae_model, text_encoder2=None,
           positive='a landscape', negative='', seed=1, steps=20, cfg=4.0,
           sampler='euler', scheduler='normal', width=512, height=320, length=9,
           fps=24.0, source_image=None, weight_dtype='default', clip_vision_model=None,
           sampling_shift=None, lora=None, lora_strength=1.0, output_video=None,
           gguf_loader='upstream'):
    return _run(**locals())


def _recovered_selection(family, recovered_encoder):
    if type(recovered_encoder) is not bool:
        raise ValueError('recovered_encoder must be a boolean')
    if recovered_encoder and family != 'minimax':
        raise ValueError('recovered_encoder is only supported for MiniMax')
    return recovered_encoder
