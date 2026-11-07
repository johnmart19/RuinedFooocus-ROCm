"""Native image sampling options; model recipes never replace explicit values."""

import re


LCM_FIELDS = {
    "lcm_noise": "s_noise",
    "lcm_noise_end": "s_noise_end",
    "lcm_noise_clip": "noise_clip_std",
}


def hidream_o1_dev(checkpoint):
    name = str(checkpoint or "").replace("\\", "/").rsplit("/", 1)[-1]
    return bool(re.search(r"(?:^|[_. -])dev(?:[_. -]|$)", name, re.IGNORECASE))


def lcm_options(values, family, checkpoint=None):
    clip = 2.5 if family in ("HiDreamO1", "HiDream O1") and hidream_o1_dev(checkpoint) else 0.0
    defaults = dict(s_noise=1.0, s_noise_end=1.0, noise_clip_std=clip)
    return {
        native: defaults[native] if values.get(key) is None else values[key]
        for key, native in LCM_FIELDS.items()
    }


def sample_lcm(
    *,
    model,
    noise,
    steps,
    cfg,
    scheduler,
    positive,
    negative,
    latent_image,
    denoise,
    noise_options,
    noise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
):
    from comfy.sample import sample_custom
    from comfy_extras.nodes_custom_sampler import BasicScheduler
    from comfy_extras.nodes_advanced_samplers import SamplerLCM

    sampler = SamplerLCM.execute(**noise_options)[0]
    sigmas = BasicScheduler.execute(
        model=model, scheduler=scheduler, steps=steps, denoise=1.0 if denoise is None else denoise
    )[0]
    return sample_custom(
        model=model,
        noise=noise,
        cfg=cfg,
        sampler=sampler,
        sigmas=sigmas,
        positive=positive,
        negative=negative,
        latent_image=latent_image,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
