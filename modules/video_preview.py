import time

from latent_preview import Latent2RGBPreviewer
from modules import async_worker as worker


def video_callback(model, gen_data, label="Generating video"):
    latent_format = model.model.latent_format
    previewer = None
    if latent_format.latent_rgb_factors is not None:
        previewer = Latent2RGBPreviewer(
            latent_format.latent_rgb_factors,
            latent_format.latent_rgb_factors_bias,
            latent_format.latent_rgb_factors_reshape,
        )
    started = None

    def duration(seconds):
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes}:{seconds:02d}"

    def callback(step, x0, x, total_steps):
        nonlocal started
        worker.check_interrupt(gen_data)
        now = time.monotonic()
        if started is None:
            started = now
        image = None
        if previewer is not None:
            if x0.is_nested:
                x0 = x0.tensors[0]
            # Copy one frame synchronously before converting it to a PIL image.
            frame = x0[:1, :, :1] if x0.ndim == 5 else x0[:1]
            image = previewer.decode_latent_to_preview(frame.float().cpu())
            image = image.resize((gen_data["width"], gen_data["height"]))
        completed = step + 1
        elapsed = now - started
        # The first callback also includes model loading; estimate from later steps.
        seconds_per_step = elapsed / step if step else 0
        remaining = seconds_per_step * (total_steps - completed)
        timing = (f"Elapsed {duration(elapsed)} · About {duration(remaining)} left · "
                  f"{seconds_per_step:.1f} s/step") if step else "Estimating time ..."
        worker.add_result(
            gen_data["task_id"],
            "preview",
            (int(100 * completed / total_steps),
             f"{label} - {completed}/{total_steps} · {timing}", image),
        )

    return callback
