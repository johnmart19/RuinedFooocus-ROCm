# MiniMax H3 on 24GB GPUs

Select `MiniMaxH3/minimax_h3_fl2va_pruned_fp8_U16G.gguf`. The application resolves
the checkpoint and its required encoders/VAEs through the normal model database.
Missing recovered 8B encoder weights, ARA, adapter and manifest download when the
model loads. Configured model directories and existing files are respected;
downloads use the application's resumable Hugging Face transport.

Choose **Model Defaults**: 20 steps, CFG 1, `res_multistep`, `simple`, and native
1344×768 or 768×1344. Lower Turbo step counts require the matching Turbo LoRA.
See the [ComfyUI guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3).

The U16G loader preserves packed Q4 layers instead of expanding them permanently
to BF16, while keeping Q8_CR layers on the backend's native INT8 ConvRot path.
On the RX 7900 XTX, loaded weights occupy 13.98 GiB and fit on GPU with 6 GiB of
requested workspace headroom. Actual sampling memory also depends on duration
and resolution; component residency does not establish full-video performance.

Text-only requests use the recovered 8B INT8 encoder with its required ARA and
4096-to-5120 conditioning adapter. Local conditioning checks passed with finite
embeddings and 5.90 GiB peak Torch GPU allocation. The encoder is unloaded from
GPU after conditioning. Image inputs require the official 32B encoder; the app
switches between the two as needed. Explicit official encoder selections remain
respected. The recovered [encoder package](https://huggingface.co/SearchingMan/MiniMax-H3-Text-Encoders)
uses a pinned [MIT-licensed loader](https://github.com/kgonia/ComfyUI-MiniMaxH3TextEncoders)
and separately licensed Apache-2.0 weights, with package hash verification.

The original FP8 and INT8 diffusion alternatives and official 32B encoders are
also registered in the model database. They are fetched when selected or needed
for image conditioning, rather than downloaded as an extra bundle.
The additional 6 GiB Windows ROCm workspace policy applies only
to safetensors on 20–26 GiB GPUs; U16G retains the native workspace estimate.

Packed Q4 numerical checks and GPU weight-residency checks passed. Full five-
and fifteen-second video speed and quality remain unverified. Sparse attention
is not enabled automatically.
