# Download catalog

Downloads report bytes received, percentage when the size is known, and average MB/s.
Files become available only after a complete transfer. Failed partial files are removed.

Index of `modules/pathdb`, audited on 2026-09-12. Each link below passed an HTTP
availability check (Hugging Face authentication was used where needed). This is
not a claim that every weight file was downloaded or generation-tested.

## Use and compatibility

| Catalog | Application route | Settings |
| --- | --- | --- |
| checkpoint | Image loader; SDXL base and Top5 | SDXL Performance preset |
| clip / vae | Matching image/video loader | Settings encoder/VAE overrides; blank uses the model default |
| clip_vision | Wan 2.1 image-to-video conditioning | CLIP Vision; not needed for Wan text-to-video |
| latent_upscalers | LTX 2.5 distilled second stage | Automatic for the two-stage workflow |
| llm | Chat and One Button local chat enhancement | Chat model and quantization; llama.cpp or xllamacpp |
| controlnet | SDXL control LoRAs | PowerUp control type; not interchangeable with other model families |
| lora / lcm | Image LoRA loader | SDXL LoRA selection and matching checkpoint recipe |
| upscalers | Spandrel plus extra architecture adapters | PowerUp upscaler selection |
| faceswap | GFPGAN through Spandrel | PowerUp: Restore cropped face (restoration, not identity swapping) |

Use components from the same architecture and release. A list entry is not a
universal replacement for every encoder or VAE. GGUF, FP8 and NVFP4 variants also
depend on the installed backend and device. Auxiliary files such as Gemma's
`mmproj`, Ernie's prompt enhancer and LTX embedding connectors are not standalone
text encoders; only the matching workflow can consume them. Hunyuan 1.5's VAE
entry alone does not implement a Hunyuan 1.5 generation workflow.

Catalogued checkpoints and LoRAs are also offered in Settings and downloaded on
first load when online. Select the LCM LoRA and its matching sampler together; DMD2
weights require their own four-step recipe and must not be applied to unrelated
model families.

LTX 2 selects its own VAEs and Dev/Distilled connector; LTX 2.3 selects its text
projection and VAEs; LTX 2.5 uses its combined Gemma encoder and separate VAEs.
Encoder/VAE changes take effect on the next model load. Other video pipelines
may require restarting after changing advanced component settings.

Performance recipes and native resolutions are documented in [model presets](model-presets.md)
and [video](video.md). Use Custom for variants without a verified recipe; a base
recipe must not silently stand in for a distilled, editing or dual-expert workflow.

## Removed download choices

- Rogue Q3_k_l was deleted upstream; its existing Q3_k_m option remains.
- Six Q4_0_4_4 / Q4_0_4_8 / Q4_0_8_8 chat downloads use formats removed by
  [llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/10847). Use a standard Q4 or K quantization.
- SPSR is no longer implemented by the current Spandrel packages. README.md is
  documentation, not an upscaler. Neither belongs in the downloadable model menu.
- Existing local files are not removed by these catalog changes.

## Files

Paths below are catalog keys; storage follows the configurable `path_*` folder.
Legacy aliases such as `lcm_lora` retain their existing filenames.

### checkpoint (2)
| Catalog key | Storage | Source |
| --- | --- | --- |
| `checkpoints/sd_xl_base_1.0_0.9vae.safetensors` | `path_checkpoints` | [Download](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors) |
| `checkpoints/SDXL 1.0/top5_v10.safetensors` | `path_checkpoints` | [Download](https://civitai.com/api/download/models/624591) |

### clip (60)
| Catalog key | Storage | Source |
| --- | --- | --- |
| `clip/clip_aura.safetensors` | `path_clip` | [Download](https://huggingface.co/fal/AuraFlow-v0.3/resolve/main/text_encoder/model.fp16.safetensors) |
| `clip/clip_g.safetensors` | `path_clip` | [Download](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder_2/model.fp16.safetensors) |
| `clip/clip_g_hidream.gguf` | `path_clip` | [Download](https://huggingface.co/calcuis/hidream-gguf/resolve/main/clip_g_hidream_fp32-f16.gguf) |
| `clip/clip_l.safetensors` | `path_clip` | [Download](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors) |
| `clip/clip_l_hidream.gguf` | `path_clip` | [Download](https://huggingface.co/calcuis/hidream-gguf/resolve/main/clip_l_hidream_fp32-f16.gguf) |
| `clip/cow-mistral3-small-q2_k.gguf` | `path_clip` | [Download](https://huggingface.co/chatpig/flux2-dev-gguf/resolve/main/cow-mistral3-small-q2_k.gguf) |
| `clip/ernie-image-prompt-enhancer.safetensors` | `path_clip` | [Download](https://huggingface.co/Comfy-Org/ERNIE-Image/resolve/main/text_encoders/ernie-image-prompt-enhancer.safetensors) |
| `clip/gemma_2_2b_fp16.safetensors` | `path_clip` | [Download](https://huggingface.co/calcuis/lumina-gguf/resolve/main/gemma_2_2b_fp16.safetensors) |
| `clip/gemma_2_2b_it_bf16.safetensors` | `path_clip` | [Download](https://huggingface.co/unsloth/gemma-2-2b-it/resolve/main/model.safetensors) |
| `clip/gemma_2_2b_it_elm_bf16.safetensors` | `path_clip` | [Download](https://huggingface.co/Comfy-Org/PixelDiT/resolve/main/text_encoders/gemma_2_2b_it_elm_bf16.safetensors) |
| `clip/gemma_2_2b_it_elm_fp8_scaled.safetensors` | `path_clip` | [Download](https://huggingface.co/Comfy-Org/PixelDiT/resolve/main/text_encoders/gemma_2_2b_it_elm_fp8_scaled.safetensors) |
| `clip/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf` | `path_clip` | [Download](https://huggingface.co/unsloth/gemma-3-12b-it-qat-GGUF/resolve/main/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf) |
| `clip/gemma-3-12b-it-Q4_0.gguf` | `path_clip` | [Download](https://huggingface.co/unsloth/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q4_0.gguf) |
| `clip/gemma-3-12b-it-Q4_K_S.gguf` | `path_clip` | [Download](https://huggingface.co/unsloth/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q4_K_S.gguf) |
| `clip/gemma-3-12b-it-mmproj-F16.gguf` | `path_clip` | [Download](https://huggingface.co/unsloth/gemma-3-12b-it-GGUF/resolve/main/mmproj-F16.gguf) |
| `clip/gemma-3-4b-it-Q5_K_M.gguf` | `path_clip` | [Download](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q5_K_M.gguf) |
| `clip/gemma_3_4b_it_bf16.safetensors` | `path_clip` | [Download](https://huggingface.co/woctordho/comfyui-gemma-3-4b-it/resolve/main/gemma_3_4b_it_bf16.safetensors) |
| `clip/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | `path_clip` | [Download](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors) |
| `clip/qwen3vl_4b_bf16.safetensors` | `path_clip` | [Download](https://huggingface.co/Comfy-Org/Qwen3-VL/resolve/main/text_encoders/qwen3vl_4b_bf16.safetensors) |
| `clip/qwen3vl_4b_fp8_scaled.safetensors` | `path_clip` | [Download](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors) |
| `clip/qwen3vl_8b_fp8_scaled.safetensors` | `path_clip` | [Download](https://huggingface.co/Comfy-Org/Boogu-Image/resolve/main/text_encoders/qwen3vl_8b_fp8_scaled.safetensors) |
| `clip/Qwen3-VL-8B-Instruct-Q5_K_M.gguf` | `path_clip` | [Download](https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3-VL-8B-Instruct-Q5_K_M.gguf) |
| `clip/Qwen3VL-8B-Uncensored-HauhauCS-Aggressive-Q6_K.gguf` | `path_clip` | [Download](https://huggingface.co/HauhauCS/Qwen3VL-8B-Uncensored-HauhauCS-Aggressive/resolve/main/Qwen3VL-8B-Uncensored-HauhauCS-Aggressive-Q6_K.gguf) |
| `clip/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | `path_clip` | [Download](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors) |
| `clip/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `path_clip` | [Download](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) |
| `clip/jina_clip_v2_bf16.safetensors` | `path_clip` | [Download](https://huggingface.co/woctordho/comfyui-jina-clip-v2/resolve/main/jina_clip_v2_bf16.safetensors) |
| `clip/llama_q2.gguf` | `path_clip` | [Download](https://huggingface.co/calcuis/hidream-gguf/resolve/main/llama-q2_k.gguf) |
| `clip/llava_llama3_fp8_scaled.safetensors` | `path_clip` | [Download](https://huggingface.co/calcuis/hunyuan-gguf/resolve/main/llava_llama3_fp8_scaled.safetensors) |
| `clip/ltx-2.3-22b_text_projection_bf16.safetensors` | `path_clip` | [Download](https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors) |
| `clip/ltx-2.3-22b-dev_embeddings_connectors.safetensors` | `path_clip` | [Download](https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors) |
| `clip/ltx-2.3-22b-distilled_embeddings_connectors.safetensors` | `path_clip` | [Download](https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors) |
| `clip/ltx-2.3_text_projection_bf16.safetensors` | `path_clip` | [Download](https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors) |
| `clip/mistral_3_small_flux2_fp8.safetensors` | `path_clip` | [Download](https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors) |
| `clip/ministral-3-3b.safetensors` | `path_clip` | [Download](https://huggingface.co/Comfy-Org/ERNIE-Image/resolve/main/text_encoders/ministral-3-3b.safetensors) |
| `clip/Qwen2.5-VL-7B-Instruct-Q4_K_S.gguf` | `path_clip` | [Download](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-Q4_K_S.gguf) |
| `clip/qwen_2.5_vl_7b_edit-q2_k.gguf` | `path_clip` | [Download](https://huggingface.co/calcuis/pig-encoder/resolve/main/qwen_2.5_vl_7b_edit-q2_k.gguf) |
| `clip/qwen_2.5_vl_7b_fp8_scaled.safetensors` | `path_clip` | [Download](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors) |
| `clip/qwen_3_06b_base.safetensors` | `path_clip` | [Download](https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors) |
| `clip/qwen_3_4b.safetensors` | `path_clip` | [Download](https://huggingface.co/Comfy-Org/z_image_turbo/blob/main/split_files/text_encoders/qwen_3_4b.safetensors) |
| `clip/Qwen3-4B-Q2_K.gguf` | `path_clip` | [Download](https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q2_K.gguf) |
| `clip/Qwen3-4B-Q4_K_M.gguf` | `path_clip` | [Download](https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf) |
| `clip/Qwen3-4B-Q8_0.gguf` | `path_clip` | [Download](https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf) |
| `clip/Qwen3-8B-Q8_0.gguf` | `path_clip` | [Download](https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf) |
| `clip/Qwen3-8B-Q4_K_M.gguf` | `path_clip` | [Download](https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf) |
| `clip/t5xxl_old_fp32-q4_0.gguf` | `path_clip` | [Download](https://huggingface.co/calcuis/cosmos-predict2-gguf/resolve/main/t5xxl_old_fp32-q4_0.gguf) |
| `clip/t5-v1_1-xxl-encoder-Q3_K_L.gguf` | `path_clip` | [Download](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q3_K_L.gguf) |
| `clip/t5-v1_1-xxl-encoder-Q3_K_M.gguf` | `path_clip` | [Download](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q3_K_M.gguf) |
| `clip/t5-v1_1-xxl-encoder-Q3_K_S.gguf` | `path_clip` | [Download](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q3_K_S.gguf) |
| `clip/t5-v1_1-xxl-encoder-Q4_K_M.gguf` | `path_clip` | [Download](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q4_K_M.gguf) |
| `clip/t5-v1_1-xxl-encoder-Q4_K_S.gguf` | `path_clip` | [Download](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q4_K_S.gguf) |
| `clip/t5-v1_1-xxl-encoder-Q5_K_M.gguf` | `path_clip` | [Download](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q5_K_M.gguf) |
| `clip/t5-v1_1-xxl-encoder-Q5_K_S.gguf` | `path_clip` | [Download](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q5_K_S.gguf) |
| `clip/t5-v1_1-xxl-encoder-Q6_K.gguf` | `path_clip` | [Download](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q6_K.gguf) |
| `clip/t5-v1_1-xxl-encoder-Q8_0.gguf` | `path_clip` | [Download](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q8_0.gguf) |
| `clip/t5-v1_1-xxl-encoder-f16.gguf` | `path_clip` | [Download](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-f16.gguf) |
| `clip/t5-v1_1-xxl-encoder-f32.gguf` | `path_clip` | [Download](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-f32.gguf) |
| `clip/ltx-2-19b-embeddings_connector_dev_bf16.safetensors` | `path_clip` | [Download](https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/text_encoders/ltx-2-19b-embeddings_connector_dev_bf16.safetensors) |
| `clip/ltx-2-19b-embeddings_connector_distill_bf16.safetensors` | `path_clip` | [Download](https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors) |
| `clip/umt5_xxl_fp16.safetensors` | `path_clip` | [Download](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors) |
| `clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `path_clip` | [Download](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors) |

### clip_vision (2)
| Catalog key | Storage | Source |
| --- | --- | --- |
| `clip_vision/clip_vision_h_fp16.safetensors` | `path_clip_vision` | [Download](https://huggingface.co/calcuis/wan-gguf/resolve/main/clip_vision_h_fp16.safetensors) |
| `clip_vision/clip_vision_h_fp8_e4m3fn.safetensors` | `path_clip_vision` | [Download](https://huggingface.co/calcuis/wan-gguf/resolve/main/clip_vision_h_fp8_e4m3fn.safetensors) |

### controlnet (5)
| Catalog key | Storage | Source |
| --- | --- | --- |
| `cn_canny` | `path_controlnet` | [Download](https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-canny-rank128.safetensors) |
| `cn_depth` | `path_controlnet` | [Download](https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-depth-rank128.safetensors) |
| `cn_recolour` | `path_controlnet` | [Download](https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-recolor-rank128.safetensors) |
| `cn_sketch` | `path_controlnet` | [Download](https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-sketch-rank128-metadata.safetensors) |
| `4x-UltraSharp.pth` | `path_upscalers` | [Download](https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth) |

### faceswap (1)
| Catalog key | Storage | Source |
| --- | --- | --- |
| `faceswap/GFPGANv1.4.pth` | `path_faceswap` | [Download](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth) |

### latent_upscalers (1)
| Catalog key | Storage | Source |
| --- | --- | --- |
| `latent_upscalers/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | `path_latent_upscalers` | [Download](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors) |

### lcm (1)
| Catalog key | Storage | Source |
| --- | --- | --- |
| `lcm_lora` | `path_loras` | [Download](https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors) |

### llm (83)
| Catalog key | Storage | Source |
| --- | --- | --- |
| `llm/gpt-oss-20b-MXFP4.gguf` | `path_llm` | [Download](https://huggingface.co/lmstudio-community/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-MXFP4.gguf) |
| `llm/DeepSeek-R1-Distill-Llama-8B-F16.gguf` | `path_llm` | [Download](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-F16.gguf) |
| `llm/DeepSeek-R1-Distill-Llama-8B-Q2_K.gguf` | `path_llm` | [Download](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q2_K.gguf) |
| `llm/DeepSeek-R1-Distill-Llama-8B-Q2_K_L.gguf` | `path_llm` | [Download](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q2_K_L.gguf) |
| `llm/DeepSeek-R1-Distill-Llama-8B-Q3_K_M.gguf` | `path_llm` | [Download](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q3_K_M.gguf) |
| `llm/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf` | `path_llm` | [Download](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf) |
| `llm/DeepSeek-R1-Distill-Llama-8B-Q5_K_M.gguf` | `path_llm` | [Download](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q5_K_M.gguf) |
| `llm/DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf` | `path_llm` | [Download](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf) |
| `llm/DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf` | `path_llm` | [Download](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf) |
| `llm/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-IQ4_XS.gguf` | `path_llm` | [Download](https://huggingface.co/DavidAU/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-GGUF/resolve/main/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-IQ4_XS.gguf) |
| `llm/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-Q2_k.gguf` | `path_llm` | [Download](https://huggingface.co/DavidAU/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-GGUF/resolve/main/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-Q2_k.gguf) |
| `llm/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-Q3_k_m.gguf` | `path_llm` | [Download](https://huggingface.co/DavidAU/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-GGUF/resolve/main/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-Q3_k_m.gguf) |
| `llm/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-Q3_k_s.gguf` | `path_llm` | [Download](https://huggingface.co/DavidAU/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-GGUF/resolve/main/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-Q3_k_s.gguf) |
| `llm/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-Q4_k_m.gguf` | `path_llm` | [Download](https://huggingface.co/DavidAU/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-GGUF/resolve/main/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-Q4_k_m.gguf) |
| `llm/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-Q4_k_s.gguf` | `path_llm` | [Download](https://huggingface.co/DavidAU/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-GGUF/resolve/main/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-Q4_k_s.gguf) |
| `llm/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-Q5_k_s.gguf` | `path_llm` | [Download](https://huggingface.co/DavidAU/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-GGUF/resolve/main/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-Q5_k_s.gguf) |
| `llm/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-Q6_k.gguf` | `path_llm` | [Download](https://huggingface.co/DavidAU/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-GGUF/resolve/main/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-Q6_k.gguf) |
| `llm/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-Q8_0.gguf` | `path_llm` | [Download](https://huggingface.co/DavidAU/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-GGUF/resolve/main/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-Q8_0.gguf) |
| `llm/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-q5_k_m.gguf` | `path_llm` | [Download](https://huggingface.co/DavidAU/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-GGUF/resolve/main/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-q5_k_m.gguf) |
| `llm/Llama-3.2-3B-Instruct-uncensored-IQ3_M.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF/resolve/main/Llama-3.2-3B-Instruct-uncensored-IQ3_M.gguf) |
| `llm/Llama-3.2-3B-Instruct-uncensored-IQ3_XS.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF/resolve/main/Llama-3.2-3B-Instruct-uncensored-IQ3_XS.gguf) |
| `llm/Llama-3.2-3B-Instruct-uncensored-IQ4_XS.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF/resolve/main/Llama-3.2-3B-Instruct-uncensored-IQ4_XS.gguf) |
| `llm/Llama-3.2-3B-Instruct-uncensored-Q2_K.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF/resolve/main/Llama-3.2-3B-Instruct-uncensored-Q2_K.gguf) |
| `llm/Llama-3.2-3B-Instruct-uncensored-Q2_K_L.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF/resolve/main/Llama-3.2-3B-Instruct-uncensored-Q2_K_L.gguf) |
| `llm/Llama-3.2-3B-Instruct-uncensored-Q3_K_L.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF/resolve/main/Llama-3.2-3B-Instruct-uncensored-Q3_K_L.gguf) |
| `llm/Llama-3.2-3B-Instruct-uncensored-Q3_K_M.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF/resolve/main/Llama-3.2-3B-Instruct-uncensored-Q3_K_M.gguf) |
| `llm/Llama-3.2-3B-Instruct-uncensored-Q3_K_S.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF/resolve/main/Llama-3.2-3B-Instruct-uncensored-Q3_K_S.gguf) |
| `llm/Llama-3.2-3B-Instruct-uncensored-Q3_K_XL.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF/resolve/main/Llama-3.2-3B-Instruct-uncensored-Q3_K_XL.gguf) |
| `llm/Llama-3.2-3B-Instruct-uncensored-Q4_0.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF/resolve/main/Llama-3.2-3B-Instruct-uncensored-Q4_0.gguf) |
| `llm/Llama-3.2-3B-Instruct-uncensored-Q4_K_L.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF/resolve/main/Llama-3.2-3B-Instruct-uncensored-Q4_K_L.gguf) |
| `llm/Llama-3.2-3B-Instruct-uncensored-Q4_K_M.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF/resolve/main/Llama-3.2-3B-Instruct-uncensored-Q4_K_M.gguf) |
| `llm/Llama-3.2-3B-Instruct-uncensored-Q4_K_S.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF/resolve/main/Llama-3.2-3B-Instruct-uncensored-Q4_K_S.gguf) |
| `llm/Llama-3.2-3B-Instruct-uncensored-Q5_K_L.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF/resolve/main/Llama-3.2-3B-Instruct-uncensored-Q5_K_L.gguf) |
| `llm/Llama-3.2-3B-Instruct-uncensored-Q5_K_M.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF/resolve/main/Llama-3.2-3B-Instruct-uncensored-Q5_K_M.gguf) |
| `llm/Llama-3.2-3B-Instruct-uncensored-Q5_K_S.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF/resolve/main/Llama-3.2-3B-Instruct-uncensored-Q5_K_S.gguf) |
| `llm/Llama-3.2-3B-Instruct-uncensored-Q6_K.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF/resolve/main/Llama-3.2-3B-Instruct-uncensored-Q6_K.gguf) |
| `llm/Llama-3.2-3B-Instruct-uncensored-Q6_K_L.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF/resolve/main/Llama-3.2-3B-Instruct-uncensored-Q6_K_L.gguf) |
| `llm/Llama-3.2-3B-Instruct-uncensored-Q8_0.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF/resolve/main/Llama-3.2-3B-Instruct-uncensored-Q8_0.gguf) |
| `llm/Llama-3.2-3B-Instruct-uncensored-f16.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF/resolve/main/Llama-3.2-3B-Instruct-uncensored-f16.gguf) |
| `llm/OpenThinker-7B-IQ2_M.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-IQ2_M.gguf) |
| `llm/OpenThinker-7B-IQ3_M.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-IQ3_M.gguf) |
| `llm/OpenThinker-7B-IQ3_XS.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-IQ3_XS.gguf) |
| `llm/OpenThinker-7B-IQ4_NL.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-IQ4_NL.gguf) |
| `llm/OpenThinker-7B-IQ4_XS.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-IQ4_XS.gguf) |
| `llm/OpenThinker-7B-Q2_K.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-Q2_K.gguf) |
| `llm/OpenThinker-7B-Q2_K_L.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-Q2_K_L.gguf) |
| `llm/OpenThinker-7B-Q3_K_L.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-Q3_K_L.gguf) |
| `llm/OpenThinker-7B-Q3_K_M.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-Q3_K_M.gguf) |
| `llm/OpenThinker-7B-Q3_K_S.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-Q3_K_S.gguf) |
| `llm/OpenThinker-7B-Q3_K_XL.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-Q3_K_XL.gguf) |
| `llm/OpenThinker-7B-Q4_0.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-Q4_0.gguf) |
| `llm/OpenThinker-7B-Q4_1.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-Q4_1.gguf) |
| `llm/OpenThinker-7B-Q4_K_L.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-Q4_K_L.gguf) |
| `llm/OpenThinker-7B-Q4_K_M.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-Q4_K_M.gguf) |
| `llm/OpenThinker-7B-Q4_K_S.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-Q4_K_S.gguf) |
| `llm/OpenThinker-7B-Q5_K_L.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-Q5_K_L.gguf) |
| `llm/OpenThinker-7B-Q5_K_M.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-Q5_K_M.gguf) |
| `llm/OpenThinker-7B-Q5_K_S.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-Q5_K_S.gguf) |
| `llm/OpenThinker-7B-Q6_K.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-Q6_K.gguf) |
| `llm/OpenThinker-7B-Q6_K_L.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-Q6_K_L.gguf) |
| `llm/OpenThinker-7B-Q8_0.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-Q8_0.gguf) |
| `llm/OpenThinker-7B-f16.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-f16.gguf) |
| `llm/OpenThinker-7B-f32.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/OpenThinker-7B-GGUF/resolve/main/OpenThinker-7B-f32.gguf) |
| `llm/PokeeAI_pokee_research_7b-IQ2_M.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/PokeeAI_pokee_research_7b-GGUF/resolve/main/PokeeAI_pokee_research_7b-IQ2_M.gguf) |
| `llm/PokeeAI_pokee_research_7b-Q4_K_S.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/PokeeAI_pokee_research_7b-GGUF/resolve/main/PokeeAI_pokee_research_7b-Q4_K_S.gguf) |
| `llm/PokeeAI_pokee_research_7b-Q5_K_S.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/PokeeAI_pokee_research_7b-GGUF/resolve/main/PokeeAI_pokee_research_7b-Q5_K_S.gguf) |
| `llm/PokeeAI_pokee_research_7b-Q8_0.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/PokeeAI_pokee_research_7b-GGUF/resolve/main/PokeeAI_pokee_research_7b-Q8_0.gguf) |
| `llm/PokeeAI_pokee_research_7b-bf16.gguf` | `path_llm` | [Download](https://huggingface.co/bartowski/PokeeAI_pokee_research_7b-GGUF/resolve/main/PokeeAI_pokee_research_7b-bf16.gguf) |
| `llm/Qwen2.5-7B-Instruct-abliterated-v2.IQ3_M.gguf` | `path_llm` | [Download](https://huggingface.co/mradermacher/Qwen2.5-7B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen2.5-7B-Instruct-abliterated-v2.IQ3_M.gguf) |
| `llm/Qwen2.5-7B-Instruct-abliterated-v2.IQ3_S.gguf` | `path_llm` | [Download](https://huggingface.co/mradermacher/Qwen2.5-7B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen2.5-7B-Instruct-abliterated-v2.IQ3_S.gguf) |
| `llm/Qwen2.5-7B-Instruct-abliterated-v2.IQ3_XS.gguf` | `path_llm` | [Download](https://huggingface.co/mradermacher/Qwen2.5-7B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen2.5-7B-Instruct-abliterated-v2.IQ3_XS.gguf) |
| `llm/Qwen2.5-7B-Instruct-abliterated-v2.IQ4_XS.gguf` | `path_llm` | [Download](https://huggingface.co/mradermacher/Qwen2.5-7B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen2.5-7B-Instruct-abliterated-v2.IQ4_XS.gguf) |
| `llm/Qwen2.5-7B-Instruct-abliterated-v2.Q2_K.gguf` | `path_llm` | [Download](https://huggingface.co/mradermacher/Qwen2.5-7B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen2.5-7B-Instruct-abliterated-v2.Q2_K.gguf) |
| `llm/Qwen2.5-7B-Instruct-abliterated-v2.Q3_K_L.gguf` | `path_llm` | [Download](https://huggingface.co/mradermacher/Qwen2.5-7B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen2.5-7B-Instruct-abliterated-v2.Q3_K_L.gguf) |
| `llm/Qwen2.5-7B-Instruct-abliterated-v2.Q3_K_M.gguf` | `path_llm` | [Download](https://huggingface.co/mradermacher/Qwen2.5-7B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen2.5-7B-Instruct-abliterated-v2.Q3_K_M.gguf) |
| `llm/Qwen2.5-7B-Instruct-abliterated-v2.Q3_K_S.gguf` | `path_llm` | [Download](https://huggingface.co/mradermacher/Qwen2.5-7B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen2.5-7B-Instruct-abliterated-v2.Q3_K_S.gguf) |
| `llm/Qwen2.5-7B-Instruct-abliterated-v2.Q4_K_M.gguf` | `path_llm` | [Download](https://huggingface.co/mradermacher/Qwen2.5-7B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen2.5-7B-Instruct-abliterated-v2.Q4_K_M.gguf) |
| `llm/Qwen2.5-7B-Instruct-abliterated-v2.Q4_K_S.gguf` | `path_llm` | [Download](https://huggingface.co/mradermacher/Qwen2.5-7B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen2.5-7B-Instruct-abliterated-v2.Q4_K_S.gguf) |
| `llm/Qwen2.5-7B-Instruct-abliterated-v2.Q5_K_M.gguf` | `path_llm` | [Download](https://huggingface.co/mradermacher/Qwen2.5-7B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen2.5-7B-Instruct-abliterated-v2.Q5_K_M.gguf) |
| `llm/Qwen2.5-7B-Instruct-abliterated-v2.Q5_K_S.gguf` | `path_llm` | [Download](https://huggingface.co/mradermacher/Qwen2.5-7B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen2.5-7B-Instruct-abliterated-v2.Q5_K_S.gguf) |
| `llm/Qwen2.5-7B-Instruct-abliterated-v2.Q6_K.gguf` | `path_llm` | [Download](https://huggingface.co/mradermacher/Qwen2.5-7B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen2.5-7B-Instruct-abliterated-v2.Q6_K.gguf) |
| `llm/Qwen2.5-7B-Instruct-abliterated-v2.Q8_0.gguf` | `path_llm` | [Download](https://huggingface.co/mradermacher/Qwen2.5-7B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen2.5-7B-Instruct-abliterated-v2.Q8_0.gguf) |
| `llm/Qwen2.5-7B-Instruct-abliterated-v2.f16.gguf` | `path_llm` | [Download](https://huggingface.co/mradermacher/Qwen2.5-7B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen2.5-7B-Instruct-abliterated-v2.f16.gguf) |

### lora (2)
| Catalog key | Storage | Source |
| --- | --- | --- |
| `loras/SDXL 1.0/dmd2_sdxl_4step_lora.safetensors` | `path_loras` | [Download](https://huggingface.co/tianweiy/DMD2/resolve/main/dmd2_sdxl_4step_lora.safetensors) |
| `loras/SDXL 1.0/dmd2_sdxl_4step_lora_fp16.safetensors` | `path_loras` | [Download](https://huggingface.co/tianweiy/DMD2/resolve/main/dmd2_sdxl_4step_lora_fp16.safetensors) |

### upscalers (36)
| Catalog key | Storage | Source |
| --- | --- | --- |
| `RealESRGAN_x2plus.pth` | `path_upscalers` | [Download](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth) |
| `RealESRGAN_x4plus.pth` | `path_upscalers` | [Download](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) |
| `RealESRGAN_x4plus_anime_6B.pth` | `path_upscalers` | [Download](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth) |
| `DAT-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/DAT-4x.pth) |
| `DAT-Helaman-LSDIR-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/DAT-Helaman-LSDIR-4x.pth) |
| `DAT-Helaman-Nomos-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/DAT-Helaman-Nomos-4x.pth) |
| `DAT-Helaman-SSDIR-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/DAT-Helaman-SSDIR-4x.pth) |
| `ESRGAN-BigFace-v3-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/ESRGAN-BigFace-v3-4x.pth) |
| `ESRGAN-Box-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/ESRGAN-Box-4x.pth) |
| `ESRGAN-Helaman-HFA2k-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/ESRGAN-Helaman-HFA2k-4x.pth) |
| `ESRGAN-Helaman-LSDIRplus-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/ESRGAN-Helaman-LSDIRplus-4x.pth) |
| `ESRGAN-HugePaint-8x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/ESRGAN-HugePaint-8x.pth) |
| `ESRGAN-NMKD-Siax-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/ESRGAN-NMKD-Siax-4x.pth) |
| `ESRGAN-NMKD-Superscale-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/ESRGAN-NMKD-Superscale-4x.pth) |
| `ESRGAN-NMKD-Superscale-8x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/ESRGAN-NMKD-Superscale-8x.pth) |
| `ESRGAN-NMKD-YandereNeoXL-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/ESRGAN-NMKD-YandereNeoXL-4x.pth) |
| `ESRGAN-Remacri-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/ESRGAN-Remacri-4x.pth) |
| `ESRGAN-UltraSharp-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/ESRGAN-UltraSharp-4x.pth) |
| `ESRGAN-Valar-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/ESRGAN-Valar-4x.pth) |
| `HAT-2x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/HAT-2x.pth) |
| `HAT-3x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/HAT-3x.pth) |
| `HAT-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/HAT-4x.pth) |
| `HAT-Helaman-Lexica-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/HAT-Helaman-Lexica-4x.pth) |
| `HAT-Helaman-Nomos8kL-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/HAT-Helaman-Nomos8kL-4x.pth) |
| `HAT-L-2x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/HAT-L-2x.pth) |
| `HAT-L-3x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/HAT-L-3x.pth) |
| `HAT-L-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/HAT-L-4x.pth) |
| `OmniSR-Helaman-HFA2k-2x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/OmniSR-Helaman-HFA2k-2x.pth) |
| `RRDBNet-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/RRDBNet-4x.pth) |
| `RealHAT-GAN-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/RealHAT-GAN-4x.pth) |
| `RealHAT-Sharper-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/RealHAT-Sharper-4x.pth) |
| `SRFormer-Light-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/SRFormer-Light-4x.pth) |
| `SRFormer-Nomos-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/SRFormer-Nomos-4x.pth) |
| `SwiftSR-2x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/SwiftSR-2x.pth) |
| `SwiftSR-4x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/SwiftSR-4x.pth) |
| `SwinIR-Helaman-Lexica-2x.pth` | `path_upscalers` | [Download](https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/SwinIR-Helaman-Lexica-2x.pth) |

### vae (34)
| Catalog key | Storage | Source |
| --- | --- | --- |
| `vae/ae.safetensors` | `path_vae` | [Download](https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/ae.safetensors) |
| `vae/auraflow_vae_fp32.safetensors` | `path_vae` | [Download](https://huggingface.co/fal/AuraFlow-v0.3/resolve/main/vae/diffusion_pytorch_model.safetensors) |
| `vae/flux-vae.safetensors` | `path_vae` | [Download](https://huggingface.co/diffusers/FLUX.1-vae/resolve/main/diffusion_pytorch_model.safetensors) |
| `vae/pig_flux2_vae_fp32-f16.gguf` | `path_vae` | [Download](https://huggingface.co/chatpig/flux2-dev-gguf/resolve/main/pig_flux2_vae_fp32-f16.gguf) |
| `vae/flux2-vae.safetensors` | `path_vae` | [Download](https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors) |
| `vae/sdxl_vae.safetensors` | `path_vae` | [Download](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/vae/diffusion_pytorch_model.safetensors) |
| `vae/sd3_vae.safetensors` | `path_vae` | [Download](https://huggingface.co/calcuis/sd3.5-large-gguf/resolve/main/diffusion_pytorch_model.safetensors) |
| `vae/lumina2_vae_fp8.safetensors` | `path_vae` | [Download](https://huggingface.co/calcuis/lumina-gguf/resolve/main/lumina2_vae_fp8.safetensors) |
| `vae/lumina2_vae_fp16.safetensors` | `path_vae` | [Download](https://huggingface.co/calcuis/lumina-gguf/resolve/main/lumina2_vae_fp16.safetensors) |
| `vae/lumina2_vae_fp32.safetensors` | `path_vae` | [Download](https://huggingface.co/calcuis/lumina-gguf/resolve/main/lumina2_vae_fp32.safetensors) |
| `vae/mage_flow_vae.safetensors` | `path_vae` | [Download](https://huggingface.co/Comfy-Org/Mage-Flow/resolve/main/vae/mage_flow_vae_bf16.safetensors) |
| `vae/mage_flow_vae_bf16.safetensors` | `path_vae` | [Download](https://huggingface.co/Comfy-Org/Mage-Flow/resolve/main/vae/mage_flow_vae_bf16.safetensors) |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | `path_vae` | [Download](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) |
| `vae/minimax_h3_video_vae_fp16.safetensors` | `path_vae` | [Download](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors) |
| `vae/hunyuan_video_vae_bf16.safetensors` | `path_vae` | [Download](https://huggingface.co/calcuis/hunyuan-gguf/resolve/main/hunyuan_video_vae_bf16.safetensors) |
| `vae/hunyuanvideo15_vae_fp16.safetensors` | `path_vae` | [Download](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/vae/hunyuanvideo15_vae_fp16.safetensors) |
| `vae/wan_2.1_vae.safetensors` | `path_vae` | [Download](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors) |
| `vae/LTX-Video-VAE.safetensors` | `path_vae` | [Download](https://huggingface.co/Lightricks/LTX-Video/resolve/main/vae/diffusion_pytorch_model.safetensors) |
| `vae/LTX-Video-0.9.6-VAE-BF16.safetensors` | `path_vae` | [Download](https://huggingface.co/city96/LTX-Video-0.9.6-dev-gguf/resolve/main/LTX-Video-0.9.6-VAE-BF16.safetensors) |
| `vae/pig_video_97_vae_fp32-f16.gguf` | `path_vae` | [Download](https://huggingface.co/calcuis/ltxv0.9.7-gguf/resolve/main/pig_video_97_vae_fp32-f16.gguf) |
| `vae/pig_wan_2.2_vae_fp32-f16.gguf` | `path_vae` | [Download](https://huggingface.co/calcuis/wan2-gguf/resolve/main/pig_wan2_vae_fp32-f16.gguf) |
| `vae/wan2.2_vae.safetensors` | `path_vae` | [Download](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors) |
| `vae/pixart_sigma_vae_fp16.safetensors` | `path_vae` | [Download](https://huggingface.co/calcuis/pixart/resolve/main/pixart_sigma_vae_fp16.safetensors) |
| `vae/pixart_vae_fp16.safetensors` | `path_vae` | [Download](https://huggingface.co/calcuis/pixart/resolve/main/pixart_vae_fp16.safetensors) |
| `vae/qwen_image_vae.safetensors` | `path_vae` | [Download](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors) |
| `vae/pig_wan_vae_fp32-f16.gguf` | `path_vae` | [Download](https://huggingface.co/calcuis/cosmos-predict2-gguf/resolve/main/pig_wan_vae_fp32-f16.gguf) |
| `vae/sd15_vae.safetensors` | `path_vae` | [Download](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/vae/diffusion_pytorch_model.safetensors) |
| `vae/LTX2_audio_vae_bf16.safetensors` | `path_vae` | [Download](https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/VAE/LTX2_audio_vae_bf16.safetensors) |
| `vae/LTX2_video_vae_bf16.safetensors` | `path_vae` | [Download](https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/VAE/LTX2_video_vae_bf16.safetensors) |
| `vae/LTX23_audio_vae_bf16.safetensors` | `path_vae` | [Download](https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors) |
| `vae/LTX23_video_vae_bf16.safetensors` | `path_vae` | [Download](https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors) |
| `vae/ltx-2.5-audio-vae-bf16.safetensors` | `path_vae` | [Download](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-audio-vae-bf16.safetensors) |
| `vae/ltx-2.5-video-vae-bf16.safetensors` | `path_vae` | [Download](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-video-vae-conv-bf16.safetensors) |
| `vae/taeltx_2.safetensors` | `path_vae` | [Download](https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/VAE/taeltx_2.safetensors) |
