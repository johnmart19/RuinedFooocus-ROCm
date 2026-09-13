# Checkpoint tools

Under **⋮ → PowerUp → Cheat Code**, choose **Checkpoint tools** to inspect the
selected model, check for NaN/Inf weights, or create a separate FP16 copy.
Conversion only reduces FP32 tensors, preserves metadata/previews in the cache, and keeps
the original. It is lossy; test the copy before choosing it for generation.
Checks are local and do not establish a model's training history or quality.
