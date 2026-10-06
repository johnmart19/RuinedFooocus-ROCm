Upstream: https://github.com/kgonia/ComfyUI-MiniMaxH3TextEncoders
Revision: 9c56af417a41191375247770b20ec94435675ed1
Loader source license: MIT (LICENSE alongside the source).
Encoder weights are separately released under Apache-2.0 by SearchingMan.

Only the recovered 8B loader is retained. Unused pruned-24 support, node UI and
registration code are removed; retained loading/conditioning functions are
unchanged. RuinedFooocus resolves its package through the model database and
calls the loader with the configured package directory.
Upstream verifies the base, ARA and conditioning-adapter SHA-256 digests and
rejects missing/unexpected weights. Do not replace it with a plain CLIPLoader:
the recovered encoder requires its 4096-to-5120 adapter and verbatim tokenizer.
