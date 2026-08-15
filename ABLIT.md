# Abliterating Vontra MXFP4-MLX on the Mac

Mida / 32-32 recipe applied to [`Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX`](https://huggingface.co/Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX).

**Published weights:** [`drowzeys/keys-Mac-DeepSeek-V4-Flash-0731-MXFP4-MLX-Abliterated`](https://huggingface.co/drowzeys/keys-Mac-DeepSeek-V4-Flash-0731-MXFP4-MLX-Abliterated) (gated).

## Why not a drop-in of the DGX FP8 pack

The Mac checkpoint is a **bit-exact byte transplant** of official 0731 into MLX packing:

| tensors | packing |
|---|---|
| routed experts | MXFP4 (`uint32` lanes + e8m0 / 32) |
| attention / shared / indexer, including `attn.wo_b` | MXFP8 (`uint32` 4096×2048 + e8m0 4096×256) |
| embed / head / norms / routers | BF16 / F32 |

`attn.wo_b` is the mHC-resistant edit surface used on DGX. We dequant that MXFP8 with `mlx.dequantize(..., mode="mxfp8")`, project, then **keep the original e8m0 scales** and only re-encode e4m3. `mx.quantize` is the wrong tool here — it re-derives per-32 scales and adds ~5% Frobenius noise on an untouched tensor.

## Recipe

| | |
|--|--|
| Direction | `ablit/refusal_direction_reablit_20260726.npz` (same vector as the published 0731 32/32 pack) |
| Layers | **L10–42** `attn.wo_b` only |
| Stock | **L0–9** + **all MTP** (`mtp.0/1/2.attn.wo_b` untouched) |
| λ | **3.5** · k=**1** |
| Edits | 33 tensors · mean Δrel **0.0557** (DGX FP8 pack was 0.0556) |

```bash
python3 scripts/project_wob_mlx.py \
  --src ~/.omlx/models/dsv4f-mxfp4 \
  --dst ~/models/dsv4f-mxfp4-ablit-mida \
  --direction ablit/refusal_direction_reablit_20260726.npz \
  --lambda-attn 3.5 --min-layer 10 --max-layer 42 --n-directions 1
```

Unchanged shards are hardlinked to the resolved HF blob. Edited shards (24 × ~5 GB) are `cp -c` APFS clones, then only the `wo_b.weight` bytes are patched. The Vontra snapshot is never written through.

## Live on the Studio (2026-08-15)

- Host: `cityhunter@192.168.1.243` · oMLX 0.5.7 · `:11500`
- Weights: `~/models/dsv4f-mxfp4-ablit-mida`
- Recipe name `dsv4f-mxfp4` points at the ablit dir; stock is `dsv4f-mxfp4-stock`
- Refusal: **32/32 BYPASS** · 0 refuse · 0 garble (`ablit/refusal32-summary.json`)
- DSpark still fires (count 220 tok: accept 100%, tok/cycle 3.73)

Set `enable_thinking: false` in `~/.omlx/model_settings.json` (this repo's `config/model_settings.json`) or pass `chat_template_kwargs.enable_thinking=false` on each request.

Numbers: [ablit/RESULTS.md](ablit/RESULTS.md).

## Hermes first-prompt stall

Fat Hermes system (~20k tokens) + abliterated DSV4 looks hung on `hello`. Measured here: 20,344 prompt tokens, 59.7 s blank UI, plus a concurrent title-generation job on the same 163 GB model. The greeting did return; it was not a refusal hang and it did not dump the skills catalog (the older Mida first-turn failure mode).

Fix:

1. `auxiliary.title_generation.enabled: false` so Hermes does not steal the Mac on turn 1
2. `streaming.enabled: true` so tokens appear instead of a minute of nothing
3. oMLX `cache.hot_cache_max_size: "32GB"` (stock is `"0"` = off) then `omlx restart`
4. `enable_thinking: false` on the model
5. `/new` after any of the above

## 1M context

Native window is **1,048,576** (YaRN 64k × 16). Pin `max_context_window: 1048576` on the model and `sampling.max_context_window: 1048576` in oMLX; set Hermes `context_length` to the same. Advertised live as `max_model_len=1048576`.

Proven here: **41,646-token** needle retrieval, exact hit. A full 1M prompt has not been soak-tested. ~159 GB weights + ~88 GB estimated KV at 1M exceeds the ~223 GB Metal cap unless paged SSD KV carries it. 128–256k is the comfortable band on a 256 GB Studio.

## Responsible use

Safety refusals are removed. Same terms as the other Keys 0731 abliterated releases: research / red-team / evaluation. You supply filtering and access control. Do not use for anything involving minors, self-harm, or other prohibited uses in those cards.
