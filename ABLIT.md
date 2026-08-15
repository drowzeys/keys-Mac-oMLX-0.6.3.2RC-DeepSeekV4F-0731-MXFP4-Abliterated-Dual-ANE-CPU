# Abliterating Vontra MXFP4-MLX on the Mac

Mida / 32-32 recipe applied to [`Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX`](https://huggingface.co/Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX).

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

## Responsible use

Safety refusals are removed. Same terms as the other Keys 0731 abliterated releases: research / red-team / evaluation. You supply filtering and access control. Do not use for anything involving minors, self-harm, or other prohibited uses in those cards.
