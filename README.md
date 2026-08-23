# keys-Mac-oMLX-0.6.3.2RC-DeepSeekV4F-0731-MXFP4-Abliterated-Dual-ANE-CPU

**1.0 Beta** · Mac Studio M3 Ultra · oMLX **0.6.3rc2** (pack name 0.6.3.2RC) · DeepSeek-V4-Flash **0731** MXFP4 abliterated · Dual-ANE + CPU prefill · DSpark MTP · 1M context

Previous 49 tok/s / oMLX 0.5.7 pack: [`previous-version/`](previous-version/README.md)

These weights have safety refusals removed. Research / red-team only — you supply the guardrails.

## Big thanks

This recipe stands on other people's work. Please star them.

- **[jundot/omlx](https://github.com/jundot/omlx)** — oMLX, MTP/DSpark on Mac, Dual-ANE prefill  
- **[DeepSeek-AI](https://www.deepseek.com/)** — V4-Flash 0731 + DSpark  
- **[Vontra MXFP4-MLX](https://huggingface.co/Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX)** — 4-bit Mac quant  
- **[ml-explore/MLX](https://github.com/ml-explore/mlx)**  
- Spark cousins: **[Anemll](https://github.com/Anemll/dspark-vllm-gx10)**, **[MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)**, **[Tony / tonyd2wild](https://github.com/tonyd2wild)**  

Full list: **[CREDITS.md](CREDITS.md)**. Please donate / support: **[github.com/sponsors/drowzeys](https://github.com/sponsors/drowzeys)**.

## Headline (measured 2026-08-23, M3 Ultra 256 GB)

Keep **4-bit MXFP4**. Dual-ANE is **prefill-only** on DSV4 (decode is noise vs ANE-off).

### Decode (real-serve, thinking off, post-TTFT) — Dual-ANE+CPU, L10–35 anchorstock id `dsv4f-mxfp4-ablit`

| Task | C1 median decode |
|---|---:|
| **list** | **53.8 tok/s** |
| **count** | **52.3 tok/s** |
| **structural** | **49.8 tok/s** |
| **reading** | **32.2 tok/s** |
| **essay** | **29.6 tok/s** |

### Prefill (uncached ~10k tokens, nonce-first, cached=0)

| | tok/s |
|---|---:|
| Dual-ANE+CPU | **~454** |
| ANE off | **~422** |
| Delta | **~+8%** prefill only |

First request after load can be slower (~356). Prefix-cache second runs (~1114 tok/s on 5.6k) are **not** Dual-ANE wins.

### Archived MTP server-log (previous-version, DSpark accept-heavy)

count **48.9** / list **44.8** / read **49.7** tok/s on 4-bit MXFP4. Different protocol — do not mix with the table above.

## One-shot

```bash
git clone https://github.com/drowzeys/keys-Mac-oMLX-0.6.3.2RC-DeepSeekV4F-0731-MXFP4-Abliterated-Dual-ANE-CPU.git
cd keys-Mac-oMLX-0.6.3.2RC-DeepSeekV4F-0731-MXFP4-Abliterated-Dual-ANE-CPU
# Python 3.11, oMLX 0.6.3rc2, gated HF pack
bash scripts/setup-mac.sh
omlx serve --model-dir ~/.omlx/models --host 0.0.0.0 --port 11500
```

Agent recipe: [AGENTS.md](AGENTS.md)

GHCR carrier (recipe + benches, **not** a Metal runtime):

```bash
docker run --rm -v "$PWD":/out ghcr.io/drowzeys/keys-mac-dsv4f-mxfp4-dualane:1.0-beta \
  cp -r /omlx/. /out/
```

## Stack

| Piece | Value |
|---|---|
| Engine | oMLX **0.6.3rc2** `:11500` |
| Weights | [`drowzeys/keys-Mac-DeepSeek-V4-Flash-0731-MXFP4-MLX-Abliterated`](https://huggingface.co/drowzeys/keys-Mac-DeepSeek-V4-Flash-0731-MXFP4-MLX-Abliterated) (gated, ~156 GB) |
| Ablit | **L10–35** `wo_b` anchorstock (DSpark 36–42 stock) — champion match; mida L10–42 still in previous-version notes |
| Dual-ANE | `deepseek_ane_prefill_*` · CPU 0.125 / 12 threads · ~88 Dual-ANE layers |
| Context | **1,048,576** |
| MTP | **on** (`mtp_enabled: true`) · thinking **off** |

Aday777 NVFP4 Dual-ANE is a no-op on this path. Stay on MXFP4.

## Credits

See **[CREDITS.md](CREDITS.md)**. Pack: drowzeys / keys.
