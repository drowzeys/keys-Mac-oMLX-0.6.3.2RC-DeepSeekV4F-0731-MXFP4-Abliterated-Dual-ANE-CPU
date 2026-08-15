# DeepSeek-V4-Flash + DSpark on a Mac Studio — Abliterated, 49 tok/s, 32/32 refusal bypass

Running **abliterated DeepSeek-V4-Flash-0731** with **DSpark speculative decoding** natively on Apple Silicon (M3 Ultra) via [oMLX](https://github.com/jundot/omlx), a compiled DSA Metal kernel, and the 4-bit MXFP4 checkpoint that keeps the MTP draft heads.

**Headline:** same 44–49 tok/s Mac Studio recipe as before, plus the Mida L10–42 `attn.wo_b` abliteration (**32/32 BYPASS**, 0 refuse, 0 garble) and the Hermes first-prompt fix so a `hello` does not sit blank for a minute.

These weights have safety refusals removed. Research / red-team only — you supply the guardrails.

## Benchmark (oMLX + DSA kernel + MTP on, server-log tok/s)

| task | 2.4-bit base | **4-bit MXFP4 base** | accept (4-bit) |
|------|-------------:|---------------------:|---------------:|
| count (200 tok)        | ~35  | **48.9** | ~95% |
| list (220 tok)         | 36.0 | **44.8** | 85.4% |
| read (60 tok, 5.8K prompt) | 36.4 | **49.7** | 95.2% |
| essay (400 tok)        | 30.5 | 31.9 | 62.2% |

Reference points on the same model (0731):
- base decode, **no** speculative: ~31–32 tok/s
- 2× DGX-Spark, fp8, vLLM + DSpark, TP=2 (our prior champion): **42.9 tok/s**

**On count / list / read the Mac hits 44–49 tok/s, beating the 42.9 cluster.** Free-prose (essay) stays ~32 tok/s — spec-decode acceptance is task-dependent, and open-ended prose drafts far worse than structured/factual output (accept 62% vs 85–95%).

### Why 4-bit is *faster* than 2.4-bit here (counter-intuitive)
Higher precision makes the target's next-token distribution sharper, so the DSpark drafter accepts much deeper: **accept 62–82% → 85–95%, tok/cycle 2.07 → 3.33** (depth-3 drafts landing ~85%). The base forward isn't purely bandwidth-bound, so drafting 3+ tokens per target pass far outweighs the 1.8× extra weight bytes.

## Hardware / software
- **Mac:** Mac Studio M3 Ultra, 256 GB unified memory, macOS 26.4
- **oMLX** built from source with the custom Metal kernel (`OMLX_WITH_CUSTOM_KERNEL=1`), Python 3.11
- **Model:** [`Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX`](https://huggingface.co/Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX) (163 GB; 4-bit MXFP4, keeps the DSpark MTP heads)

## Reproduce

```bash
# 1. oMLX with the DSA Metal kernel (needs Xcode + Metal Toolchain)
xcodebuild -downloadComponent MetalToolchain          # via DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
git clone https://github.com/jundot/omlx && cd omlx
python3.11 -m venv .venv && source .venv/bin/activate
OMLX_WITH_CUSTOM_KERNEL=1 pip install -e .            # builds custom_kernels/glm_moe_dsa/*.metallib

# 2. Get the 4-bit MXFP4 checkpoint (keeps MTP)
hf download Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX
ln -sfn ~/.cache/huggingface/hub/models--Vontra--DeepSeek-V4-Flash-0731-MXFP4-MLX/snapshots/*/ ~/.omlx/models/dsv4f-mxfp4

# 3. Enable DSpark (MTP is OFF by default — this is the key switch)
cp config/model_settings.json ~/.omlx/model_settings.json   # {"models":{"dsv4f-mxfp4":{"mtp_enabled":true}}}

# 4. Serve + benchmark
omlx serve --model-dir ~/.omlx/models --host 0.0.0.0 --port 11500
python3 bench/omlx_bench.py http://127.0.0.1:11500
```
Read throughput from oMLX's own server log (`~/.omlx/logs/server.log`, `Completion:` lines) — it reports accurate tok/s and the per-request accept stats. A naive client streaming counter undercounts because oMLX batches multiple tokens per SSE chunk.

## Gotchas that cost us hours
- **`mtp_enabled` defaults to `False`.** Without it you get plain base decode (~31 tok/s = the checkpoint's "spec-off" number) and think DSpark is broken. It isn't — it's just off.
- **The DSA kernel must be compiled.** Otherwise oMLX warns `glm_moe_dsa extension not built` and long-context *prefill* falls back to slow MLX (17 s cold TTFT for a 5.8K prompt vs ~6 s with the kernel). Decode is unaffected by this.
- **The brew build fails; build from source with Python 3.11** (3.14 is rejected by oMLX's version pin; the brew formula's env picks the wrong one).
- **Memory:** the 163 GB checkpoint + MTP overhead needs the Mac's full RAM. Anything else large resident (e.g. pinned Ollama models) will OOM-kill the load.

## Abliteration

Same Vontra MXFP4-MLX checkpoint, same oMLX + DSpark path. Only `attn.wo_b` on **L10–42** is projected (λ=3.5, k=1). **MTP and L0–9 stay stock.** Mac packing of the published 0731 32/32 Mida recipe ([DGX pack](https://huggingface.co/drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32)).

| | |
|--|--|
| Method | dequant MXFP8 `wo_b` → rank-1 project → **keep original e8m0 scales**, re-encode e4m3 |
| Direction | `ablit/refusal_direction_reablit_20260726.npz` (same vector as the DGX 32/32 pack) |
| Edits | **33** tensors · mean Δrel **0.0557** (min 0.0464 · max 0.0743) |
| Refusal suite | **32/32 BYPASS** twice · 0 refuse · 0 garble · 0 empty |
| Coherence | hello / Paris / `17*19=323` / `print("hello")` clean |
| DSpark | still on (count accept **100%**, tok/cycle **3.73** on a 220-tok count) |

Full per-layer Δrel + suite log: [ablit/RESULTS.md](ablit/RESULTS.md) · projector notes: [ABLIT.md](ABLIT.md).

```bash
# after the stock recipe above is on disk:
python3 scripts/project_wob_mlx.py \
  --src ~/.omlx/models/dsv4f-mxfp4 \
  --dst ~/models/dsv4f-mxfp4-ablit-mida \
  --direction ablit/refusal_direction_reablit_20260726.npz \
  --lambda-attn 3.5 --min-layer 10 --max-layer 42 --n-directions 1

# keep stock, serve ablit under the same recipe name
ln -sfn "$(readlink -f ~/.omlx/models/dsv4f-mxfp4)" ~/.omlx/models/dsv4f-mxfp4-stock
ln -sfn ~/models/dsv4f-mxfp4-ablit-mida ~/.omlx/models/dsv4f-mxfp4
ln -sfn ~/models/dsv4f-mxfp4-ablit-mida ~/.omlx/models/dsv4f-mxfp4-ablit
cp config/model_settings.json ~/.omlx/model_settings.json   # mtp_enabled + enable_thinking=false
omlx restart
```

Requests need `chat_template_kwargs.enable_thinking = false` (or the model_settings toggle above). Otherwise 0731 spends the token budget inside a hidden plan and short answers look truncated.

Flip back to stock: `ln -sfn ~/.omlx/models/dsv4f-mxfp4-stock ~/.omlx/models/dsv4f-mxfp4 && omlx restart`.

The projector **never writes through the Hugging Face blob store** — unchanged shards are hardlinked, edited shards are APFS-cloned then patched.

### Hermes first-prompt fix (the “stuck on hello” stall)

Abliterated DSV4 + a fat Hermes system prompt looks hung on the first turn. It is not a refusal hang. On this box a `hellow` sent **20,344 prompt tokens**, sat blank **59.7 s**, and a **title-generation** job hit the same Mac serve at the same time (58 s). The greeting *did* come back; the UI just showed nothing.

Same family as the earlier Mida first-turn bug (skills-catalog dump + `skill_view` on `hello`). This checkpoint did **not** spill the catalog — it just prefills ~20k with cache off and streaming off.

```yaml
# ~/.hermes/config.yaml  (fragment)
model:
  default: dsv4f-mxfp4-ablit
  provider: custom
  base_url: http://127.0.0.1:11500/v1   # or the Studio LAN IP
  max_tokens: 8192
  context_length: 1048576
  extra_body:
    temperature: 0.0
    chat_template_kwargs: { enable_thinking: false, thinking: false }
agent:
  tool_use_enforcement: false
  disabled_toolsets: [clarify]
auxiliary:
  title_generation:
    enabled: false          # do not steal the 163 GB model on turn 1
streaming:
  enabled: true             # tokens appear instead of a 60s blank
```

On the Mac (`~/.omlx/settings.json`):

```json
{ "cache": { "enabled": true, "hot_cache_max_size": "32GB" } }
```

oMLX ships `hot_cache_max_size: "0"` (disabled). A repeated ~20k Hermes prefix then re-prefills every `/new`. After `omlx restart`, `/new`, and one warm turn, the prefix should stick in the 32 GB hot cache.

Also set `enable_thinking: false` on the model (this repo’s `config/model_settings.json`). Without it, 0731 spends the first budget inside a hidden plan and short replies look truncated.

## Open item
Cold long-context TTFT is still ~13 s for a 5.8K-token prompt (the prefill of a 163 GB model). Offloading prefill to a companion DGX Spark (its native CUDA DSA kernels) would cut that — true prefill/decode disaggregation — but oMLX has no KV-injection API, so that path needs a custom decode server. Not done here.

## Credits
- Base model: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- 4-bit MXFP4 MLX quant (keeps MTP): [`Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX`](https://huggingface.co/Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX)
- Runtime + DSpark MTP + DSA kernels: [`jundot/omlx`](https://github.com/jundot/omlx)

This repo contributes the **serving recipe, 49 tok/s bench, MXFP8 `wo_b` abliteration projector, refusal-suite results, and Hermes first-prompt fix** — not the 163 GB weight files.
