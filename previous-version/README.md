# DeepSeek-V4-Flash + DSpark on a Mac Studio — Abliterated, 49 tok/s, 32/32 refusal bypass

Running **abliterated DeepSeek-V4-Flash-0731** with **DSpark speculative decoding** natively on Apple Silicon (M3 Ultra) via [oMLX](https://github.com/jundot/omlx), a compiled DSA Metal kernel, and the 4-bit MXFP4 checkpoint that keeps the MTP draft heads.

**Headline:** same 44–49 tok/s Mac Studio recipe as before, plus the Mida L10–42 `attn.wo_b` abliteration (**32/32 BYPASS**, 0 refuse, 0 garble), a **1,048,576** advertised context window, and the Hermes first-prompt fix so a `hello` does not sit blank for a minute.

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

**One-shot for agents:** [AGENTS.md](AGENTS.md). **Prebuilts (no Docker):** [PREBUILT.md](PREBUILT.md).

## Hardware / software
- **Mac:** Mac Studio M3 Ultra, 256 GB unified memory, macOS 26.4+
- **oMLX prebuilt:** Homebrew `jundot/omlx` **≥ 0.5.7** + Python **3.11** (this Studio used 0.5.7). There is **no GHCR/Docker image** — Metal is not a Linux container.
- **Abliterated weights:** [`drowzeys/keys-Mac-DeepSeek-V4-Flash-0731-MXFP4-MLX-Abliterated`](https://huggingface.co/drowzeys/keys-Mac-DeepSeek-V4-Flash-0731-MXFP4-MLX-Abliterated) (gated; 34 shards / 156 GiB; 32/32)
- **Stock quant (re-project only):** [`Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX`](https://huggingface.co/Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX)

## Reproduce (prebuilt path)

```bash
# 0. This recipe — configs + bench live HERE (do not start in the omlx source tree)
git clone https://github.com/drowzeys/keys-Mac-DeepSeek-V4-Flash-DSpark-0731-MXFP4-MLX-Abliterated-49tps
cd keys-Mac-DeepSeek-V4-Flash-DSpark-0731-MXFP4-MLX-Abliterated-49tps

# 1. Prebuilt oMLX (pin 3.11 — brew's default python can be 3.14 and oMLX rejects it)
brew install python@3.11 huggingface-cli
brew tap jundot/omlx https://github.com/jundot/omlx
brew install omlx
# optional long-ctx prefill kernel (needs full Xcode):
# brew install omlx --HEAD --with-custom-kernel

# 2. Gated HF pack (~156 GiB). Request access first if hf download 401s.
#    https://huggingface.co/drowzeys/keys-Mac-DeepSeek-V4-Flash-0731-MXFP4-MLX-Abliterated
bash scripts/setup-mac.sh
# downloads the pack, ln -sfn into ~/.omlx/models/dsv4f-mxfp4-ablit,
# copies config/model_settings.json, sets hot_cache 32GB + 1M window

# 3. Serve + bench (model id is the symlink name)
omlx serve --model-dir ~/.omlx/models --host 0.0.0.0 --port 11500
# other terminal:
python3 bench/omlx_bench.py http://127.0.0.1:11500 dsv4f-mxfp4-ablit
```

Read tok/s from oMLX's server log (`~/.omlx/logs/server.log`, `Completion:` / `Chat completion:`). A naive client streaming counter undercounts because oMLX batches multiple tokens per SSE chunk.

Source-build fallback (DSA kernel from scratch): Xcode Metal toolchain, then `OMLX_WITH_CUSTOM_KERNEL=1 pip install -e .` in a 3.11 venv of [jundot/omlx](https://github.com/jundot/omlx). See [PREBUILT.md](PREBUILT.md).

## Gotchas that cost us hours
- **`mtp_enabled` defaults to `False`.** Without it you get plain base decode (~31 tok/s) and think DSpark is broken. `scripts/setup-mac.sh` copies `config/model_settings.json` so it is on.
- **`cp config/model_settings.json` only works from *this* clone**, not from the omlx source tree. Use `setup-mac.sh` or set `RECIPE=`.
- **`bench/omlx_bench.py` defaults to `dsv4f-mxfp4-ablit`.** Pass the model id as argv2. Do not leave it on `dsv4f-2p4bit`.
- **DSA kernel is optional for decode.** Without it, long-context *prefill* falls back to slow MLX (~17 s vs ~6 s TTFT on a 5.8k prompt).
- **Python 3.14:** oMLX rejects it. `brew install python@3.11` first so the formula does not pick 3.14.
- **Memory:** 163 GB checkpoint + MTP needs the Mac’s full RAM. Unload Ollama / other pinned models first.
- **HF pack is gated.** `hf download` 401/403 → request access on the model card, then retry. Owner `drowzeys` can download immediately.

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

**Preferred:** download the published pack (step 2 above). To re-project from stock Vontra instead:

```bash
hf download Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX
ln -sfn ~/.cache/huggingface/hub/models--Vontra--DeepSeek-V4-Flash-0731-MXFP4-MLX/snapshots/*/ ~/.omlx/models/dsv4f-mxfp4
python3 scripts/project_wob_mlx.py \
  --src ~/.omlx/models/dsv4f-mxfp4 \
  --dst ~/models/dsv4f-mxfp4-ablit-mida \
  --direction ablit/refusal_direction_reablit_20260726.npz \
  --lambda-attn 3.5 --min-layer 10 --max-layer 42 --n-directions 1
ln -sfn ~/models/dsv4f-mxfp4-ablit-mida ~/.omlx/models/dsv4f-mxfp4-ablit
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

## 1M context

0731 is natively **1,048,576** tokens (YaRN, 64k × 16). Once the model is loaded, oMLX advertises `max_model_len=1048576`. The historical `sampling.max_context_window: 32768` is only a fallback when native length is unknown — pin 1M explicitly so nothing silently clamps.

```jsonc
// ~/.omlx/model_settings.json  (this repo’s config/model_settings.json)
{
  "models": {
    "dsv4f-mxfp4-ablit": {
      "mtp_enabled": true,
      "enable_thinking": false,
      "max_context_window": 1048576
    }
  }
}

// ~/.omlx/settings.json
{
  "sampling": { "max_context_window": 1048576, "max_tokens": 32768 },
  "cache": { "enabled": true, "hot_cache_max_size": "32GB" }
}
```

Hermes: `model.context_length: 1048576` (Hermes refuses anything under 64k).

**Measured on this Studio**

| prompt tokens | result |
|--------------:|--------|
| 18,085 | Hermes first turn completed (63 s, fat system prompt) |
| **41,646** | needle at the start retrieved **exactly** (`NEEDLECODE-7F3A9C21-MAC1M`, 140 s) |

A full **1M fill has not been soak-tested**. Weights are ~159 GB; Apple Metal ceiling on this box is ~223 GB (~63 GB headroom). oMLX’s KV estimate is ~5.4 MB per 64 tokens, DSA rotating window 128, paged SSD cache on (~372 GB):

| window | KV estimate | fit on 256 GB Studio |
|-------:|------------:|----------------------|
| 128k | ~11 GB | comfortable |
| 256k | ~22 GB | fine |
| 512k | ~44 GB | tight, SSD paging helps |
| **1M** | **~88 GB** | over Metal headroom unless paged SSD KV carries it |

Prefill is the other wall. 18k already took ~63 s; a cold 1M prefill is tens of minutes, and the dense DSA path can blow up before RAM does. Use the compiled `glm_moe_dsa` kernel (windowed). Offloading that prefill to a DGX Spark still needs a KV-injection API oMLX does not have.

## Open item
Cold long-context TTFT is still the cost of prefilling a 163 GB model (5.8k ~13 s in the original bench; 18k Hermes ~63 s). True prefill/decode disaggregation onto a companion DGX Spark is not done here.

## Credits
- Base model: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- 4-bit MXFP4 MLX quant (keeps MTP): [`Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX`](https://huggingface.co/Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX)
- Runtime + DSpark MTP + DSA kernels: [`jundot/omlx`](https://github.com/jundot/omlx)

Weights: [`drowzeys/keys-Mac-DeepSeek-V4-Flash-0731-MXFP4-MLX-Abliterated`](https://huggingface.co/drowzeys/keys-Mac-DeepSeek-V4-Flash-0731-MXFP4-MLX-Abliterated). This GitHub repo is the **serving recipe** (49 tok/s bench, projector, 1M knobs, refusal results, Hermes first-prompt fix).
