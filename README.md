# DeepSeek-V4-Flash + DSpark on a Mac Studio — 49 tok/s, beating a 2× DGX-Spark vLLM cluster

Running **DeepSeek-V4-Flash-0731** with **DSpark speculative decoding** natively on Apple Silicon (M3 Ultra) via [oMLX](https://github.com/jundot/omlx), a compiled DSA Metal kernel, and the 4-bit MXFP4 checkpoint that keeps the MTP draft heads.

**Headline result:** on structured / agentic workloads the Mac Studio *alone* decodes DSV4-Flash **faster than a 2-node DGX-Spark vLLM cluster** — single machine, speculative decode on, lossless.

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

## Open item
Cold long-context TTFT is still ~13 s for a 5.8K-token prompt (the prefill of a 163 GB model). Offloading prefill to a companion DGX Spark (its native CUDA DSA kernels) would cut that — true prefill/decode disaggregation — but oMLX has no KV-injection API, so that path needs a custom decode server. Not done here.

## Credits
- Base model: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- 4-bit MXFP4 MLX quant (keeps MTP): [`Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX`](https://huggingface.co/Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX)
- Runtime + DSpark MTP + DSA kernels: [`jundot/omlx`](https://github.com/jundot/omlx)

This repo contributes the **serving recipe + benchmark methodology + the counter-intuitive 4-bit-beats-2.4-bit finding**, not the model weights.
