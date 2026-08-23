# Agent one-shot — Mac Studio DSV4-Flash ablit + DSpark

**Weights (gated):** https://huggingface.co/drowzeys/keys-Mac-DeepSeek-V4-Flash-0731-MXFP4-MLX-Abliterated  
**This recipe:** https://github.com/drowzeys/keys-Mac-DeepSeek-V4-Flash-DSpark-0731-MXFP4-MLX-Abliterated-49tps

Do **not** start from Vontra and re-project unless the user asks. Download the published ablit pack.

## Preconditions

- Mac Studio-class Apple Silicon, **256 GB** unified (163 GB weights + MTP will not fit in 128 GB)
- Homebrew. **No Docker/GHCR** — this is Metal, not a Linux image. See [PREBUILT.md](PREBUILT.md).
- **Python 3.11** (`brew install python@3.11` — oMLX rejects 3.14)
- Hugging Face access to the gated weight repo (`hf auth login`; owner `drowzeys` already has it)
- Unload anything large (Ollama pinned models) before load

## One-shot (prebuilt)

```bash
# 0. This recipe (configs + bench live here — clone THIS, not only omlx)
git clone https://github.com/drowzeys/keys-Mac-DeepSeek-V4-Flash-DSpark-0731-MXFP4-MLX-Abliterated-49tps
cd keys-Mac-DeepSeek-V4-Flash-DSpark-0731-MXFP4-MLX-Abliterated-49tps

# 1. Prebuilt oMLX
brew install python@3.11 huggingface-cli
brew tap jundot/omlx https://github.com/jundot/omlx
brew install omlx          # ≥ 0.5.7. optional: --HEAD --with-custom-kernel

# 2. Weights + knobs
#    https://huggingface.co/drowzeys/keys-Mac-DeepSeek-V4-Flash-0731-MXFP4-MLX-Abliterated
bash scripts/setup-mac.sh  # 401 → request gated access, retry

# 3. Serve
omlx serve --model-dir ~/.omlx/models --host 0.0.0.0 --port 11500
# other terminal:
curl -sS http://127.0.0.1:11500/v1/models
python3 bench/omlx_bench.py http://127.0.0.1:11500 dsv4f-mxfp4-ablit
```

Smoke (thinking **off**):

```bash
curl -sS http://127.0.0.1:11500/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4f-mxfp4-ablit","messages":[{"role":"user","content":"What is 17*19?"}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

Expect `323`. Read tok/s from `~/.omlx/logs/server.log` (`Completion:` / `Chat completion:`), not a client token counter.

## Hermes (optional)

```yaml
model:
  default: dsv4f-mxfp4-ablit
  provider: custom
  base_url: http://127.0.0.1:11500/v1
  max_tokens: 8192
  context_length: 1048576
  extra_body:
    temperature: 0.0
    chat_template_kwargs: { enable_thinking: false, thinking: false }
agent:
  tool_use_enforcement: false
  disabled_toolsets: [clarify]
auxiliary:
  title_generation: { enabled: false }   # first-prompt stall if left on
streaming: { enabled: true }
```

Then `/new`. First turn still prefills ~20k tokens once; hot cache should make the next turn cheap.

## Do not

- Leave `mtp_enabled` unset (you get ~31 tok/s and think DSpark is broken).
- Point the bench at `dsv4f-2p4bit` unless you meant the 2.4-bit comparison.
- Re-project from Vontra unless asked — the HF pack **is** the ablit.
- Load a second 100 GB+ model on the same oMLX LRU (it can evict this one).
- Raise vLLM `--gpu-memory-utilization` on GB10 boxes above 0.85 (unrelated to this Mac recipe; fleet cap).

## Links

| | |
|--|--|
| Ablit weights | https://huggingface.co/drowzeys/keys-Mac-DeepSeek-V4-Flash-0731-MXFP4-MLX-Abliterated |
| This recipe | https://github.com/drowzeys/keys-Mac-DeepSeek-V4-Flash-DSpark-0731-MXFP4-MLX-Abliterated-49tps |
| Sister DGX/FP8 32/32 | https://huggingface.co/drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32 |
| Stock MXFP4-MLX | https://huggingface.co/Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX |
| oMLX | https://github.com/jundot/omlx |
