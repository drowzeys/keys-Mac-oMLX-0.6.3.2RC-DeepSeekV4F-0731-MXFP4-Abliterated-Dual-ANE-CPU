# Prebuilts (no Docker)

This is an **Apple Silicon / Metal** recipe. There is **no GHCR/Docker image** — Metal does not run in a Linux container. Use these prebuilts instead.

| What | Prebuilt | Why |
|------|----------|-----|
| **Ablit weights** | [`drowzeys/keys-Mac-DeepSeek-V4-Flash-0731-MXFP4-MLX-Abliterated`](https://huggingface.co/drowzeys/keys-Mac-DeepSeek-V4-Flash-0731-MXFP4-MLX-Abliterated) | 156 GiB MXFP4-MLX pack, 34 shards, gated. Do not re-project Vontra unless asked. |
| **oMLX runtime** | Homebrew `jundot/omlx` **≥ 0.5.7** | This Studio served the 32/32 pack on **0.5.7**. Pin `python@3.11` so brew does not pick 3.14. |
| **DSA Metal kernel** (long-ctx prefill) | `brew install omlx --HEAD --with-custom-kernel` *or* source `OMLX_WITH_CUSTOM_KERNEL=1` | Needs full Xcode + Metal toolchain. Decode works without it; 5.8k TTFT is ~17 s vs ~6 s. |
| **Python** | `brew install python@3.11` | oMLX’s pin rejects 3.14. |
| **HF CLI** | `brew install huggingface-cli` or `pipx install huggingface_hub` | For `hf download`. |

## Install oMLX (preferred)

```bash
brew install python@3.11
brew tap jundot/omlx https://github.com/jundot/omlx
# default bottle is enough for DSpark MTP decode
brew install omlx
# optional: windowed DSA kernel for long-context prefill
# brew install omlx --HEAD --with-custom-kernel
omlx --version    # expect 0.5.7+
```

Then from a clone of this recipe:

```bash
bash scripts/setup-mac.sh
omlx serve --model-dir ~/.omlx/models --host 0.0.0.0 --port 11500
```

## Not provided

- No `ghcr.io/...` image. If you need a Linux/DGX serve of the *same* ablit, use the sister FP8 pack [`drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32`](https://huggingface.co/drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32) with the Anemll/Mia dual-Spark recipe — that is a different runtime.
- No prebuilt `.metallib` in this repo (it is machine/Xcode-specific).
