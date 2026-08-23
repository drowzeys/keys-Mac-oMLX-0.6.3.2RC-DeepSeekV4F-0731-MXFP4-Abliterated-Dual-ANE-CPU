# Credits

**Big thanks.** Native DSV4F + DSpark on a Mac Studio is possible because of the
projects below. Star and cite them. This pack is the 0731 MXFP4 ablit + Dual-ANE+CPU
recipe and benches on top.

## Engine

- **[jundot/omlx](https://github.com/jundot/omlx)** (Apache-2.0) — oMLX serve, MTP/DSpark
  on Apple Silicon, Dual-ANE prefill (`deepseek_ane_prefill_*` in 0.6.3rc2).
- **[ml-explore/MLX](https://github.com/ml-explore/mlx)** — Metal arrays / mlx-lm.
- **Apple** — Neural Engine. The Dual-ANE path uses undocumented APIs and can break
  on a macOS update.

## Model and DSpark

- **[DeepSeek-AI](https://www.deepseek.com/)** — DeepSeek-V4-Flash **0731** and DSpark /
  DeepSpec speculative decoding.
- **[Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX](https://huggingface.co/Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX)**
  — 4-bit MXFP4 MLX quant this ablit pack starts from.
- **[Fraser Price](https://huggingface.co/fraserprice/DeepSeek-V4-Flash-DSpark)** —
  DSpark runtime lineage used across Spark and Mac recipes.

## Two-Spark cousin (same 0731 family)

- **[Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10)** — GB10 vLLM + DSpark image.
- **[MiaAI-Lab 2× Spark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)**
- **[Tony / tonyd2wild](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark)**
  — 1M NVFP4-KV dual-Spark recipe we compare against.

## What this pack adds

L10–35 `wo_b` MXFP4 ablit packing, Dual-ANE+CPU prefill knobs, 1M window, Hermes
first-prompt notes, and the 2026-08-23 real-serve benches. Dual-ANE on DSV4 is
**prefill-only** (~+8% uncached 10k). Stay on MXFP4; Aday777 NVFP4 Dual-ANE no-ops here.
