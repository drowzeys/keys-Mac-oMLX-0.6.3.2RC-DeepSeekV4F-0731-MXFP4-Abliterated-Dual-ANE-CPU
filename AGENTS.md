# Agent one-shot — Mac DSV4F 0731 MXFP4 ablit Dual-ANE+CPU (1.0 Beta)

**Repo:** https://github.com/drowzeys/keys-Mac-oMLX-0.6.3.2RC-DeepSeekV4F-0731-MXFP4-Abliterated-Dual-ANE-CPU  
**Weights (gated):** https://huggingface.co/drowzeys/keys-Mac-DeepSeek-V4-Flash-0731-MXFP4-MLX-Abliterated  
**Engine:** oMLX **0.6.3rc2** (pack title 0.6.3.2RC)

```bash
git clone https://github.com/drowzeys/keys-Mac-oMLX-0.6.3.2RC-DeepSeekV4F-0731-MXFP4-Abliterated-Dual-ANE-CPU.git
cd keys-Mac-oMLX-0.6.3.2RC-DeepSeekV4F-0731-MXFP4-Abliterated-Dual-ANE-CPU
brew install python@3.11
brew tap jundot/omlx && brew install omlx   # 0.6.3rc2
bash scripts/setup-mac.sh
omlx serve --model-dir ~/.omlx/models --host 0.0.0.0 --port 11500
```

Keep **MXFP4**. Dual-ANE is prefill-only (~+8% uncached 10k). Decode vs ANE-off is a wash. Thinking **off**. Context 1M.

Old 49 tok/s oMLX 0.5.7 recipe: `previous-version/`.
