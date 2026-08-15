#!/bin/bash
# Idempotent Mac setup: HF ablit pack + oMLX model symlink + MTP/1M/hot-cache knobs.
# Run from a clone of this recipe repo. Does not start oMLX.
set -euo pipefail
RECIPE="$(cd "$(dirname "$0")/.." && pwd)"
HF_ID="${HF_ID:-drowzeys/keys-Mac-DeepSeek-V4-Flash-0731-MXFP4-MLX-Abliterated}"
LOCAL="${LOCAL:-$HOME/models/dsv4f-mxfp4-ablit}"
LINK="${LINK:-$HOME/.omlx/models/dsv4f-mxfp4-ablit}"

echo "==> recipe $RECIPE"
echo "==> weights $HF_ID -> $LOCAL"

if ! command -v omlx >/dev/null 2>&1; then
  echo "omlx not on PATH. Prebuilt: brew tap jundot/omlx && brew install omlx" >&2
  echo "See PREBUILT.md — there is no Docker image for Metal." >&2
  exit 1
fi
if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "need hf or huggingface-cli on PATH (brew install huggingface-cli)" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOCAL")" "$HOME/.omlx/models"
if [[ ! -f "$LOCAL/model.safetensors.index.json" ]]; then
  hf download "$HF_ID" --local-dir "$LOCAL"
else
  echo "checkpoint already at $LOCAL"
fi
ln -sfn "$LOCAL" "$LINK"
cp "$RECIPE/config/model_settings.json" "$HOME/.omlx/model_settings.json"

python3 - <<'PY'
import json
from pathlib import Path
p = Path.home() / ".omlx/settings.json"
d = json.loads(p.read_text()) if p.exists() else {}
d.setdefault("cache", {})
d["cache"]["enabled"] = True
d["cache"]["hot_cache_max_size"] = "32GB"
d.setdefault("sampling", {})
d["sampling"]["max_context_window"] = 1048576
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(d, indent=2) + "\n")
print("patched", p)
PY

echo "symlink $LINK -> $(readlink "$LINK")"
echo "next: omlx serve --model-dir ~/.omlx/models --host 0.0.0.0 --port 11500"
echo "bench: python3 $RECIPE/bench/omlx_bench.py http://127.0.0.1:11500 dsv4f-mxfp4-ablit"
