# Raw benchmark evidence (oMLX server-log `Completion:` lines)

## 4-bit MXFP4 (Vontra) — kernel + mtp_enabled
```
Completion: model=dsv4f-mxfp4, 200 tokens in 4.78s (48.9 tok/s), prompt: 19
Completion: model=dsv4f-mxfp4, 220 tokens in 5.43s (44.5 tok/s), prompt: 22
Completion: model=dsv4f-mxfp4, 60 tokens in 5.48s (47.5 tok/s), prompt: 5784
Completion: model=dsv4f-mxfp4, 400 tokens in 13.00s (31.9 tok/s), prompt: 22   # essay
MTP: tok/cycle=3.33 accept=40/42 (95.2%) depth[d1=15/15,d2=14/14,d3=11/13]     # read
MTP: tok/cycle=2.93 accept=146/171 (85.4%) depth[d1=62/72,d2=48/57,d3=36/42]   # list
```

## 2.4-bit (mlx-community) — kernel + mtp_enabled
```
Completion: 200 tokens (34.9 tok/s) ; 220 tokens (36.0 tok/s) ; 5784-prompt (36.4 tok/s) ; essay (30.5 tok/s)
MTP: tok/cycle=2.07 accept=31/38 (81.6%)
```

## base, no speculative: ~31-32 tok/s
## 2x DGX-Spark fp8 vLLM DSpark TP=2 (prior champion): 42.9 tok/s
