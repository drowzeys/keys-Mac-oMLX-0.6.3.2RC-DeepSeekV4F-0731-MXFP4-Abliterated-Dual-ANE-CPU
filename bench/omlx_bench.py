#!/usr/bin/env python3
"""oMLX DSV4F+DSpark task-class bench via /v1/completions (raw — no chat-template issue).
count / list / read / essay. Separates TTFT (first token) from decode tok/s (rest)."""
import json, time, urllib.request, sys

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:11500"
MODEL = "dsv4f-2p4bit"
_DOC = ("In distributed inference, prefill is compute-bound while decode is memory-bandwidth-bound. "
        "The KV cache stores per-layer key and value projections for every prompt token. ") * 180

TASKS = {
    "count": ("Count from 1 to 60, one number per line:\n1\n2\n3\n", 200),
    "list":  ("List 40 distinct programming languages, numbered one per line:\n1. Python\n2. Rust\n3.", 220),
    "read":  (_DOC + "\n\nQuestion: Based only on the passage, prefill is bound by ___ and decode is bound by ___.\nAnswer:", 60),
    "essay": ("Write a detailed 350-word essay on why heterogeneous compute clusters suit disaggregated LLM inference.\n\nEssay:\n", 400),
}


def run(prompt, max_tokens):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "stream": True, "temperature": 0.0}
    req = urllib.request.Request(URL + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; n = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                ch = json.loads(p)
            except json.JSONDecodeError:
                continue
            txt = ch.get("choices", [{}])[0].get("text", "")
            if txt:
                if first is None:
                    first = time.time()
                n += 1
    end = time.time()
    ttft = (first - t0) if first else float("nan")
    dec = (n - 1) / (end - first) if (first and end > first and n > 1) else 0.0
    return ttft, dec, n


def main():
    print("[warmup]...", flush=True); run("Hello. ", 8); print("[warmup] done\n", flush=True)
    rows = []
    for name, (prompt, mt) in TASKS.items():
        best = None
        for _ in range(2):
            ttft, dec, n = run(prompt, mt)
            if best is None or dec > best[1]:
                best = (ttft, dec, n)
        rows.append((name, *best))
        print(f"{name:6} | TTFT {best[0]:.2f}s | decode {best[1]:.1f} tok/s | {best[2]} tok", flush=True)
    print("\n===== oMLX DSpark DSV4F (2.4bit) — Mac-only, raw endpoint =====")
    print(f"{'task':8} {'TTFT(s)':>9} {'decode tok/s':>13} {'tokens':>8}")
    for name, ttft, dec, n in rows:
        print(f"{name:8} {ttft:>9.2f} {dec:>13.1f} {n:>8}")
    print("BENCH-DONE")


if __name__ == "__main__":
    main()
