#!/usr/bin/env python3
"""Project the 0731 refusal direction onto Vontra MXFP8 attn.wo_b (MLX packing).

Mida / 32-32 recipe:
  layers L10-42, lambda=3.5, k=1, MTP stock, L0-9 stock.

W <- W - λ V^T (V W)  in the 4096-d output space, then row-magnitude restore.

Never writes through the Hugging Face blob store: unchanged shards are hardlinked
to the resolved blob; edited shards are APFS-cloned then surgically patched
(wo_b.weight bytes only) so we never deserialize the bf16 / MXFP4 experts.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import mlx.core as mx  # noqa: E402


ST_DTYPES = {
    "U8": np.uint8,
    "U16": np.uint16,
    "U32": np.uint32,
    "U64": np.uint64,
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
}


def project(W: np.ndarray, V: np.ndarray, lam: float) -> np.ndarray:
    """W [out,in], V [k,out] -> W - λ V^T (V W)."""
    if V.ndim == 1:
        V = V[None, :]
    V = V.astype(np.float32, copy=False)
    if V.shape[1] != W.shape[0] and V.shape[0] == W.shape[0]:
        V = V.T
    Q = []
    for i in range(V.shape[0]):
        v = V[i].astype(np.float32).copy()
        for q in Q:
            v = v - float(np.dot(v, q)) * q
        n = float(np.linalg.norm(v))
        Q.append(v / max(n, 1e-8))
    V = np.stack(Q, 0)
    W32 = W.astype(np.float32, copy=False)
    VW = V @ W32
    return W32 - lam * (V.T @ VW)


def _to_numpy(a, dtype=None) -> np.ndarray:
    """mlx -> numpy. Cast on the mlx side first so bf16/fp8 never hit the buffer protocol."""
    if dtype is not None:
        a = a.astype(dtype)
    mx.eval(a)
    return np.array(a)


def dequant_mxfp8(weight_u32: np.ndarray, scales_u8: np.ndarray) -> np.ndarray:
    w = mx.array(np.ascontiguousarray(weight_u32))
    s = mx.array(np.ascontiguousarray(scales_u8))
    deq = mx.dequantize(w, s, group_size=32, bits=8, mode="mxfp8")
    return _to_numpy(deq, mx.float32)


def encode_e4m3(x: np.ndarray) -> np.ndarray:
    """Vectorized OCP e4m3fn encode. NaN codes (0x7f/0xff) are never produced."""
    x = np.asarray(x, dtype=np.float32)
    sign = x < 0
    ax = np.minimum(np.abs(x), np.float32(448.0))
    out = np.zeros(x.shape, dtype=np.uint8)
    tiny = ax < np.float32(0.001953125 * 0.5)
    sub_lim = np.float32(0.015625)
    sub = (~tiny) & (ax < sub_lim)
    out[sub] = np.rint(ax[sub] * np.float32(512.0)).clip(0, 7).astype(np.uint8)
    norm = ax >= sub_lim
    mant, exp = np.frexp(ax[norm])
    frac = mant * 2.0
    E = (exp - 1) + 7
    m = np.rint((frac - 1.0) * 8.0)
    carry = m >= 8
    m = np.where(carry, 0, m)
    E = np.where(carry, E + 1, E)
    E = np.clip(E, 1, 15)
    m = np.where((E == 15) & (m >= 7), 6, m)
    m = np.clip(m, 0, 7)
    out[norm] = ((E.astype(np.int32) << 3) | m.astype(np.int32)).astype(np.uint8)
    out = np.where(sign, out | np.uint8(0x80), out)
    out[tiny] = 0
    return out.astype(np.uint8)


def pack_mxfp8_u32(e4: np.ndarray) -> np.ndarray:
    """Pack 4 little-endian e4m3 bytes per uint32 (MLX / Vontra layout)."""
    n, k = e4.shape
    if k % 4 != 0:
        raise ValueError(f"inner dim {k} not divisible by 4")
    x = e4.reshape(n, k // 4, 4).astype(np.uint32)
    return (x[:, :, 0] | (x[:, :, 1] << 8) | (x[:, :, 2] << 16) | (x[:, :, 3] << 24)).astype(np.uint32)


def requant_keep_scales(W: np.ndarray, scales_u8: np.ndarray) -> np.ndarray:
    """Re-encode W with the original e8m0 group-32 scales. Scales stay untouched."""
    scale = np.exp2(scales_u8.astype(np.float32) - np.float32(127.0))
    scale_exp = np.repeat(scale, 32, axis=1)
    e4 = encode_e4m3(W.astype(np.float32) / scale_exp)
    return pack_mxfp8_u32(e4)


def load_direction(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        z = np.load(path)
        if "directions" in z:
            d = z["directions"]
        elif "broad" in z:
            d = z["broad"]
        else:
            raise KeyError(f"no directions/broad in {path}: {list(z.files)}")
        d = np.asarray(d, dtype=np.float32)
        if d.ndim == 1:
            d = d[None, :]
        return d
    raise ValueError(f"expected .npz direction, got {path}")


def layer_id(name: str) -> int | None:
    if name.startswith("mtp."):
        return None
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "layers":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def read_st_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        return header, 8 + n


def read_st_tensor(path: Path, header: dict, data_start: int, name: str) -> np.ndarray:
    info = header[name]
    start, end = info["data_offsets"]
    dt = ST_DTYPES.get(info["dtype"])
    if dt is None:
        raise TypeError(f"{name} dtype {info['dtype']} not in ST_DTYPES")
    with open(path, "rb") as f:
        f.seek(data_start + start)
        buf = f.read(end - start)
    arr = np.frombuffer(buf, dtype=dt)
    return arr.reshape(info["shape"]).copy()


def write_st_tensor(path: Path, header: dict, data_start: int, name: str, arr: np.ndarray) -> None:
    info = header[name]
    start, end = info["data_offsets"]
    raw = np.ascontiguousarray(arr).tobytes()
    expect = end - start
    if len(raw) != expect:
        raise RuntimeError(f"{name} nbytes {len(raw)} != header {expect}")
    with open(path, "r+b") as f:
        f.seek(data_start + start)
        f.write(raw)


def clone_or_copy(src: Path, dst: Path) -> str:
    """APFS clone (copy-on-write) when possible; else full copy."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    r = subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True, text=True)
    if r.returncode == 0:
        return "clone"
    shutil.copy2(src, dst)
    return "copy"


def copy_sidecar(src: Path, dst: Path) -> None:
    for p in src.iterdir():
        if p.name.startswith("model-") and p.name.endswith(".safetensors"):
            continue
        dest = dst / p.name
        if dest.exists():
            continue
        if p.is_dir():
            shutil.copytree(p.resolve(), dest, dirs_exist_ok=True)
        else:
            shutil.copy2(p.resolve(), dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--direction", type=Path, required=True)
    ap.add_argument("--lambda-attn", type=float, default=3.5)
    ap.add_argument("--min-layer", type=int, default=10)
    ap.add_argument("--max-layer", type=int, default=42)
    ap.add_argument("--n-directions", type=int, default=1)
    ap.add_argument("--only-shard", type=str, default="", help="debug: edit this shard only")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = args.src.resolve()
    dst = args.dst
    dst.mkdir(parents=True, exist_ok=True)

    dirs = load_direction(args.direction)
    if args.n_directions > 0:
        dirs = dirs[: args.n_directions]
    print(
        f"src={src}\ndst={dst}\ndirs={dirs.shape} lambda={args.lambda_attn} "
        f"layers=[{args.min_layer},{args.max_layer}] mtp=stock",
        flush=True,
    )

    idx = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map = idx["weight_map"]

    targets: list[tuple[str, str]] = []
    for name, shard in weight_map.items():
        if not name.endswith("attn.wo_b.weight"):
            continue
        if name.startswith("mtp."):
            continue
        lid = layer_id(name)
        if lid is None:
            continue
        if args.min_layer <= lid <= args.max_layer:
            targets.append((name, shard))

    edit_names = {n for n, _ in targets}
    print(f"targets: {len(targets)} wo_b.weight tensors", flush=True)
    if args.only_shard:
        targets = [(n, s) for n, s in targets if s == args.only_shard]
        edit_names = {n for n, _ in targets}
        print(f"only-shard {args.only_shard}: {len(targets)} tensors", flush=True)

    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        by_shard[shard].append(name)

    if not args.dry_run:
        copy_sidecar(src, dst)

    stats = []
    shards_to_edit = sorted({s for _, s in targets})
    t_all = time.time()

    for shard_name in sorted(by_shard):
        src_path = (src / shard_name).resolve()
        dst_path = dst / shard_name
        needs = any(k in edit_names for k in by_shard[shard_name])
        if not needs:
            if args.only_shard:
                continue
            if args.dry_run:
                print(f"link {shard_name}", flush=True)
                continue
            if dst_path.exists() or dst_path.is_symlink():
                dst_path.unlink()
            try:
                os.link(src_path, dst_path)
            except OSError:
                # cross-device or APFS clone fallback
                shutil.copy2(src_path, dst_path)
            print(f"link {shard_name}", flush=True)
            continue

        print(f"edit {shard_name} ({src_path.stat().st_size/1e9:.2f}G) ...", flush=True)
        t0 = time.time()
        header, data_start = read_st_header(src_path)
        names_here = [n for n in edit_names if n in header]
        if args.dry_run:
            for name in names_here:
                scale_name = name.replace(".weight", ".scales")
                W = dequant_mxfp8(
                    read_st_tensor(src_path, header, data_start, name),
                    read_st_tensor(src_path, header, data_start, scale_name),
                )
                Wp = project(W, dirs, args.lambda_attn)
                row0 = np.linalg.norm(W, axis=1).clip(min=1e-8)
                row1 = np.linalg.norm(Wp, axis=1).clip(min=1e-8)
                Wp = Wp * (row0 / row1)[:, None]
                delta = float(np.linalg.norm(Wp - W) / max(float(np.linalg.norm(W)), 1e-12))
                stats.append({"tensor": name, "rel_fro": delta, "shape": list(W.shape)})
                print(f"  {name} {tuple(W.shape)} Δrel={delta:.4f}", flush=True)
            print(f"  dry-run skip write ({time.time()-t0:.1f}s)", flush=True)
            continue

        how = clone_or_copy(src_path, dst_path)
        header_d, data_start_d = read_st_header(dst_path)
        for name in names_here:
            scale_name = name.replace(".weight", ".scales")
            w0 = read_st_tensor(dst_path, header_d, data_start_d, name)
            s0 = read_st_tensor(dst_path, header_d, data_start_d, scale_name)
            W = dequant_mxfp8(w0, s0)
            assert W.shape[0] == dirs.shape[-1], f"{name} out={W.shape} V={dirs.shape}"
            Wp = project(W, dirs, args.lambda_attn)
            row0 = np.linalg.norm(W, axis=1).clip(min=1e-8)
            row1 = np.linalg.norm(Wp, axis=1).clip(min=1e-8)
            Wp = Wp * (row0 / row1)[:, None]
            delta = float(np.linalg.norm(Wp - W) / max(float(np.linalg.norm(W)), 1e-12))
            wq = requant_keep_scales(Wp, s0)
            if wq.shape != w0.shape or wq.dtype != w0.dtype:
                raise RuntimeError(
                    f"{name} packed {wq.dtype}{tuple(wq.shape)} != {w0.dtype}{tuple(w0.shape)}"
                )
            write_st_tensor(dst_path, header_d, data_start_d, name, wq)
            stats.append({"tensor": name, "rel_fro": delta, "shape": list(W.shape)})
            print(f"  {name} {tuple(W.shape)} Δrel={delta:.4f} ({how})", flush=True)
        print(f"  patched {dst_path} in {time.time()-t0:.1f}s", flush=True)

    if not args.dry_run:
        shutil.copy2(src / "model.safetensors.index.json", dst / "model.safetensors.index.json")
        meta = {
            "method": "layer-range-wo_b-projection-mxfp8-mlx-keep-scales",
            "scale_policy": "keep original e8m0 group-32 scales; re-encode e4m3 only",
            "lambda_attn": args.lambda_attn,
            "min_layer": args.min_layer,
            "max_layer": args.max_layer,
            "edit_mtp": False,
            "n_directions": int(dirs.shape[0]),
            "direction": str(args.direction),
            "n_edited": len(stats),
            "stats": stats,
            "source_stock": "Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX",
            "recipe": "champion-100pct-reablit",
            "params": f"L{args.min_layer}-{args.max_layer} λ={args.lambda_attn} k={int(dirs.shape[0])} MTP-stock",
            "note": "mHC-resistant family. Direct MXFP8 wo_b edit in MLX packing. MTP + L0-9 stock.",
        }
        (dst / "ABLIT_META.json").write_text(json.dumps(meta, indent=2))
    print(
        f"DONE edited={len(stats)} shards={len(shards_to_edit)} elapsed={time.time()-t_all:.1f}s dst={dst}",
        flush=True,
    )


if __name__ == "__main__":
    main()
