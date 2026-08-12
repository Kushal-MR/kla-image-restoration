#!/usr/bin/env python3
"""
inference.py — KLA benchmarking entry point.

    python inference.py <test_images_dir> <output_dir>

Reads every degraded .npy in the input directory, restores it to 2x
resolution, and writes the result to the output directory under the SAME
filename. Runs end to end with no manual edits.

INFERENCE-TIME NOTES
--------------------
Scoring measures the whole script -- interpreter start, imports, model load,
disk read, inference, disk write -- not just the forward pass. The forward
pass is roughly a quarter of the measured time, so most of the remaining
seconds are in everything around it.

  * Imports are torch and numpy only. Pulling lpips, matplotlib, scipy or
    pandas into this path would cost seconds of wall clock for nothing.
  * torch.compile is deliberately NOT used: its 30-60 s warm-up falls inside
    the measured window and is not recovered over a 400-image test set.
  * Weights load straight to the target device rather than via host RAM.
  * bf16 autocast where the GPU supports it (Ampere and later, which includes
    the H100 used for benchmarking), fp16 otherwise. Selected at runtime.
  * channels_last memory format and cudnn.benchmark: only a couple of input
    sizes occur, so autotuning pays for itself immediately.
  * Files are read and written on background threads so disk I/O overlaps GPU
    compute.
  * Images are grouped by resolution and run in batches, never one at a time.
  * No test-time augmentation: ~0.15 dB for 8x the latency is a bad trade
    under a speed metric.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from nafnet_sr import NAFNetSR  # noqa: E402

# Relative to THIS FILE, never to the caller's working directory, so the
# script runs correctly no matter where it is invoked from.
DEFAULT_WEIGHTS = os.path.join(HERE, "weights", "best.pt")


def load_npy(path):
    a = np.asarray(np.load(path), dtype=np.float32)
    # One NaN propagates through every convolution and ruins the whole output.
    # Cheap to guard, expensive to miss.
    return np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)


def _select_runtime(model, device, sample_arr):
    """Return (model, amp_dtype, channels_last) for the fastest working setup."""
    if device != "cuda":
        return model, None, False

    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    probe = torch.from_numpy(sample_arr)[None, None].to(device)

    attempts = [
        ("channels_last + amp", True, amp),
        ("contiguous + amp", False, amp),
        ("contiguous + fp32", False, None),
    ]
    last = None
    for label, cl, dt in attempts:
        try:
            m = model.to(memory_format=torch.channels_last if cl
                         else torch.contiguous_format)
            x = probe.to(memory_format=torch.channels_last if cl
                         else torch.contiguous_format)
            with torch.inference_mode():
                if dt is not None:
                    with torch.autocast("cuda", dtype=dt):
                        m(x)
                else:
                    m(x)
            torch.cuda.synchronize()
            print(f"runtime: {label}")
            return m, dt, cl
        except RuntimeError as e:
            last = e
            torch.cuda.empty_cache()

    # Last resort: cudnn's algorithm search itself can be the problem.
    torch.backends.cudnn.benchmark = False
    try:
        m = model.to(memory_format=torch.contiguous_format)
        with torch.inference_mode():
            m(probe)
        torch.cuda.synchronize()
        print("runtime: contiguous + fp32, cudnn.benchmark off")
        return m, None, False
    except RuntimeError:
        raise last


def main():
    t_start = time.time()

    ap = argparse.ArgumentParser(description="Restore degraded images.")
    ap.add_argument("input_dir", help="directory of degraded .npy images")
    ap.add_argument("output_dir", help="directory to write restored .npy images to")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Skip macOS resource-fork duplicates (._000000.npy). Archives made on a
    # Mac are full of them and loading one yields garbage.
    files = sorted(f for f in os.listdir(args.input_dir)
                   if f.endswith(".npy") and not f.startswith("._"))
    if not files:
        print(f"no .npy files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    ckpt = torch.load(args.weights, map_location=device)
    cfg = ckpt.get("cfg", {})
    model = NAFNetSR(
        img_channel=1,
        width=cfg.get("width", 32),
        middle_blk_num=cfg.get("middle_blk_num", 12),
        enc_blk_nums=tuple(cfg.get("enc_blk_nums", (2, 2, 4, 8))),
        dec_blk_nums=tuple(cfg.get("dec_blk_nums", (2, 2, 2, 2))),
        scale=2,
        use_log_channel=True,
    )
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model = model.to(device).eval()

    pool = ThreadPoolExecutor(max_workers=4)
    arrays = list(pool.map(lambda f: load_npy(os.path.join(args.input_dir, f)), files))

    # Pick the fastest configuration that ACTUALLY RUNS on this GPU.
    #
    # channels_last plus fp16 is usually the quickest combination, but cuDNN's
    # kernel coverage for DEPTHWISE convolutions in that layout is patchy, and
    # NAFNet is built from them. On a T4 it raises
    #     "FIND was unable to find an engine to execute this computation"
    # Rather than hardcode a guess and risk the benchmark machine failing the
    # same way, try the options in order of speed and keep the first that
    # survives a real forward pass. Costs one small forward; buys certainty.
    model, amp_dtype, chan_last = _select_runtime(model, device, arrays[0])

    # Group by shape so every batch is uniform and needs no padding.
    buckets = {}
    for name, arr in zip(files, arrays):
        buckets.setdefault(arr.shape, []).append((name, arr))

    writes = []
    n_done = 0
    with torch.inference_mode():
        for _shape, items in buckets.items():
            for i in range(0, len(items), args.batch):
                chunk = items[i:i + args.batch]
                x = torch.from_numpy(np.stack([a for _, a in chunk])[:, None])
                x = x.to(device, non_blocking=True)
                if chan_last:
                    x = x.to(memory_format=torch.channels_last)

                if amp_dtype is not None:
                    with torch.autocast("cuda", dtype=amp_dtype):
                        y = model(x)
                else:
                    y = model(x)

                # Every ground-truth image is individually min-max normalised
                # to exactly [0,1], so any value outside that range is
                # guaranteed error. Clamping is free accuracy.
                y = y.float().clamp_(0.0, 1.0).cpu().numpy()[:, 0]

                for (name, _), out in zip(chunk, y):
                    writes.append(pool.submit(
                        np.save, os.path.join(args.output_dir, name),
                        np.ascontiguousarray(out, dtype=np.float32)))
                    n_done += 1

    for w in writes:
        w.result()
    pool.shutdown(wait=True)

    dt = time.time() - t_start
    print(f"restored {n_done} images in {dt:.2f} s "
          f"({dt / max(n_done, 1) * 1000:.1f} ms/image, device={device})")


if __name__ == "__main__":
    main()
