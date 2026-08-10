#!/usr/bin/env python3
"""
Convert a folder of photos into a fast-loading cache, once.

WHY THIS EXISTS
---------------
Training draws a random crop from a random photo, thousands of times per
epoch. DIV2K photos are ~2040x1356 RGB PNGs; decoding one costs ~144 ms.
At batch 16 over 2 workers that is ~1.15 s per iteration spent decoding,
which leaves the GPU idle roughly 80% of the time -- measured, not guessed.

Pre-converting each photo once to a greyscale uint8 .npy makes loading
~1000x cheaper. Crops are then read straight out of a memory-mapped file,
so only the bytes of the crop itself are touched.

Resolution is preserved exactly. Downscaling the source photos would be a
cheaper cache but a worse model: this is a super-resolution task, and
smoothing the training corpus biases the network toward soft content.

    python scripts/prepare_photos.py --src <div2k_dir> --out <cache_dir>
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_photos(src):
    out = []
    for root, _, files in os.walk(src):
        for f in files:
            if f.lower().endswith(EXTS) and not f.startswith("._"):
                out.append(os.path.join(root, f))
    return sorted(out)


def prepare(src, out, verbose=True):
    """Convert every photo under src into a greyscale uint8 .npy in out."""
    os.makedirs(out, exist_ok=True)
    photos = find_photos(src)
    if not photos:
        raise RuntimeError(f"no photos found under {src}")

    done, skipped = 0, 0
    t0 = time.time()
    for i, p in enumerate(photos):
        dst = os.path.join(out, f"{i:05d}.npy")
        if os.path.exists(dst):          # resume-safe: never redo work
            skipped += 1
            continue
        im = Image.open(p)
        # "L" is PIL's 8-bit greyscale. It applies the same luminance weights
        # (0.299/0.587/0.114) we use elsewhere, in C, which is much faster
        # than doing the arithmetic in numpy.
        a = np.asarray(im.convert("L"), dtype=np.uint8)
        np.save(dst, a)
        done += 1
        if verbose and (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"   {i+1}/{len(photos)}  ({el:.0f}s elapsed)", flush=True)

    if verbose:
        size = sum(os.path.getsize(os.path.join(out, f))
                   for f in os.listdir(out) if f.endswith(".npy"))
        print(f"photo cache ready: {done} converted, {skipped} already present, "
              f"{size/1e9:.2f} GB in {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    prepare(args.src, args.out)


if __name__ == "__main__":
    main()
