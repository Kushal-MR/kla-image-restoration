#!/usr/bin/env python3
"""
inspect_npy.py — what is actually inside the supplied .npy files.

Answers the questions you need settled before writing any model code:
shape, dtype, value range, whether ground truth is min-max normalised, how
many degraded images fall outside [0,1], and whether any file contains NaN.

    python scripts/inspect_npy.py --gt_dir data/train/GT --lr_dir data/train/NoisyLR
"""

import argparse
import glob
import os

import numpy as np


def describe(files, label, n):
    shapes, dtypes = set(), set()
    mins, maxs, nan = [], [], 0
    exact01 = 0
    for f in files[:n]:
        a = np.load(f)
        shapes.add(a.shape)
        dtypes.add(str(a.dtype))
        mins.append(float(a.min()))
        maxs.append(float(a.max()))
        if not np.isfinite(a).all():
            nan += 1
        if a.min() == 0.0 and a.max() == 1.0:
            exact01 += 1

    k = len(files[:n])
    print(f"\n--- {label} ({k} sampled of {len(files)}) ---")
    print(f"  shapes            : {shapes}")
    print(f"  dtypes            : {dtypes}")
    print(f"  min / max overall : {min(mins):.6f} / {max(maxs):.6f}")
    print(f"  files with NaN    : {nan}")
    print(f"  below zero        : {sum(1 for m in mins if m < 0)} / {k}")
    print(f"  above one         : {sum(1 for m in maxs if m > 1)} / {k}")
    print(f"  exactly [0,1]     : {exact01} / {k}")
    return exact01, k, mins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--lr_dir", required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()

    gt = sorted(glob.glob(os.path.join(a.gt_dir, "*.npy")))
    lr = sorted(glob.glob(os.path.join(a.lr_dir, "*.npy")))
    gt = [f for f in gt if not os.path.basename(f).startswith("._")]
    lr = [f for f in lr if not os.path.basename(f).startswith("._")]

    print(f"GT files {len(gt)}   NoisyLR files {len(lr)}")
    print(f"filenames pair up one-to-one: "
          f"{ {os.path.basename(f) for f in gt} == {os.path.basename(f) for f in lr} }")

    e01, k, _ = describe(gt, "GROUND TRUTH (the clean target)", a.n)
    _, k2, lr_mins = describe(lr, "NOISY LR (the degraded input)", a.n)

    print("\nWhat this means")
    if e01 == k:
        print("  * Every GT image has exactly one pixel at 0 and one at 1, so they are")
        print("    individually min-max normalised. Two consequences: clamp model output")
        print("    to [0,1] (anything outside is guaranteed error), and min-max your own")
        print("    crops before degrading them so synthetic pairs match.")
    neg = sum(1 for m in lr_mins if m < 0)
    if neg:
        print(f"  * {100*neg/k2:.0f}% of degraded images contain negative pixels.")
        print("    Multiplicative speckle can never produce a negative from a")
        print("    non-negative image, so this is direct proof that additive Gaussian")
        print("    noise is present. Do not clamp the input at zero, and use a SIGNED")
        print("    log for the second channel.")


if __name__ == "__main__":
    main()
