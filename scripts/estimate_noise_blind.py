#!/usr/bin/env python3
"""
estimate_noise_blind.py — estimate speckle strength with no ground truth.

The paired estimator in day1_setup.py needs both the clean and degraded
versions of an image. The test set only ships the degraded half, so this
works from the degraded images alone: it splits each image into small
patches, and uses the fact that for multiplicative speckle a patch's variance
grows with the square of its mean. Regressing a LOW percentile of variance
against mean-squared isolates the flat patches, where the variance is noise
rather than image content.

    python scripts/estimate_noise_blind.py --dir data/Test_NoisyLR

IMPORTANT CAVEAT: this estimator reads high. Measured against synthetic data
with a known L it overstates by roughly 15-20%, because even the flattest
patches contain some real structure. Use it to COMPARE two sets (is the test
set noisier than the training set?), not as an absolute measurement. Where
ground truth exists, the paired estimator is the authoritative one.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from make_training_data import _blind_L, _patch_stats, degrade, minmax  # noqa: E402


def calibrate(rng, n=40):
    """Measure the estimator's bias on data whose L we chose ourselves."""
    from PIL import Image
    rows = []
    for true_L in (15.0, 36.82, 90.0):
        imgs = []
        for _ in range(n):
            base = rng.random((20, 20)).astype(np.float32)
            g = np.asarray(Image.fromarray(base).resize((256, 256), Image.BICUBIC))
            lr, _ = degrade(minmax(g.astype(np.float64)), L=true_L, sigma=0.0,
                            kernel="bicubic", rng=rng)
            imgs.append(lr)
        est = _blind_L(imgs)
        rows.append((true_L, est, 100 * (est - true_L) / true_L))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder of degraded .npy images")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--skip_calibration", action="store_true")
    a = ap.parse_args()

    rng = np.random.default_rng(0)

    if not a.skip_calibration:
        print("Calibration — feed it a known L and see what it reports back:")
        print(f"  {'asked for':>10}{'measured':>12}{'error':>10}")
        for t, e, err in calibrate(rng):
            print(f"  {t:>10.2f}{e:>12.1f}{err:>9.0f}%")
        print("  (consistently high, as expected -- flat patches still hold structure)\n")

    files = sorted(f for f in glob.glob(os.path.join(a.dir, "*.npy"))
                   if not os.path.basename(f).startswith("._"))[:a.n]
    if not files:
        raise SystemExit(f"no .npy files in {a.dir}")
    imgs = [np.load(f).astype(np.float64) for f in files]

    pooled = _blind_L(imgs)
    per_image = np.array([_blind_L([im]) for im in imgs], dtype=np.float64)
    per_image = per_image[np.isfinite(per_image)]

    mins = np.array([im.min() for im in imgs])
    print(f"{len(imgs)} images from {a.dir}")
    print(f"  pooled L estimate      : {pooled:.1f}")
    print(f"  per-image L, median    : {np.median(per_image):.1f}")
    print(f"  per-image L, 5-95 pct  : {np.percentile(per_image,5):.1f} "
          f"to {np.percentile(per_image,95):.1f}")
    print(f"  images going negative  : {100*(mins<0).mean():.0f}%")
    print(f"  median minimum pixel   : {np.median(mins):+.4f}")
    print("\n  The negative tail is the useful part: only additive noise can push a")
    print("  pixel below zero, so its depth measures sigma directly.")


if __name__ == "__main__":
    main()
