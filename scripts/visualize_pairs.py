"""
STEP 2 — Actually LOOK at the images
====================================
Numbers are abstract. Five minutes of looking catches problems that would
otherwise waste days -- e.g. "are the pairs even correctly matched?"

Produces one PNG with, for each sample:
   clean (256x256) | degraded (128x128) | histogram of both

A HISTOGRAM is just a bar chart of "how many pixels have each brightness".
It's the fastest way to see that the degraded image has values pushed
outside the clean image's range.

Run it:  python3 step2_look.py --root /path/to/train --out look.png
"""

import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")            # "Agg" = draw to a file, don't try to open a window.
                                 # Essential on a server / Kaggle where there is no screen.
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--out", default="look.png")
p.add_argument("--n", type=int, default=4, help="how many pairs to show")
args = p.parse_args()

gt_files = sorted(glob.glob(os.path.join(args.root, "GT", "*.npy")))
lr_files = sorted(glob.glob(os.path.join(args.root, "NoisyLR", "*.npy")))

# One row per sample, three columns: clean, degraded, histogram
fig, axes = plt.subplots(args.n, 3, figsize=(13, 3.4 * args.n))

for i in range(args.n):
    gt = np.load(gt_files[i])
    lr = np.load(lr_files[i])

    # --- column 1: the clean image ---
    # cmap="gray" = draw as greyscale. vmin/vmax fix the brightness scale so
    # both pictures are drawn on the SAME scale and are honestly comparable.
    # Without vmin/vmax, matplotlib auto-stretches each image and the noisy
    # one would deceptively look just as clean.
    axes[i, 0].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[i, 0].set_title(f"CLEAN (GT)  {gt.shape}")
    axes[i, 0].axis("off")

    # --- column 2: the degraded image ---
    axes[i, 1].imshow(lr, cmap="gray", vmin=0, vmax=1)
    axes[i, 1].set_title(f"DEGRADED (NoisyLR)  {lr.shape}")
    axes[i, 1].axis("off")

    # --- column 3: histograms of both ---
    # .ravel() flattens the 2D grid into one long 1D list of pixel values,
    # because a histogram doesn't care where a pixel was, only its value.
    axes[i, 2].hist(gt.ravel(), bins=80, alpha=0.6, label="clean", color="tab:blue")
    axes[i, 2].hist(lr.ravel(), bins=80, alpha=0.6, label="degraded", color="tab:orange")
    axes[i, 2].axvline(0, color="k", ls="--", lw=0.8)   # mark 0.0
    axes[i, 2].axvline(1, color="k", ls="--", lw=0.8)   # mark 1.0
    axes[i, 2].set_title("pixel brightness distribution")
    axes[i, 2].legend(fontsize=8)

    print(f"sample {i}:  clean range [{gt.min():.3f}, {gt.max():.3f}]   "
          f"degraded range [{lr.min():.3f}, {lr.max():.3f}]   "
          f"clean mean {gt.mean():.4f}  degraded mean {lr.mean():.4f}")

plt.tight_layout()
plt.savefig(args.out, dpi=95)
print(f"\nsaved -> {args.out}")
