"""
STEP 5 — Prove the recipe is right by USING it
==============================================
Fitting numbers is easy. Fitting the WRONG numbers is also easy, and looks
identical. So we close the loop:

    take a real GT image
    -> apply our fitted recipe to it
    -> compare our fake NoisyLR against KLA's real NoisyLR

If the recipe is right, the two should be statistically indistinguishable:
same mean, same spread, same fraction of pixels below zero, same maximum.
They will NOT be pixel-identical -- the noise is random, we can't reproduce
their exact dice rolls. We're matching the STATISTICS, not the pixels.

This is the single check that tells you whether Phase 2 (making unlimited
synthetic training data) will actually work.

Run it:  python3 step5_verify.py --root /path/to/train --cfg degradation_config.json
"""

import argparse, glob, json, os
import numpy as np
from PIL import Image

KERNELS = {"box": Image.BOX, "bilinear": Image.BILINEAR,
           "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}


def degrade(gt, L, sigma, kernel, rng):
    """
    THE RECIPE. This function is the entire point of Phase 0.
    Feed it any clean photo and it produces a degraded one exactly the way
    KLA did. In Phase 2 you point this at DIV2K and get unlimited training data.
    """
    h, w = gt.shape
    # 1. shrink by half, smoothly
    pil = Image.fromarray(gt.astype(np.float32), mode="F")
    clean = np.asarray(pil.resize((w // 2, h // 2), KERNELS[kernel]), dtype=np.float32)

    # 2. multiply in the grain.
    #    Gamma(shape=L, scale=1/L) has mean 1 and variance 1/L -- exactly the
    #    "averaging L noisy looks" model. mean 1 is why brightness is preserved.
    g = rng.gamma(shape=L, scale=1.0 / L, size=clean.shape).astype(np.float32)

    # 3. add the static on top
    n = rng.normal(0.0, sigma, clean.shape).astype(np.float32)

    return clean * g + n          # NOTE: no clipping. Negatives are real.


p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--cfg", required=True)
p.add_argument("--n", type=int, default=300)
p.add_argument("--png", default=None)
args = p.parse_args()

cfg = json.load(open(args.cfg))
L = cfg["L_median"]; sigma = cfg["sigma_median"]
print(f"Using recipe:  L = {L},  sigma = {sigma},  kernels = {cfg['downsample_kernels']}\n")

gt_files = sorted(glob.glob(os.path.join(args.root, "GT", "*.npy")))[: args.n]
lr_files = sorted(glob.glob(os.path.join(args.root, "NoisyLR", "*.npy")))[: args.n]
rng = np.random.default_rng(0)

real, fake = [], []
for gf, lf in zip(gt_files, lr_files):
    gt = np.load(gf)
    kernel = cfg["downsample_kernels"][rng.integers(len(cfg["downsample_kernels"]))]
    real.append(np.load(lf))
    fake.append(degrade(gt, L, sigma, kernel, rng))

def stats(arrs):
    return {
        "mean":        float(np.mean([a.mean() for a in arrs])),
        "std":         float(np.mean([a.std() for a in arrs])),
        "max":         float(np.median([a.max() for a in arrs])),
        "min":         float(np.median([a.min() for a in arrs])),
        "pct_below_0": float(np.mean([(a < 0).mean() for a in arrs]) * 100),
        "img_below_0": float(np.mean([a.min() < 0 for a in arrs]) * 100),
    }

R, F = stats(real), stats(fake)
print(f"{'statistic':<26}{'KLA real':>12}{'our fake':>12}{'diff':>10}")
print("-" * 60)
for k in R:
    d = F[k] - R[k]
    print(f"{k:<26}{R[k]:>12.4f}{F[k]:>12.4f}{d:>+10.4f}")

print("\nVERDICT")
ok_spread = abs(F["std"] - R["std"]) / R["std"] < 0.05
ok_neg = abs(F["img_below_0"] - R["img_below_0"]) < 15
print(f"  noise spread matches within 5%      : {ok_spread}")
print(f"  negative-pixel behaviour matches    : {ok_neg}")
if ok_spread and ok_neg:
    print("  -> RECIPE CONFIRMED. You can now manufacture unlimited training data.")
else:
    print("  -> MISMATCH. Do not proceed to Phase 2 -- refit first.")

if args.png:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(12, 7.5))
    gt0 = np.load(gt_files[0])
    ax[0,0].imshow(gt0, cmap="gray", vmin=0, vmax=1); ax[0,0].set_title("clean GT")
    ax[0,1].imshow(real[0], cmap="gray", vmin=0, vmax=1); ax[0,1].set_title("KLA's real NoisyLR")
    ax[0,2].imshow(fake[0], cmap="gray", vmin=0, vmax=1); ax[0,2].set_title("OUR fake NoisyLR")
    for a in ax[0]: a.axis("off")
    ax[1,0].axis("off")
    ax[1,1].hist(np.concatenate([a.ravel() for a in real[:50]]), bins=120, color="tab:blue")
    ax[1,1].set_title("real pixel distribution")
    ax[1,2].hist(np.concatenate([a.ravel() for a in fake[:50]]), bins=120, color="tab:orange")
    ax[1,2].set_title("fake pixel distribution")
    plt.tight_layout(); plt.savefig(args.png, dpi=95)
    print(f"\n  picture -> {args.png}")
