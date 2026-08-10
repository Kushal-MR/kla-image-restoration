"""
STEP 3 — Work out the exact recipe KLA used to spoil the images
===============================================================

THE MODEL WE ARE TESTING
------------------------
We believe they did this, in this order:

    clean_small = shrink(GT, half size)          <- loses fine detail
    noisy       = clean_small * g  +  n          <- adds the mess

where
    g  is random "grain",   averages 1,  variance 1/L    (multiplies)
    n  is random "static",  averages 0,  variance sigma^2 (adds)

L is the grain dial. HIGH L = clean, LOW L = messy. (It means "number of
looks" -- how many times you averaged. Average 37 noisy views and the noise
mostly cancels.)

THE TRICK THAT LETS US MEASURE L AND SIGMA
------------------------------------------
Look at the error of a single pixel:

    r = noisy - clean_small = clean_small*(g - 1) + n

Take the variance of that (variance = "how much it wobbles"). Because grain
and static are independent, their wobbles add:

    var(r)  =  I^2 * (1/L)  +  sigma^2          where I = clean_small pixel

That is the equation of a STRAIGHT LINE,  y = m*x + c, with
        y = var(r)     x = I^2     m = 1/L     c = sigma^2

So: group pixels by brightness, measure how much the error wobbles in each
group, plot wobble against brightness-squared, fit a straight line.
    slope     -> L = 1/slope
    intercept -> sigma = sqrt(intercept)

That's it. That is the whole "forensics" step. It's a line fit.

WHY THIS WORKS INTUITIVELY
--------------------------
In DARK areas, I is near 0, so the I^2 term vanishes and all that's left is
the static. Dark areas measure sigma.
In BRIGHT areas the I^2 term dominates. Bright areas measure L.
One line fit reads both off at once.

Run it:  python3 step3_recipe.py --root /path/to/train --n 200
"""

import argparse, glob, json, os
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# The five ways you can shrink an image by half.
# We must try each, because we need clean_small and we don't yet know how
# they made it.
# ---------------------------------------------------------------------------
KERNELS = {
    "nearest":  Image.NEAREST,   # keep every 2nd pixel, bin the rest (jagged)
    "box":      Image.BOX,       # average each 2x2 block            (smooth)
    "bilinear": Image.BILINEAR,  # weighted average, small window    (smooth)
    "bicubic":  Image.BICUBIC,   # weighted average, wider window    (smooth)
    "lanczos":  Image.LANCZOS,   # weighted average, widest window   (smooth)
}

def shrink(img, kernel):
    """Halve the size of a float32 2D array using the given method."""
    h, w = img.shape
    # PIL mode "F" means "32-bit float, one channel". Lets us resize floats
    # without squashing them into 0-255 whole numbers and losing precision.
    pil = Image.fromarray(img.astype(np.float32), mode="F")
    out = pil.resize((w // 2, h // 2), resample=KERNELS[kernel])
    return np.asarray(out, dtype=np.float64)


def fit_pair(gt, lr, kernel, n_bins=24, edge_reject=0.75):
    """
    For ONE image pair and ONE guessed shrink method, return
    (slope, intercept, leftover_structure).
    """
    clean = shrink(gt, kernel)          # our guess at the un-noised small image
    r = lr.astype(np.float64) - clean   # the error / residual

    # --- throw away pixels sitting on sharp edges ---------------------------
    # Near an edge, different shrink methods disagree a lot, and that
    # disagreement would masquerade as "noise" and inflate our estimate.
    # np.gradient gives the rate of change in each direction.
    gy, gx = np.gradient(clean)
    grad = np.hypot(gy, gx)                       # total steepness per pixel
    keep = grad < np.quantile(grad, edge_reject)  # keep the flattest 75%

    I = clean[keep]
    R = r[keep]

    # --- bin pixels by brightness ------------------------------------------
    # Equal-count bins (quantiles) rather than equal-width, so every bin has
    # enough pixels for its variance to be trustworthy.
    edges = np.quantile(I, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(I, edges[1:-1]), 0, len(edges) - 2)

    xs, ys, ws = [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        cnt = int(sel.sum())
        if cnt < 200:            # too few pixels -> variance is unreliable
            continue
        xs.append((I[sel] ** 2).mean())   # x = average of I^2 in this bin
        ys.append(R[sel].var())           # y = how much the error wobbles
        ws.append(cnt)                    # weight = how many pixels backed it

    if len(xs) < 4:
        return None

    xs, ys, ws = np.array(xs), np.array(ys), np.array(ws, float)

    # --- weighted straight-line fit  y = slope*x + intercept ---------------
    W = ws / ws.sum()
    mx, my = (W * xs).sum(), (W * ys).sum()
    cov = (W * (xs - mx) * (ys - my)).sum()
    var = (W * (xs - mx) ** 2).sum()
    slope = cov / var
    intercept = my - slope * mx

    # --- how much STRUCTURE is left over? ----------------------------------
    # If we guessed the shrink method right, r should be pure random noise:
    # neighbouring values unrelated. If we guessed wrong, r still contains
    # bits of the real picture, and neighbouring values will be correlated.
    # We "whiten" (divide by the noise size we just predicted) then measure
    # how strongly each pixel matches the one next to it.
    pred_sd = np.sqrt(np.maximum(slope, 1e-12) * clean**2
                      + max(intercept, 1e-12))
    w2 = r / pred_sd
    w2 = w2 - w2.mean()
    a = w2[:, :-1].ravel(); b = w2[:, 1:].ravel()      # each pixel vs its right neighbour
    denom = np.sqrt((a**2).mean() * (b**2).mean()) + 1e-12
    leftover = abs((a * b).mean() / denom)             # 0 = pure noise, big = structure left

    return slope, intercept, leftover


# ---------------------------------------------------------------------------
# SELF-TEST: prove the estimator works before trusting it on real data
# ---------------------------------------------------------------------------
def self_test():
    print("SELF-TEST — make fake data with a KNOWN answer, see if we recover it")
    rng = np.random.default_rng(0)
    true_L, true_sigma = 36.0, 0.022
    errs_L, errs_s = [], []
    for _ in range(6):
        # a fake "photo": smooth blobs, min-max normalised like the real GT
        base = rng.random((32, 32))
        gt = np.asarray(Image.fromarray(base.astype(np.float32), mode="F")
                        .resize((256, 256), Image.BICUBIC), dtype=np.float32)
        gt = (gt - gt.min()) / (gt.max() - gt.min())

        clean = shrink(gt, "box")
        g = rng.gamma(shape=true_L, scale=1.0 / true_L, size=clean.shape)  # mean 1, var 1/L
        lr = clean * g + rng.normal(0, true_sigma, clean.shape)

        s, c, _ = fit_pair(gt, lr.astype(np.float32), "box")
        errs_L.append(abs(1 / s - true_L) / true_L * 100)
        errs_s.append(abs(np.sqrt(max(c, 0)) - true_sigma) / true_sigma * 100)
    print(f"  true L = {true_L}, sigma = {true_sigma}")
    print(f"  L     recovered with mean error {np.mean(errs_L):.2f} %")
    print(f"  sigma recovered with mean error {np.mean(errs_s):.2f} %")
    print("  -> if these are small, the method is sound.\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", default="degradation_config.json")
    p.add_argument("--skip-self-test", action="store_true")
    args = p.parse_args()

    if not args.skip_self_test:
        self_test()

    gt_files = sorted(glob.glob(os.path.join(args.root, "GT", "*.npy")))[: args.n]
    lr_files = sorted(glob.glob(os.path.join(args.root, "NoisyLR", "*.npy")))[: args.n]

    results = {k: {"slope": [], "intercept": [], "leftover": []} for k in KERNELS}

    print(f"Fitting {len(gt_files)} pairs against {len(KERNELS)} shrink methods...")
    for i, (gf, lf) in enumerate(zip(gt_files, lr_files)):
        gt = np.load(gf); lr = np.load(lf)
        for k in KERNELS:
            out = fit_pair(gt, lr, k)
            if out is None:
                continue
            s, c, lo = out
            results[k]["slope"].append(s)
            results[k]["intercept"].append(c)
            results[k]["leftover"].append(lo)
        if (i + 1) % 50 == 0:
            print(f"   ...{i+1}/{len(gt_files)} pairs")

    print(f"\n{'kernel':<10}{'L':>10}{'sigma':>11}{'leftover':>11}")
    print("-" * 42)
    table = {}
    for k in KERNELS:
        L = 1.0 / np.median(results[k]["slope"])
        sig = np.sqrt(max(np.median(results[k]["intercept"]), 0.0))
        lo = np.median(results[k]["leftover"])
        table[k] = (L, sig, lo)
        print(f"{k:<10}{L:>10.2f}{sig:>11.5f}{lo:>11.4f}")

    # --- decide smooth vs pixel-picking ------------------------------------
    smooth = [k for k in KERNELS if k != "nearest"]
    lo_smooth = min(table[k][2] for k in smooth)
    lo_near = table["nearest"][2]

    print()
    if lo_near > lo_smooth * 1.15:
        verdict = "smooth"
        print("Shrinking: SMOOTH (averaging). 'nearest' leaves clearly more")
        print("           structure behind, so it is ruled out.")
    else:
        verdict = "ambiguous"
        print("Shrinking: cannot separate nearest from smooth. Investigate.")

    spread = [table[k][2] for k in smooth]
    print(f"           The four smooth methods score {min(spread):.4f}-{max(spread):.4f}")
    print("           -- effectively identical. We CANNOT tell which one.")
    print("           Correct response: randomise over all four when you")
    print("           generate synthetic data. That helps on unseen images anyway.")

    # Use the best-fitting smooth kernel's numbers as the headline answer
    best = min(smooth, key=lambda k: table[k][2])
    Ls = np.array([1.0 / s for s in results[best]["slope"]])
    Ls = Ls[np.isfinite(Ls)]
    L_med = float(np.median(Ls))
    sig_med = float(table[best][1])
    lo_p, hi_p = float(np.percentile(Ls, 5)), float(np.percentile(Ls, 95))

    print(f"\nRESULT")
    print(f"  Grain strength  L     = {L_med:.2f}   (varies {lo_p:.1f} to {hi_p:.1f} between images)")
    print(f"  Extra static    sigma = {sig_med:.5f}")
    print(f"  Shrinking             = {verdict}, randomise over {smooth}")

    # Train on a DELIBERATELY WIDER range than reality, so that an unusual
    # test image is still inside what the model has seen.
    cfg = {
        "L_median": round(L_med, 3),
        "L_p5": round(lo_p, 3), "L_p95": round(hi_p, 3),
        "L_train_min": round(lo_p / 2, 2), "L_train_max": round(hi_p * 2, 2),
        "sigma_median": round(sig_med, 6),
        "sigma_train_min": 0.0, "sigma_train_max": round(sig_med * 2, 6),
        "downsample_kernels": smooth,
        "gt_is_minmax_normalised": True,
        "scale": 2,
    }
    with open(args.out, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\n  -> train with L between {cfg['L_train_min']} and {cfg['L_train_max']} (wider than measured, on purpose)")
    print(f"  -> written to {args.out}")

if __name__ == "__main__":
    main()
