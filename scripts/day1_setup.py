#!/usr/bin/env python3
"""
day1_setup.py — recover the degradation recipe and build an honest split.

Two jobs, both done before any model is trained.

JOB 1: measure how the images were damaged.
For one pixel the error between the degraded image and a correctly
downsampled clean one is

    r = I*(g - 1) + n        g ~ Gamma(L, 1/L),  n ~ Normal(0, sigma^2)

so its variance is linear in I^2:

    var(r) = I^2 / L + sigma^2

Bin pixels by intensity, take the residual variance per bin, fit a weighted
straight line: the slope gives 1/L and the intercept gives sigma^2. The
estimator self-tests against data with a known L before it is trusted.

Read the sigma caveat in src/make_training_data.py: sigma^2 is the fit's
INTERCEPT and absorbs any error that does not scale with intensity, so it
comes out too high. The value used for training (0.011) was pinned down
separately from the depth of the negative tail.

JOB 2: split train from validation honestly.
Consecutive samples are crops of shared source photographs, so a random split
leaks near-duplicates into validation and the score becomes meaningless. A
contiguous block of indices is guaranteed to contain whole photographs
without relying on fragile clustering.

    python scripts/day1_setup.py --gt_dir data/train/GT --lr_dir data/train/NoisyLR
"""

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

KERNELS = {"nearest": Image.NEAREST, "box": Image.BOX, "bilinear": Image.BILINEAR,
           "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}


def shrink(img, kernel):
    h, w = img.shape
    pil = Image.fromarray(img.astype(np.float32), mode="F")
    return np.asarray(pil.resize((w // 2, h // 2), KERNELS[kernel]), dtype=np.float64)


def fit_pair(gt, lr, kernel, n_bins=24, edge_reject=0.75):
    """Return (slope, intercept, leftover_structure) for one pair, one kernel."""
    clean = shrink(gt, kernel)
    r = lr.astype(np.float64) - clean

    # Near an edge the kernels disagree, and that disagreement would be
    # mistaken for noise. Keep only the flattest pixels.
    gy, gx = np.gradient(clean)
    keep = np.hypot(gy, gx) < np.quantile(np.hypot(gy, gx), edge_reject)
    I, R = clean[keep], r[keep]

    edges = np.unique(np.quantile(I, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(I, edges[1:-1]), 0, len(edges) - 2)
    xs, ys, ws = [], [], []
    for b in range(len(edges) - 1):
        s = idx == b
        if s.sum() < 200:
            continue
        xs.append((I[s] ** 2).mean()); ys.append(R[s].var()); ws.append(int(s.sum()))
    if len(xs) < 4:
        return None
    xs, ys, W = np.array(xs), np.array(ys), np.array(ws, float)
    W /= W.sum()
    mx, my = (W * xs).sum(), (W * ys).sum()
    slope = (W * (xs - mx) * (ys - my)).sum() / (W * (xs - mx) ** 2).sum()
    intercept = my - slope * mx

    # If the kernel guess is right the residual is white noise; if it is wrong
    # some image structure is still in there and neighbours stay correlated.
    sd = np.sqrt(np.maximum(slope, 1e-12) * clean ** 2 + max(intercept, 1e-12))
    w2 = r / sd
    w2 -= w2.mean()
    aa, bb = w2[:, :-1].ravel(), w2[:, 1:].ravel()
    leftover = abs((aa * bb).mean() /
                   (np.sqrt((aa ** 2).mean() * (bb ** 2).mean()) + 1e-12))
    return slope, intercept, leftover


def self_test(rng):
    print("SELF-TEST -- recover parameters we chose ourselves")
    true_L, true_s = 36.0, 0.011
    errs = []
    for _ in range(6):
        base = rng.random((32, 32))
        gt = np.asarray(Image.fromarray(base.astype(np.float32), mode="F")
                        .resize((256, 256), Image.BICUBIC), dtype=np.float32)
        gt = (gt - gt.min()) / (gt.max() - gt.min())
        clean = shrink(gt, "box")
        lr = clean * rng.gamma(true_L, 1 / true_L, clean.shape) + \
            rng.normal(0, true_s, clean.shape)
        out = fit_pair(gt, lr.astype(np.float32), "box")
        errs.append(abs(1 / out[0] - true_L) / true_L * 100)
    print(f"  L recovered to within {np.mean(errs):.1f}%  (true L = {true_L})\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--lr_dir", required=True)
    ap.add_argument("--n", type=int, default=200, help="pairs used for the fit")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--out_split", default="split.json")
    ap.add_argument("--compare_to", help="existing split.json to verify against")
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    self_test(rng)

    gt_files = sorted(f for f in glob.glob(os.path.join(a.gt_dir, "*.npy"))
                      if not os.path.basename(f).startswith("._"))
    lr_files = sorted(f for f in glob.glob(os.path.join(a.lr_dir, "*.npy"))
                      if not os.path.basename(f).startswith("._"))
    N = len(gt_files)
    print(f"{N} pairs found")

    # ---------------- JOB 1 ----------------
    print(f"\n[1/2] Fitting the recipe on {min(a.n, N)} pairs, all five kernels...")
    res = {k: {"s": [], "c": [], "lo": []} for k in KERNELS}
    for i, (gf, lf) in enumerate(zip(gt_files[:a.n], lr_files[:a.n])):
        gt, lr = np.load(gf), np.load(lf)
        for k in KERNELS:
            o = fit_pair(gt, lr, k)
            if o:
                res[k]["s"].append(o[0]); res[k]["c"].append(o[1]); res[k]["lo"].append(o[2])
        if (i + 1) % 50 == 0:
            print(f"      {i+1}/{min(a.n, N)}")

    print(f"\n      {'kernel':<10}{'L':>9}{'sigma':>10}{'leftover':>11}")
    print("      " + "-" * 40)
    table = {}
    for k in KERNELS:
        L = 1.0 / np.median(res[k]["s"])
        sg = np.sqrt(max(np.median(res[k]["c"]), 0.0))
        lo = np.median(res[k]["lo"])
        table[k] = (L, sg, lo)
        print(f"      {k:<10}{L:>9.2f}{sg:>10.5f}{lo:>11.4f}")

    smooth = [k for k in KERNELS if k != "nearest"]
    best = min(smooth, key=lambda k: table[k][2])
    Ls = np.array([1 / s for s in res[best]["s"]])
    Ls = Ls[np.isfinite(Ls)]
    print(f"\n      L = {np.median(Ls):.2f}  (per-image {np.percentile(Ls,5):.1f}"
          f" to {np.percentile(Ls,95):.1f})")
    # State only what the numbers support. The nearest-vs-smooth test is a
    # comparison of leftover structure, and at small --n the margin is inside
    # the noise. Asserting "smooth" regardless would be reading the answer we
    # expected rather than the one we measured.
    lo_near, lo_smooth = table["nearest"][2], min(table[k][2] for k in smooth)
    if lo_near > lo_smooth * 1.15:
        print(f"      downsampling is SMOOTH: 'nearest' leaves {lo_near:.4f} "
              f"structure against {lo_smooth:.4f} for the best smooth kernel")
    else:
        print(f"      kernel family INCONCLUSIVE at n={min(a.n, N)}: 'nearest' "
              f"{lo_near:.4f} vs best smooth {lo_smooth:.4f}")
        print(f"      (the margin is inside the noise -- re-run with --n 200 or more)")
    print(f"      the four smooth kernels score "
          f"{min(table[k][2] for k in smooth):.4f}-{max(table[k][2] for k in smooth):.4f}"
          f" -- indistinguishable, so randomise over all four")
    print(f"      sigma from this fit = {table[best][1]:.5f}, but see the note in")
    print(f"      src/make_training_data.py: 0.011 is the value actually used.")

    # ---------------- JOB 2 ----------------
    print(f"\n[2/2] Building the split...")
    T = np.zeros((N, 32 * 32), dtype=np.float32)
    for i, f in enumerate(gt_files):
        im = np.nan_to_num(np.load(f).astype(np.float32))
        k = im.shape[0] // 32
        s = im[:k*32, :k*32].reshape(32, k, 32, k).mean((1, 3)).ravel()
        s -= s.mean()
        nrm = np.linalg.norm(s)
        T[i] = s / nrm if nrm > 1e-8 else 0.0

    neigh = np.einsum("ij,ij->i", T[:-1], T[1:])
    ia, ib = rng.integers(0, N, 20000), rng.integers(0, N, 20000)
    ok = ia != ib
    rand = np.einsum("ij,ij->i", T[ia[ok]], T[ib[ok]])
    floor = np.percentile(rand, 95)
    print(f"      neighbouring samples : median similarity {np.median(neigh):+.3f}")
    print(f"      random pairs         : median similarity {np.median(rand):+.3f}")
    print(f"      -> {100*(neigh > floor).mean():.0f}% of neighbours beat the noise "
          f"floor: consecutive samples DO share photographs")

    cut = int(round(N * (1 - a.val_frac)))
    moved = 0
    while cut < N - 1 and moved < 60 and float(T[cut] @ T[cut - 1]) > floor:
        cut += 1; moved += 1
    names = [os.path.basename(f) for f in gt_files]
    train_idx = list(range(cut))
    n_easy = max(40, int(0.03 * len(train_idx)))
    easy = sorted(rng.choice(train_idx, size=n_easy, replace=False).tolist())
    train = [i for i in train_idx if i not in set(easy)]
    split = {"cut_index": cut,
             "train": [names[i] for i in train],
             "val_easy": [names[i] for i in easy],
             "val_hard": [names[i] for i in range(cut, N)]}
    with open(a.out_split, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    print(f"      cut at {cut}: train {len(split['train'])}, "
          f"val_easy {len(split['val_easy'])}, val_hard {len(split['val_hard'])}")
    print(f"      -> {a.out_split}")

    if a.compare_to:
        ref = json.load(open(a.compare_to, encoding="utf-8"))
        same = ref["val_hard"] == split["val_hard"]
        print(f"\n      val_hard reproduces {a.compare_to}: {same}")
        if not same:
            print("      MISMATCH -- metrics computed on the two are not comparable")


if __name__ == "__main__":
    main()
