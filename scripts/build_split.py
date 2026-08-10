"""
STEP 4 — Build an HONEST train / validation split
=================================================

WHY THIS IS THE MOST IMPORTANT SCRIPT IN PHASE 0
------------------------------------------------
Your 3200 samples are NOT 3200 different photos. They are crops of a few
hundred photos, in runs of consecutive indices. We saw it with our own eyes
in step 2: samples 1, 2 and 3 were the same rocky scene.

If you shuffle randomly and keep 15% aside as "validation":
    - near-identical crops of the SAME photo land on BOTH sides
    - your model trains on one crop and is "tested" on its twin
    - it is an exam where you already saw the paper
    - your score looks fantastic and predicts NOTHING

Then the real test set lands and you fall apart, with no warning.

WHAT THIS SCRIPT DOES
---------------------
  1. Measures how similar neighbouring samples are, versus random pairs,
     to PROVE that consecutive samples share photos.
  2. Cuts the dataset at a fixed index. Everything after the cut is
     validation. Because the dataset was generated one photo at a time in
     order, a contiguous block is guaranteed to be whole photos.
  3. Nudges the cut point so it doesn't slice through the middle of a group.
  4. Also makes a small "easy" validation set from inside training, so you
     can see the gap between the two.

WHY A BLOCK AND NOT CLUSTERING?
-------------------------------
You could try to detect the groups automatically and hold out whole groups.
We do measure them -- but we DON'T let the split depend on it, because that
detection is unreliable: one wrong merge can chain A-B-C together and swallow
half your dataset. (That is exactly the bug Aarush hit: it reported one group
containing 1746 of 3200 samples, which is nonsense.) A contiguous block needs
no detection to be correct.

TWO SCORES YOU WILL WATCH FOREVER
---------------------------------
  val_easy : held-out crops of photos the model DID train on
             -> tells you the model learned the basic job
  val_hard : photos the model has NEVER seen
             -> the only number that predicts the leaderboard

val_hard will be worse. That is normal and correct. If val_easy is great and
val_hard is bad, you are overfitting.

Run it:  python3 step4_split.py --root /path/to/train --out split.json
"""

import argparse, glob, json, os
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--out", default="split.json")
p.add_argument("--val_frac", type=float, default=0.15)
p.add_argument("--thumb", type=int, default=32, help="thumbnail size for comparison")
args = p.parse_args()

gt_files = sorted(glob.glob(os.path.join(args.root, "GT", "*.npy")))
N = len(gt_files)
print(f"{N} samples found\n")

# ---------------------------------------------------------------------------
# 1. MAKE A TINY THUMBNAIL OF EVERY IMAGE
# ---------------------------------------------------------------------------
# To ask "are these the same photo?" we don't need full resolution. A 32x32
# thumbnail keeps the overall layout and tones, throws away the detail, and
# makes the comparison ~64x cheaper.
print(f"[1/3] Building {args.thumb}x{args.thumb} thumbnails of all {N} images...")
T = np.zeros((N, args.thumb * args.thumb), dtype=np.float32)
for i, f in enumerate(gt_files):
    a = np.load(f).astype(np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)   # kill any NaN FIRST
    k = a.shape[0] // args.thumb
    # reshape+mean = average each block down to one pixel. Fast, no libraries.
    small = a[:k*args.thumb, :k*args.thumb].reshape(args.thumb, k, args.thumb, k).mean((1, 3))
    v = small.ravel()
    v = v - v.mean()                       # remove overall brightness...
    n = np.linalg.norm(v)
    T[i] = v / n if n > 1e-8 else 0.0      # ...and overall contrast.
    # After this, the dot product of two rows is their CORRELATION:
    #   +1 = identical layout,  0 = unrelated,  -1 = inverted.
    if (i + 1) % 1000 == 0:
        print(f"      ...{i+1}/{N}")

# ---------------------------------------------------------------------------
# 2. PROVE THAT NEIGHBOURS SHARE PHOTOS
# ---------------------------------------------------------------------------
print("\n[2/3] Do consecutive samples come from the same photo?")
neigh = np.einsum("ij,ij->i", T[:-1], T[1:])          # sample i vs sample i+1
rng = np.random.default_rng(0)
ia = rng.integers(0, N, 20000); ib = rng.integers(0, N, 20000)
ok = ia != ib
rand = np.einsum("ij,ij->i", T[ia[ok]], T[ib[ok]])    # random pairs = the noise floor

floor = np.percentile(rand, 95)     # 95% of unrelated pairs score below this
above = (neigh > floor).mean() * 100

print(f"      neighbouring pairs : median similarity {np.median(neigh):+.3f}")
print(f"      random pairs       : median similarity {np.median(rand):+.3f}")
print(f"                           95th percentile  {floor:+.3f}  <- noise floor")
print(f"      -> {above:.0f}% of neighbouring pairs beat the noise floor.")
if above > 10:
    print("      -> CONFIRMED: consecutive samples DO share source photos.")
    print("         A random split would be dishonest. Use a block split.")
else:
    print("      -> weak evidence; block split is still the safe choice.")

# ---------------------------------------------------------------------------
# 3. CUT THE DATASET
# ---------------------------------------------------------------------------
print("\n[3/3] Choosing where to cut...")
cut = int(round(N * (1 - args.val_frac)))
print(f"      target cut at index {cut}")

# Don't slice through the middle of a run. Walk forward while the sample at
# the boundary still looks like the one before it. Capped at 60 steps so a
# long accidental chain can't drag the boundary miles away.
moved = 0
while cut < N - 1 and moved < 60 and float(T[cut] @ T[cut - 1]) > floor:
    cut += 1
    moved += 1
if moved:
    print(f"      boundary landed mid-group; nudged forward {moved} to index {cut}")
else:
    print("      boundary already falls cleanly between groups")

train_idx = list(range(0, cut))
val_hard_idx = list(range(cut, N))

# val_easy: a few samples taken OUT of training, spread evenly so they come
# from many different photos rather than one run.
n_easy = max(40, int(0.03 * len(train_idx)))
easy_idx = sorted(rng.choice(train_idx, size=n_easy, replace=False).tolist())
train_idx = [i for i in train_idx if i not in set(easy_idx)]

names = [os.path.basename(f) for f in gt_files]
split = {
    "cut_index": cut,
    "train":    [names[i] for i in train_idx],
    "val_easy": [names[i] for i in easy_idx],
    "val_hard": [names[i] for i in val_hard_idx],
}
with open(args.out, "w") as f:
    json.dump(split, f, indent=2)

print(f"\n      train    : {len(train_idx):5d} samples  (indices 0 to {cut-1}, minus val_easy)")
print(f"      val_easy : {len(easy_idx):5d} samples  (taken from training -- SAME photos)")
print(f"      val_hard : {len(val_hard_idx):5d} samples  (indices {cut} to {N-1} -- UNSEEN photos)")
print(f"\n      val_hard is your real score. It will look worse than val_easy.")
print(f"      If val_easy is great and val_hard is bad, you are overfitting.")
print(f"      -> written to {args.out}")
