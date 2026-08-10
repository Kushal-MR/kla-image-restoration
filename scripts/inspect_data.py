"""
STEP 1 — What is actually inside these files?
=============================================
Before you analyse anything, you look at it. This script answers five questions:

  1. How many files are there, and do GT and NoisyLR pair up by filename?
  2. What SHAPE is each image?  (how many rows x columns of pixels)
  3. What DTYPE is it?         (are pixels whole numbers 0-255, or decimals 0.0-1.0?)
  4. What RANGE do the values cover?  (min and max)
  5. Are there any NaN pixels? (NaN = "not a number", a poison value that
     silently destroys training if it gets into your data)

Run it:  python3 step1_inspect.py --root /path/to/train
"""

import argparse                 # lets us pass folder paths on the command line
import glob                     # finds files matching a pattern, e.g. "*.npy"
import os
import numpy as np              # the array/maths library. EVERYTHING here is numpy.

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True, help="folder containing GT/ and NoisyLR/")
parser.add_argument("--n", type=int, default=50, help="how many files to sample")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 1. FIND THE FILES
# ---------------------------------------------------------------------------
# sorted() matters: it guarantees GT[i] and LR[i] refer to the SAME sample.
# If you forget sorted(), the operating system hands you files in random order
# and you end up comparing image 47's clean version against image 912's noisy
# version. The maths still runs. The answer is garbage. This is the single most
# common silent bug in this kind of work.
gt_files = sorted(glob.glob(os.path.join(args.root, "GT", "*.npy")))
lr_files = sorted(glob.glob(os.path.join(args.root, "NoisyLR", "*.npy")))

print(f"GT files      : {len(gt_files)}")
print(f"NoisyLR files : {len(lr_files)}")

# Do the filenames actually match one-to-one?
gt_names = {os.path.basename(f) for f in gt_files}   # {...} makes a SET
lr_names = {os.path.basename(f) for f in lr_files}
print(f"Filenames match one-to-one : {gt_names == lr_names}")

# ---------------------------------------------------------------------------
# 2. LOOK INSIDE A SAMPLE OF THEM
# ---------------------------------------------------------------------------
# We don't need all 3200 to learn the format. 50 is plenty.
def describe(files, label):
    shapes, dtypes = set(), set()
    mins, maxs, nan_count = [], [], 0

    for f in files[: args.n]:
        a = np.load(f)                       # np.load reads a .npy file into an array
        shapes.add(a.shape)                  # a.shape is e.g. (256, 256) = 256 rows, 256 cols
        dtypes.add(str(a.dtype))             # a.dtype is e.g. float32
        mins.append(float(a.min()))
        maxs.append(float(a.max()))
        if not np.isfinite(a).all():         # isfinite = "not NaN and not infinity"
            nan_count += 1

    print(f"\n--- {label} ---")
    print(f"  shapes seen : {shapes}")
    print(f"  dtypes seen : {dtypes}")
    print(f"  min value   : {min(mins):.6f}   (lowest across the sample)")
    print(f"  max value   : {max(maxs):.6f}   (highest across the sample)")
    print(f"  files with NaN/Inf : {nan_count}")
    # How many images dip below zero? Only ADDITIVE noise can do that,
    # so this is direct evidence about the noise model.
    below_zero = sum(1 for m in mins if m < 0)
    print(f"  files going below 0 : {below_zero} / {len(mins)}")

describe(gt_files, "GROUND TRUTH (the clean target)")
describe(lr_files, "NOISY LR (the degraded input)")

# ---------------------------------------------------------------------------
# 3. THE MIN-MAX CHECK
# ---------------------------------------------------------------------------
# If EVERY clean image has exactly one pixel at 0.0 and exactly one at 1.0,
# that is not luck. It means they ran  (x - x.min()) / (x.max() - x.min())
# on each image. That is called per-image min-max normalisation.
print("\n--- Is GT per-image min-max normalised? ---")
exact = 0
for f in gt_files[: args.n]:
    a = np.load(f)
    if a.min() == 0.0 and a.max() == 1.0:
        exact += 1
print(f"  {exact} / {args.n} GT images have min exactly 0.0 and max exactly 1.0")
if exact == args.n:
    print("  -> YES. Two free wins:")
    print("     (a) clamp your model output to [0,1] -- anything outside is pure error")
    print("     (b) min-max normalise your synthetic crops before degrading them")
