#!/usr/bin/env python3
"""
evaluate.py — KLA benchmarking entry point.

    python evaluate.py <test_images_dir> <output_dir>

Reads every degraded image in the input directory, restores it to 2x
resolution, and writes the result to the output directory using the SAME
filename. Runs end to end with no manual edits.

INFERENCE-TIME NOTES
--------------------
KLA measures the WHOLE script: interpreter start, imports, model load, disk
read, inference, disk write. Every decision below is made with that in mind.

  * Imports are minimal -- torch and numpy only. Importing lpips, matplotlib,
    scipy or pandas here would cost seconds of wall clock for no benefit.
  * torch.compile is deliberately NOT used: its 30-60 s warm-up sits inside
    the measured window and cannot be amortised over a single test set.
  * Weights load straight to the GPU (map_location) instead of via host RAM.
  * bf16 autocast on Ampere/Hopper; the H100 is very fast at bf16 and it is
    more forgiving of overflow than fp16.
  * channels_last memory format, cudnn.benchmark on (only a couple of input
    sizes occur, so autotune pays for itself immediately).
  * Images are grouped by size and run in batches rather than one at a time.
  * PNG output uses compression level 1: writing is timed and the size
    penalty is small.
  * No test-time augmentation. Eight flipped passes buy ~0.15 dB for 8x the
    time, which is a bad trade under a speed metric.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from model.nafnet_sr import NAFNetSR  # noqa: E402

# Model path is relative to THIS FILE, never to the caller's working
# directory, so the script runs correctly from anywhere.
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(HERE, "weights", "best.pt")

IMAGE_EXTS = (".npy", ".png", ".tif", ".tiff", ".jpg", ".jpeg")


def load_image(path):
    """Return (array float32 2D, kind) where kind records how to write it back."""
    if path.lower().endswith(".npy"):
        a = np.load(path)
        return np.asarray(a, dtype=np.float32), "npy"
    from PIL import Image
    im = Image.open(path)
    a = np.asarray(im)
    scale = 65535.0 if a.dtype == np.uint16 else 255.0
    if a.ndim == 3:
        a = a[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (np.asarray(a, dtype=np.float32) / scale), ("png16" if im.mode == "I;16" else "png8")


def save_image(path, arr, kind):
    if kind == "npy":
        np.save(path, arr.astype(np.float32))
        return
    from PIL import Image
    if kind == "png16":
        Image.fromarray((arr * 65535).round().astype(np.uint16), mode="I;16").save(
            path, compress_level=1)
    else:
        Image.fromarray((arr * 255).round().astype(np.uint8)).save(path, compress_level=1)


def main():
    t_start = time.time()

    ap = argparse.ArgumentParser(description="Restore degraded images.")
    ap.add_argument("input_dir", help="directory of degraded test images")
    ap.add_argument("output_dir", help="directory to write restored images to")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Ignore macOS resource-fork duplicates (._000000.npy) -- loading those
    # yields garbage, and zips made on a Mac are full of them.
    files = sorted(
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith(IMAGE_EXTS) and not f.startswith("._")
    )
    if not files:
        print(f"no images found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    ckpt = torch.load(args.weights, map_location=device)
    width = ckpt.get("width", 32)
    model = NAFNetSR(width=width)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model = model.to(device).eval()
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

    # Group by shape so every batch is uniform -- lets us batch without padding.
    buckets = {}
    kinds = {}
    for f in files:
        arr, kind = load_image(os.path.join(args.input_dir, f))
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        buckets.setdefault(arr.shape, []).append((f, arr))
        kinds[f] = kind

    n_done = 0
    with torch.inference_mode():
        for shape, items in buckets.items():
            for i in range(0, len(items), args.batch):
                chunk = items[i:i + args.batch]
                x = torch.from_numpy(
                    np.stack([a for _, a in chunk])[:, None]
                ).to(device, non_blocking=True).float()
                if device == "cuda":
                    x = x.to(memory_format=torch.channels_last)

                if use_bf16:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        y = model(x)
                else:
                    y = model(x)

                # Ground truth is per-image min-max normalised to exactly
                # [0, 1], so anything outside that range is guaranteed error.
                y = y.float().clamp_(0.0, 1.0).cpu().numpy()[:, 0]

                for (name, _), out in zip(chunk, y):
                    save_image(os.path.join(args.output_dir, name), out, kinds[name])
                    n_done += 1

    dt = time.time() - t_start
    print(f"restored {n_done} images in {dt:.2f} s "
          f"({dt / max(n_done, 1) * 1000:.1f} ms/image, device={device})")


if __name__ == "__main__":
    main()
