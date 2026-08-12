#!/usr/bin/env python3
"""
report_results.py — score the trained model and produce the submission figures.

Evaluates on the held-out val_hard split, against KLA's OWN degraded files
(never synthetic ones), and writes:

    <out>/metrics.json     per-image and summary PSNR / SSIM / LPIPS,
                           alongside the bicubic baseline
    <out>/success_{1..3}.png   largest improvements over bicubic
    <out>/failure_{1..3}.png   smallest, i.e. where the model helps least

Usage:
    python report_results.py --gt_dir data/train/GT --lr_dir data/train/NoisyLR \
           --split configs/split.json --ckpt weights/best.pt --out outputs/results

Note on averaging: PSNR is averaged PER IMAGE, which is the standard
convention. Averaging the mean-squared error across the whole set first and
converting once gives a different (lower) number, because the dB conversion is
non-linear and one bad image drags the pooled MSE down disproportionately.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from nafnet_sr import NAFNetSR  # noqa: E402


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def _gauss_window(ws=11, sigma=1.5, device="cpu", dtype=torch.float32):
    c = torch.arange(ws, device=device, dtype=dtype) - (ws - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def ssim(pred, target, ws=11, C1=0.01 ** 2, C2=0.03 ** 2):
    """Same implementation as train.py, so training and reporting agree."""
    pred, target = pred.float(), target.float()
    w = _gauss_window(ws, device=pred.device, dtype=pred.dtype)
    pad = ws // 2
    mu1 = F.conv2d(pred, w, padding=pad)
    mu2 = F.conv2d(target, w, padding=pad)
    mu1s, mu2s, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = F.conv2d(pred * pred, w, padding=pad) - mu1s
    s2 = F.conv2d(target * target, w, padding=pad) - mu2s
    s12 = F.conv2d(pred * target, w, padding=pad) - mu12
    return (((2 * mu12 + C1) * (2 * s12 + C2)) /
            ((mu1s + mu2s + C1) * (s1 + s2 + C2))).mean().item()


def psnr(pred, target):
    mse = F.mse_loss(pred.clamp(0, 1).float(), target.float()).item()
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def load_npy(p):
    a = np.asarray(np.load(p), dtype=np.float32)
    return np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--lr_dir", required=True)
    ap.add_argument("--split", default=os.path.join(HERE, "configs/split.json"))
    ap.add_argument("--ckpt", default=os.path.join(HERE, "weights/best.pt"))
    ap.add_argument("--out", default=os.path.join(HERE, "outputs/results"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no_lpips", action="store_true",
                    help="skip LPIPS (avoids downloading the AlexNet weights)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    names = json.load(open(a.split, encoding="utf-8"))["val_hard"]
    print(f"val_hard: {len(names)} samples (held-out source photographs)")

    ckpt = torch.load(a.ckpt, map_location=device)
    cfg = ckpt.get("cfg", {})
    model = NAFNetSR(img_channel=1,
                     width=cfg.get("width", 32),
                     middle_blk_num=cfg.get("middle_blk_num", 12),
                     enc_blk_nums=tuple(cfg.get("enc_blk_nums", (2, 2, 4, 8))),
                     dec_blk_nums=tuple(cfg.get("dec_blk_nums", (2, 2, 2, 2))),
                     scale=2, use_log_channel=True)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model = model.to(device).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model : NAFNet-SR {ckpt.get('size','w32')}, {n_par:,} params, "
          f"epoch {ckpt.get('epoch')}")

    lp = None
    if not a.no_lpips:
        try:
            import lpips
            lp = lpips.LPIPS(net="alex").to(device).eval()
        except Exception as e:
            print(f"LPIPS unavailable ({e}); continuing without it")

    per_image = []
    t_forward = []
    with torch.inference_mode():
        for i in range(0, len(names), a.batch):
            chunk = names[i:i + a.batch]
            gt = torch.from_numpy(np.stack([load_npy(os.path.join(a.gt_dir, n))
                                            for n in chunk]))[:, None].to(device)
            lr = torch.from_numpy(np.stack([load_npy(os.path.join(a.lr_dir, n))
                                            for n in chunk]))[:, None].to(device)

            out = model(lr).clamp(0, 1)
            # The number to beat. Same interpolation the model uses internally
            # for its residual skip, so the comparison is like for like.
            base = F.interpolate(lr, scale_factor=2, mode="bicubic",
                                 align_corners=False).clamp(0, 1)

            if lp is not None:
                # LPIPS wants 3 channels in [-1, 1]
                def to_lp(t):
                    return (t.repeat(1, 3, 1, 1) * 2 - 1)
                d_out = lp(to_lp(out), to_lp(gt)).flatten()
                d_base = lp(to_lp(base), to_lp(gt)).flatten()

            for j, n in enumerate(chunk):
                rec = dict(
                    idx=i + j, name=n,
                    psnr=psnr(out[j:j+1], gt[j:j+1]),
                    ssim=ssim(out[j:j+1], gt[j:j+1]),
                    psnr_base=psnr(base[j:j+1], gt[j:j+1]),
                    ssim_base=ssim(base[j:j+1], gt[j:j+1]),
                )
                if lp is not None:
                    rec["lpips"] = float(d_out[j])
                    rec["lpips_base"] = float(d_base[j])
                per_image.append(rec)

            if (i // a.batch) % 5 == 0:
                print(f"  {min(i+a.batch, len(names))}/{len(names)}", flush=True)

        # --- forward-pass timing, separate from the scoring loop ---------
        probe = torch.from_numpy(np.stack([load_npy(os.path.join(a.lr_dir, n))
                                           for n in names[:a.batch]]))[:, None].to(device)
        for _ in range(3):                      # warm up cudnn autotuning
            model(probe)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            model(probe)
        if device == "cuda":
            torch.cuda.synchronize()
        ms_fwd = (time.time() - t0) / 10 / probe.shape[0] * 1000

    def summ(key):
        v = np.array([r[key] for r in per_image if key in r], dtype=np.float64)
        return dict(mean=float(v.mean()), std=float(v.std())) if v.size else None

    metrics = {
        "n_val": len(per_image),
        "model": f"NAFNet-SR {ckpt.get('size','w32')}",
        "params": n_par,
        "epoch": ckpt.get("epoch"),
        "config": cfg,
        "summary": {k: summ(k) for k in
                    ("psnr", "ssim", "lpips", "psnr_base", "ssim_base", "lpips_base")
                    if summ(k) is not None},
        "batch_size": a.batch,
        "ms_per_image_forward": ms_fwd,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "timing_method": "10 forward passes after 3 warmup, cuda synchronised",
        "per_image": per_image,
    }
    with open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    s = metrics["summary"]
    print("\n" + "=" * 58)
    print(f"{'metric':<10}{'bicubic':>12}{'ours':>12}{'change':>12}")
    print("-" * 58)
    print(f"{'PSNR (dB)':<10}{s['psnr_base']['mean']:>12.2f}{s['psnr']['mean']:>12.2f}"
          f"{s['psnr']['mean']-s['psnr_base']['mean']:>+12.2f}")
    print(f"{'SSIM':<10}{s['ssim_base']['mean']:>12.4f}{s['ssim']['mean']:>12.4f}"
          f"{s['ssim']['mean']-s['ssim_base']['mean']:>+12.4f}")
    if "lpips" in s:
        print(f"{'LPIPS':<10}{s['lpips_base']['mean']:>12.4f}{s['lpips']['mean']:>12.4f}"
              f"{s['lpips']['mean']-s['lpips_base']['mean']:>+12.4f}")
    print(f"\nforward pass: {ms_fwd:.2f} ms/image (batch {a.batch}, {metrics['device']})")

    # --- figures ------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = sorted(per_image, key=lambda r: r["psnr"] - r["psnr_base"])
    picks = [("failure", ranked[:3]), ("success", ranked[-3:][::-1])]
    with torch.inference_mode():
        for label, rows in picks:
            for k, r in enumerate(rows, 1):
                n = r["name"]
                gt = load_npy(os.path.join(a.gt_dir, n))
                lr = load_npy(os.path.join(a.lr_dir, n))
                t = torch.from_numpy(lr)[None, None].to(device)
                out = model(t).clamp(0, 1)[0, 0].cpu().numpy()
                base = F.interpolate(t, scale_factor=2, mode="bicubic",
                                     align_corners=False).clamp(0, 1)[0, 0].cpu().numpy()

                fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
                for axis, im, title in zip(
                        ax, [lr, base, out, gt],
                        [f"degraded input {lr.shape}",
                         f"bicubic  {r['psnr_base']:.2f} dB",
                         f"ours  {r['psnr']:.2f} dB",
                         "ground truth"]):
                    axis.imshow(im, cmap="gray", vmin=0, vmax=1)
                    axis.set_title(title); axis.axis("off")
                fig.suptitle(f"{label} {k} — {n} — "
                             f"{r['psnr']-r['psnr_base']:+.2f} dB vs bicubic")
                plt.tight_layout()
                plt.savefig(os.path.join(a.out, f"{label}_{k}.png"), dpi=95)
                plt.close(fig)

    print(f"\nwrote metrics.json and 6 figures to {a.out}")


if __name__ == "__main__":
    main()
