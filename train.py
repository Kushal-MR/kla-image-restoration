"""
PHASE 1+2 — TRAINING
====================
Trains NAFNetSR and reports SSIM / PSNR on BOTH validation sets after every
epoch. Designed to run on Kaggle (P100 or T4) but works anywhere.

THE ONE RULE: watch val_hard, not val_easy.
  val_easy = crops of photos the model trained on  -> tells you it is learning
  val_hard = photos it has NEVER seen              -> predicts the leaderboard
val_hard will be worse. If the gap grows, you are overfitting.

Typical Kaggle usage:
    python train.py \
        --photos   /kaggle/input/div2k-high-resolution-images/DIV2K_train_HR \
        --real     /kaggle/input/kla-train/train \
        --cfg      degradation_config.json \
        --split    split.json \
        --out      /kaggle/working \
        --width 32 --epochs 30 --iters-per-epoch 800 --batch 16
"""

import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import numpy as np
import torch
from torch.utils.data import DataLoader

from model.nafnet_sr import NAFNetSR, CombinedLoss, ssim, psnr
from dataset import SyntheticDataset, RealPairDataset, MixedDataset, load_split


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--photos", required=True, help="DIV2K HR folder")
    p.add_argument("--real", required=True, help="folder containing GT/ and NoisyLR/")
    p.add_argument("--cfg", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", default=".")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--gt-size", type=int, default=256, help="GT crop size (LR is half)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--iters-per-epoch", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--photo-cache", default=None,
                   help="folder for the prepared greyscale photo cache. Built "
                        "automatically on first use. Without it, training is "
                        "bound by PNG decoding rather than by the GPU.")
    p.add_argument("--real-frac", type=float, default=0.25)
    p.add_argument("--val-batches", type=int, default=24)
    p.add_argument("--resume", default=None)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, max_batches):
    """Average SSIM and PSNR over a validation set."""
    model.eval()
    S, P, n = 0.0, 0.0, 0
    for i, (lr, gt) in enumerate(loader):
        if i >= max_batches:
            break
        lr, gt = lr.to(device), gt.to(device)
        out = model(lr).clamp(0, 1)     # GT is guaranteed in [0,1], so clamping is free accuracy
        S += ssim(out, gt).item()
        P += psnr(out, gt).item()
        n += 1
    model.train()
    return (S / max(n, 1)), (P / max(n, 1))


@torch.no_grad()
def bicubic_baseline(loader, device, max_batches):
    """
    The number you MUST beat. If the trained model does not beat plain
    bicubic upscaling, something is broken in your data pipeline -- stop and
    fix that before training longer.
    """
    S, P, n = 0.0, 0.0, 0
    for i, (lr, gt) in enumerate(loader):
        if i >= max_batches:
            break
        lr, gt = lr.to(device), gt.to(device)
        up = torch.nn.functional.interpolate(lr, scale_factor=2, mode="bicubic",
                                             align_corners=False).clamp(0, 1)
        S += ssim(up, gt).item(); P += psnr(up, gt).item(); n += 1
    return (S / max(n, 1)), (P / max(n, 1))


def check_gpu_supported():
    """
    Fail early and readably if PyTorch has no kernels for this GPU.

    Kaggle still offers the Tesla P100, but recent PyTorch builds dropped
    support for its sm_60 architecture. Without this check the run dies deep
    inside a .to(device) call with "CUDA error: no kernel image is available
    for execution on the device", which reads like a bug in our code and is
    not obviously about the GPU at all.
    """
    major, minor = torch.cuda.get_device_capability(0)
    have = f"sm_{major}{minor}"
    supported = {a.split("_")[1] for a in torch.cuda.get_arch_list() if a.startswith("sm_")}
    if f"{major}{minor}" not in supported:
        name = torch.cuda.get_device_name(0)
        raise SystemExit(
            f"\n{'='*70}\n"
            f"This PyTorch build has no kernels for {name} ({have}).\n"
            f"It supports: {', '.join('sm_' + s for s in sorted(supported))}\n\n"
            f"On Kaggle: set Accelerator to 'GPU T4 x2' instead of 'GPU P100'.\n"
            f"The P100 is sm_60 and current PyTorch builds no longer ship\n"
            f"kernels for it.\n"
            f"{'='*70}"
        )


def main():
    a = build_args()
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"gpu   : {torch.cuda.get_device_name(0)}")
        check_gpu_supported()
        torch.backends.cudnn.benchmark = True   # only two input sizes, so warm-up is cheap

    torch.manual_seed(0); np.random.seed(0)

    # ---------------- data ----------------
    train_names, easy_names, hard_names = load_split(a.split)
    print(f"split: train {len(train_names)}  val_easy {len(easy_names)}  val_hard {len(hard_names)}")

    # Build the fast photo cache if asked for. Costs a few minutes once and
    # removes ~1.1 s per iteration of PNG decoding thereafter.
    photo_dir = a.photos
    if a.photo_cache:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
        from prepare_photos import prepare, find_photos
        n_src = len(find_photos(a.photos))
        n_have = len([f for f in os.listdir(a.photo_cache)
                      if f.endswith(".npy")]) if os.path.isdir(a.photo_cache) else 0
        if n_have < n_src:
            print(f"preparing photo cache ({n_have}/{n_src} done) -> {a.photo_cache}")
            prepare(a.photos, a.photo_cache)
        else:
            print(f"photo cache ready: {n_have} photos in {a.photo_cache}")
        photo_dir = a.photo_cache

    synth = SyntheticDataset(photo_dir, a.cfg, gt_size=a.gt_size,
                             length=a.iters_per_epoch * a.batch)
    real_train = RealPairDataset(a.real, names=train_names, gt_size=a.gt_size)
    train_ds = MixedDataset(synth, real_train, real_frac=a.real_frac,
                            length=a.iters_per_epoch * a.batch)
    print(f"photos found for synthesis: {len(synth.files)}")

    val_easy = RealPairDataset(a.real, names=easy_names)
    val_hard = RealPairDataset(a.real, names=hard_names)

    dl = dict(num_workers=a.workers, pin_memory=(device == "cuda"))
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True,
                          drop_last=True, persistent_workers=a.workers > 0, **dl)
    easy_dl = DataLoader(val_easy, batch_size=8, shuffle=False, **dl)
    hard_dl = DataLoader(val_hard, batch_size=8, shuffle=False, **dl)

    # ---------------- model ----------------
    model = NAFNetSR(width=a.width).to(device)
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model : NAFNet-w{a.width}, {n_par/1e6:.2f}M parameters")

    crit = CombinedLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4, betas=(0.9, 0.9))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr * 0.02)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    start_epoch = 0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        print(f"resumed from {a.resume} at epoch {start_epoch}")

    # ---------------- the number to beat ----------------
    bs_e, bp_e = bicubic_baseline(easy_dl, device, a.val_batches)
    bs_h, bp_h = bicubic_baseline(hard_dl, device, a.val_batches)
    print(f"\nBICUBIC BASELINE   easy SSIM {bs_e:.4f} PSNR {bp_e:.2f} | "
          f"hard SSIM {bs_h:.4f} PSNR {bp_h:.2f}")
    print("You must beat these. If you do not, the data pipeline is broken.\n")

    log = []
    best = -1e9
    for epoch in range(start_epoch, a.epochs):
        t0 = time.time()
        running = 0.0
        # Split the epoch's wall clock into "waiting for data" and "computing".
        # If t_data dominates, adding GPU power changes nothing -- the fix is
        # the photo cache or more workers. Without this split you cannot tell
        # a slow model from a slow data pipeline.
        t_data = t_step = 0.0
        t_mark = time.time()
        for it, (lr, gt) in enumerate(train_dl):
            t_data += time.time() - t_mark        # time spent waiting for the batch
            t_body = time.time()                  # everything after this is compute
            lr, gt = lr.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            if device == "cuda":
                lr = lr.to(memory_format=torch.channels_last)
                gt = gt.to(memory_format=torch.channels_last)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(lr)
                loss, parts = crit(out, gt)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            # Clip gradients: one freak batch can otherwise blow the model up.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            running += loss.item()

            if device == "cuda":
                torch.cuda.synchronize()      # CUDA is async; sync so timing is honest
            t_step += time.time() - t_body
            t_mark = time.time()

            if (it + 1) % 100 == 0:
                frac = t_data / max(t_data + t_step, 1e-9) * 100
                print(f"  ep{epoch} it{it+1}/{a.iters_per_epoch} "
                      f"loss {running/(it+1):.4f}  "
                      f"{(t_data+t_step)/(it+1):.3f} s/it  "
                      f"({frac:.0f}% waiting on data)", flush=True)

        sched.step()
        se, pe = evaluate(model, easy_dl, device, a.val_batches)
        sh, ph = evaluate(model, hard_dl, device, a.val_batches)
        dt = time.time() - t0

        print(f"EPOCH {epoch:3d} | loss {running/max(it+1,1):.4f} | "
              f"easy SSIM {se:.4f} PSNR {pe:.2f} | "
              f"HARD SSIM {sh:.4f} PSNR {ph:.2f} | "
              f"vs bicubic {ph-bp_h:+.2f} dB | {dt/60:.1f} min")

        log.append(dict(epoch=epoch, loss=running/max(it+1,1),
                        easy_ssim=se, easy_psnr=pe, hard_ssim=sh, hard_psnr=ph))
        json.dump(log, open(os.path.join(a.out, "training_log.json"), "w"), indent=2)

        # Select the checkpoint on val_HARD. Selecting on val_easy would pick
        # the most overfitted model.
        score = sh * 100 + ph
        ck = dict(model=model.state_dict(), opt=opt.state_dict(), epoch=epoch,
                  width=a.width, hard_ssim=sh, hard_psnr=ph)
        torch.save(ck, os.path.join(a.out, "last.pt"))
        if score > best:
            best = score
            torch.save(ck, os.path.join(a.out, "best.pt"))
            print(f"           -> new best (val_hard), saved best.pt")

    print("\ndone.")


if __name__ == "__main__":
    main()
