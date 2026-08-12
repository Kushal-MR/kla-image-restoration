"""
Trains NAFNet-SR for the KLA image restoration challenge.

What it trains on
-----------------
Three sources, mixed together every epoch:

  1. KLA's real pairs, used exactly as given. Keeps you anchored to the truth.
  2. KLA's clean images, re-damaged fresh each time with random settings.
  3. Outside photos (DIV2K etc.), also damaged fresh each time.

Sources 2 and 3 are what make the model cope with images it has never seen,
which is half the score. Source 1 stops it drifting away from the real problem.

How it is scored while training
-------------------------------
Validation always uses KLA's REAL damaged files, never synthetic ones, on the
photos held out by day1_setup.py. That number is the honest one.

Run it
------
    python train.py --smoke_test        # ~1 minute, checks everything works

    python train.py --gt_dir GT --lr_dir NoisyLR --split split.json \
                    --extra_dir DIV2K_train_HR --size w32 --epochs 60
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# The notebook this was developed in copied every module into one working
# directory, so these were plain top-level imports. In the repository they
# live in src/, so put that on the path first -- relative to THIS file, not
# to the caller's working directory.
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from make_training_data import degrade, minmax
from nafnet_sr import NAFNetSR, get_device

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------

def load_gray(path):
    """Load a .npy array or an ordinary image file as greyscale floats."""
    if path.lower().endswith('.npy'):
        a = np.load(path).astype(np.float64)
    else:
        from PIL import Image
        a = np.asarray(Image.open(path).convert('L')).astype(np.float64) / 255.0
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


class PairDataset(Dataset):
    """mode='real'      -> KLA's own damaged files, untouched
       mode='synthetic' -> clean image damaged fresh on every access
    """

    def __init__(self, gt_paths, lr_paths=None, mode='synthetic',
                 crop=256, augment=True, seed=None):
        self.gt = list(gt_paths)
        self.lr = list(lr_paths) if lr_paths else None
        self.mode = mode
        self.crop = crop
        self.augment = augment
        self.seed = seed
        if mode == 'real' and not self.lr:
            raise ValueError("mode='real' needs lr_paths")

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, i):
        # A fixed seed makes validation identical every epoch, so the numbers are
        # comparable run to run. Training leaves it None so the noise is fresh.
        rng = (np.random.default_rng(self.seed + i) if self.seed is not None
               else np.random.default_rng())

        gt = load_gray(self.gt[i])

        if self.mode == 'real':
            lr = load_gray(self.lr[i])
        else:
            if self.crop and min(gt.shape) > self.crop:
                y = int(rng.integers(0, gt.shape[0] - self.crop + 1))
                x = int(rng.integers(0, gt.shape[1] - self.crop + 1))
                gt = gt[y:y + self.crop, x:x + self.crop]
            # Every GT image KLA supplied is stretched to exactly [0,1].
            # Matching that keeps synthetic pairs statistically identical.
            gt = minmax(gt)
            lr, _ = degrade(gt, rng=rng, blur=bool(rng.random() < 0.1))

        if self.augment:
            if rng.random() < 0.5:
                gt, lr = gt[:, ::-1], lr[:, ::-1]
            if rng.random() < 0.5:
                gt, lr = gt[::-1], lr[::-1]
            k = int(rng.integers(4))
            if k:
                gt, lr = np.rot90(gt, k), np.rot90(lr, k)

        gt = torch.from_numpy(np.ascontiguousarray(gt, dtype=np.float32))[None]
        lr = torch.from_numpy(np.ascontiguousarray(lr, dtype=np.float32))[None]
        return lr, gt


# ----------------------------------------------------------------------
# losses and metrics
# ----------------------------------------------------------------------

class Charbonnier(torch.nn.Module):
    """Smoothed absolute error. Steadier than squared error, sharper than plain L1."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.e2 = eps ** 2

    def forward(self, p, t):
        return torch.sqrt((p - t) ** 2 + self.e2).mean()


def fft_loss(pred, target):
    """Compares the two images by their frequency content.

    This is what stops the model cheating. Blurring is an easy way to make grain
    vanish, and pixel losses barely punish it. Blurring destroys high frequencies,
    which this term notices immediately.
    """
    pf = torch.fft.rfft2(pred.float(), norm='ortho')
    tf = torch.fft.rfft2(target.float(), norm='ortho')
    return (torch.view_as_real(pf) - torch.view_as_real(tf)).abs().mean()


def _gauss_window(ws=11, sigma=1.5, device='cpu', dtype=torch.float32):
    c = torch.arange(ws, device=device, dtype=dtype) - (ws - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def ssim(pred, target, ws=11, C1=0.01 ** 2, C2=0.03 ** 2):
    """Structural similarity, one of the three scores KLA uses. 1.0 is perfect."""
    pred, target = pred.float(), target.float()
    w = _gauss_window(ws, device=pred.device, dtype=pred.dtype)
    pad = ws // 2
    mu1 = F.conv2d(pred, w, padding=pad)
    mu2 = F.conv2d(target, w, padding=pad)
    mu1s, mu2s, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = F.conv2d(pred * pred, w, padding=pad) - mu1s
    s2 = F.conv2d(target * target, w, padding=pad) - mu2s
    s12 = F.conv2d(pred * target, w, padding=pad) - mu12
    num = (2 * mu12 + C1) * (2 * s12 + C2)
    den = (mu1s + mu2s + C1) * (s1 + s2 + C2)
    return (num / den).mean()


def psnr(pred, target):
    """Higher is better; +1 dB is a solid gain.

    Clamped to [0,1] first because every KLA ground-truth image lives in exactly
    that range, so anything outside it is pure error.
    """
    mse = F.mse_loss(pred.clamp(0, 1).float(), target.float())
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-12))


class TotalLoss(torch.nn.Module):
    def __init__(self, w_fft=0.05, w_ssim=0.10):
        super().__init__()
        self.char = Charbonnier()
        self.w_fft = w_fft
        self.w_ssim = w_ssim

    def forward(self, pred, target):
        loss = self.char(pred, target)
        if self.w_fft:
            loss = loss + self.w_fft * fft_loss(pred, target)
        if self.w_ssim:
            loss = loss + self.w_ssim * (1 - ssim(pred.clamp(0, 1), target))
        return loss


# ----------------------------------------------------------------------
# train / validate
# ----------------------------------------------------------------------

class EMA:
    """Keeps a slow-moving average of the weights, and scores THAT.

    Individual training steps jitter the weights around the good solution. The
    average sits much closer to the middle of it, so it scores better and, more
    importantly, stops the epoch-to-epoch bouncing that made the last run's
    validation swing by 0.5 dB and then fall apart near the end.

    Standard practice in image restoration; typically worth 0.1-0.3 dB on its own.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                     alpha=1 - self.decay)

    def apply_to(self, model):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if k in self.shadow}
        model.load_state_dict({k: v.to(dtype=model.state_dict()[k].dtype)
                               for k, v in self.shadow.items()}, strict=False)

    def restore(self, model):
        if self.backup:
            model.load_state_dict(self.backup, strict=False)
            self.backup = {}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    ps, ss, n = 0.0, 0.0, 0
    for lr, gt in loader:
        lr, gt = lr.to(device), gt.to(device)
        out = model(lr).clamp(0, 1)
        b = lr.shape[0]
        ps += float(psnr(out, gt)) * b
        ss += float(ssim(out, gt)) * b
        n += b
    model.train()
    return ps / max(n, 1), ss / max(n, 1)


@torch.no_grad()
def baseline_scores(loader, device, scale=2):
    """What plain bicubic upscaling gets. The model must beat this, or something
    is broken. Cheapest possible sanity check."""
    ps, ss, n = 0.0, 0.0, 0
    for lr, gt in loader:
        lr, gt = lr.to(device), gt.to(device)
        up = F.interpolate(lr, scale_factor=scale, mode='bicubic',
                           align_corners=False).clamp(0, 1)
        b = lr.shape[0]
        ps += float(psnr(up, gt)) * b
        ss += float(ssim(up, gt)) * b
        n += b
    return ps / max(n, 1), ss / max(n, 1)


def build_loaders(a):
    gt_files = sorted(glob.glob(os.path.join(a.gt_dir, '*.npy')))
    lr_files = [os.path.join(a.lr_dir, os.path.basename(f)) for f in gt_files]
    keep = [i for i, f in enumerate(lr_files) if os.path.exists(f)]
    gt_files = [gt_files[i] for i in keep]
    lr_files = [lr_files[i] for i in keep]
    if not gt_files:
        raise SystemExit('No matching GT / NoisyLR pairs found. Check the paths.')

    split = json.load(open(a.split))
    tr_idx = [i for i in split['train'] if i < len(gt_files)]
    va_idx = [i for i in split['val_hard'] if i < len(gt_files)]
    print(f'  KLA pairs: {len(gt_files)}  ->  train {len(tr_idx)}, val {len(va_idx)}')

    sets = [
        # 1. the real pairs, untouched
        PairDataset([gt_files[i] for i in tr_idx], [lr_files[i] for i in tr_idx],
                    mode='real', crop=None),
        # 2. the same clean images, re-damaged fresh every time
        PairDataset([gt_files[i] for i in tr_idx], mode='synthetic', crop=a.crop),
    ]

    if a.extra_dir:
        extra = []
        for e in IMG_EXT + ('.npy',):
            extra += glob.glob(os.path.join(a.extra_dir, '**', '*' + e), recursive=True)
        extra = sorted(set(extra))
        if extra:
            reps = max(1, a.extra_repeat)
            sets.append(PairDataset(extra * reps, mode='synthetic', crop=a.crop))
            print(f'  outside photos: {len(extra)} x{reps} = {len(extra) * reps}')
        else:
            print(f'  WARNING: no images found under {a.extra_dir}')

    train_ds = ConcatDataset(sets)
    # Validation uses KLA's REAL damaged files -- the honest measure.
    val_ds = PairDataset([gt_files[i] for i in va_idx], [lr_files[i] for i in va_idx],
                         mode='real', crop=None, augment=False, seed=1234)

    pin = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True,
                          num_workers=a.workers, pin_memory=pin, drop_last=True,
                          persistent_workers=a.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=max(1, a.batch // 2), shuffle=False,
                        num_workers=a.workers, pin_memory=pin)
    print(f'  training samples per epoch: {len(train_ds)}')
    return train_dl, val_dl


CFGS = {
    'w16': dict(width=16, middle_blk_num=4, enc_blk_nums=(1, 1, 2, 4),
                dec_blk_nums=(1, 1, 1, 1)),
    'w32': dict(width=32, middle_blk_num=12, enc_blk_nums=(2, 2, 4, 8),
                dec_blk_nums=(2, 2, 2, 2)),
    'w64': dict(width=64, middle_blk_num=12, enc_blk_nums=(2, 2, 4, 8),
                dec_blk_nums=(2, 2, 2, 2)),
}


def run(a):
    device = get_device()
    print(f'device: {device}')
    torch.manual_seed(0)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    train_dl, val_dl = build_loaders(a)

    cfg = CFGS[a.size]
    model = NAFNetSR(img_channel=1, scale=2, use_log_channel=True, **cfg).to(device)
    print(f'model: NAFNet-SR {a.size}, '
          f'{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters')

    crit = TotalLoss(a.w_fft, a.w_ssim)
    # beta2=0.999, not NAFNet's 0.9. Their 0.9 makes Adam's variance estimate very
    # noisy; it works for them with long warmups and huge batches, but here it is
    # the most likely cause of the last run's unstable, drifting late epochs.
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4,
                            betas=(0.9, a.beta2))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs,
                                                       eta_min=a.lr * 0.01)
    ema = EMA(model, decay=a.ema) if a.ema > 0 else None
    warmup_steps = a.warmup
    step = 0
    # Pick the right half-precision type for the actual GPU.
    #   bf16 needs Ampere (compute 8.0+): A100, L4, RTX 30xx/40xx, H100
    #   fp16 is the right choice on Turing (7.5): the T4 that Kaggle gives you
    # Using bf16 on a T4 technically runs but falls back to slow software paths.
    use_amp = device.type == 'cuda'
    amp_dtype = torch.float16
    if use_amp:
        major, _ = torch.cuda.get_device_capability()
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        print(f'GPU: {torch.cuda.get_device_name(0)}  ->  using '
              f'{"bfloat16" if amp_dtype is torch.bfloat16 else "float16"}')
    # GradScaler is required for fp16 (it stops small gradients underflowing);
    # harmless with bf16.
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_dtype is torch.float16)

    os.makedirs(a.out, exist_ok=True)
    bp, bs = baseline_scores(val_dl, device)
    print(f'\nbicubic baseline on val: PSNR {bp:.2f} dB   SSIM {bs:.4f}')
    print('your model must beat this.\n')

    best = -1e9
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tot, nb = 0.0, 0
        for lr_img, gt in train_dl:
            # Gentle start: ramp the learning rate up over the first few hundred
            # steps so the very first updates cannot throw the weights off.
            if step < warmup_steps:
                for g in opt.param_groups:
                    g['lr'] = a.lr * (step + 1) / warmup_steps
            step += 1

            lr_img = lr_img.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                loss = crit(model(lr_img), gt)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            if ema is not None:
                ema.update(model)
            tot += float(loss.detach())
            nb += 1
        if step >= warmup_steps:
            sched.step()

        # Score the averaged weights, and save those too -- they are what you ship.
        if ema is not None:
            ema.apply_to(model)
        vp, vs = validate(model, val_dl, device)
        flag = ''
        if vp > best:
            best = vp
            torch.save({'model': model.state_dict(), 'cfg': cfg, 'size': a.size,
                        'epoch': ep, 'val_psnr': vp, 'val_ssim': vs},
                       os.path.join(a.out, 'best.pt'))
            flag = '   <- best, saved'
        torch.save({'model': model.state_dict(), 'cfg': cfg, 'size': a.size,
                    'epoch': ep}, os.path.join(a.out, 'last.pt'))
        if ema is not None:
            ema.restore(model)
        print(f'epoch {ep:>3}/{a.epochs}  loss {tot / max(nb, 1):.4f}  '
              f'val PSNR {vp:.2f} (+{vp - bp:.2f} vs bicubic)  SSIM {vs:.4f}  '
              f'{time.time() - t0:.0f}s{flag}')

    print(f'\nbest val PSNR {best:.2f} dB -> {os.path.join(a.out, "best.pt")}')


def smoke_test():
    """Trains briefly on made-up data. Proves the whole pipeline runs."""
    print('SMOKE TEST -- fake data, 2 epochs, just checking nothing is broken.\n')
    import tempfile
    from PIL import Image
    d = tempfile.mkdtemp()
    gd, ld, ed = (os.path.join(d, x) for x in ('gt', 'lr', 'extra'))
    for p in (gd, ld, ed):
        os.makedirs(p)
    rng = np.random.default_rng(0)
    for i in range(12):
        base = rng.random((16, 16)).astype(np.float32)
        g = minmax(np.asarray(Image.fromarray(base).resize((256, 256), Image.BICUBIC))
                   .astype(np.float64))
        lr, _ = degrade(g, rng=rng)
        np.save(os.path.join(gd, f'{i:06d}.npy'), g.astype(np.float32))
        np.save(os.path.join(ld, f'{i:06d}.npy'), lr.astype(np.float32))
    for i in range(6):
        Image.fromarray((rng.random((400, 400)) * 255).astype(np.uint8)).save(
            os.path.join(ed, f'x{i}.png'))
    json.dump({'train': list(range(9)), 'val_hard': [9, 10, 11], 'val_easy': []},
              open(os.path.join(d, 'split.json'), 'w'))

    run(argparse.Namespace(
        gt_dir=gd, lr_dir=ld, split=os.path.join(d, 'split.json'), extra_dir=ed,
        extra_repeat=2, crop=256, size='w16', epochs=2, batch=2, lr=1e-3,
        beta2=0.999, ema=0.99, warmup=5, workers=0, w_fft=0.05, w_ssim=0.10,
        out=os.path.join(d, 'ckpt')))
    print('\nSmoke test done. Two epochs and a saved checkpoint = pipeline works.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gt_dir')
    p.add_argument('--lr_dir')
    p.add_argument('--split', default='split.json')
    p.add_argument('--extra_dir', help='folder of outside photos, e.g. DIV2K_train_HR')
    p.add_argument('--extra_repeat', type=int, default=4,
                   help='random crops taken per outside photo, per epoch')
    p.add_argument('--crop', type=int, default=256)
    p.add_argument('--size', default='w32', choices=['w16', 'w32', 'w64'])
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--beta2', type=float, default=0.999)
    p.add_argument('--ema', type=float, default=0.999,
                   help='weight-averaging decay; 0 turns it off')
    p.add_argument('--warmup', type=int, default=500,
                   help='steps to ramp the learning rate up over')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--w_fft', type=float, default=0.05)
    p.add_argument('--w_ssim', type=float, default=0.10)
    p.add_argument('--out', default='checkpoints')
    p.add_argument('--smoke_test', action='store_true')
    a = p.parse_args()
    if a.smoke_test:
        smoke_test()
    elif not a.gt_dir or not a.lr_dir:
        p.error('need --gt_dir and --lr_dir (or --smoke_test)')
    else:
        run(a)