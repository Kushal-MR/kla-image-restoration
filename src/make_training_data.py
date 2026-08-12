"""
Turns ordinary photos into KLA-style training pairs.

The recipe, recovered from your 3200 training pairs:
    clean 256x256  ->  shrink by half (smooth)  ->  x grain (L~37)  ->  + static (~0.022)

Two ways to use this file:

1. `degrade()` -- the damage function on its own. Feed it a clean image, get back a
   spoiled half-size one.

2. `RestorationDataset` -- a PyTorch dataset that damages images fresh every time
   they are used. This is the important one. The model then never sees the same
   grain twice, so it cannot memorise any single noise pattern; it has to learn what
   grain actually is. That is what makes it survive the unseen test images.

Check it works:
    python make_training_data.py --self_test
    python make_training_data.py --compare_real --lr_dir path/to/real/NoisyLR
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image

# Measured from your training data by day1_setup.py
L_MEASURED = 36.82

# CORRECTED after comparing synthetic output against the real NoisyLR files.
# day1_setup's line-fit put the static at 0.02239, but that value is the fit's
# *intercept*, which quietly absorbs any part of the noise that does not scale
# with brightness -- including fitting error. Checked directly instead: 57% of
# real images dip just below zero, with a typical minimum of -0.0022. Only the
# additive static can push a pixel below zero, so that shallow dip pins it down.
# Pinned down by testing against the real files at two settings:
#   0.0224 -> minimum -0.0269, 74% negative   (too much static)
#   0.0060 -> minimum +0.0036, 42% negative   (too little)
#   real   -> minimum -0.0022, 57% negative
# Interpolating the two observables gives 0.009 and 0.014 respectively, so the
# true value sits around 0.011. It is not worth chasing further: training draws
# sigma at random from the range below, so the model sees the correct level
# regularly wherever inside that range the truth actually sits.
SIGMA_MEASURED = 0.011

# Deliberately wider than measured, so unusual test images do not surprise the model
L_RANGE = (10.3, 113.5)
SIGMA_RANGE = (0.0, 0.025)
KERNELS = ['bicubic', 'bilinear', 'box', 'lanczos']

_PIL = {
    'bicubic': Image.BICUBIC,
    'bilinear': Image.BILINEAR,
    'box': Image.BOX,
    'lanczos': Image.LANCZOS,
    'nearest': Image.NEAREST,
}


def minmax(img, eps=1e-8):
    """Stretch an image to exactly [0,1].

    KLA did this to every ground truth image -- each one has exactly one pixel at 0
    and one at 1. Matching it means your synthetic images have the same statistics
    as theirs.
    """
    img = img.astype(np.float64)
    lo, hi = img.min(), img.max()
    if hi - lo < eps:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def shrink(img, out_hw, kernel='bicubic'):
    im = Image.fromarray(img.astype(np.float32))
    return np.asarray(im.resize((out_hw[1], out_hw[0]), _PIL[kernel])).astype(np.float64)


def degrade(gt, scale=2, L=None, sigma=None, kernel=None, rng=None, blur=False):
    """Apply KLA's damage to a clean image.

    gt      : clean image, already in [0,1]
    returns : spoiled image at half size, NOT clipped

    Not clipping is deliberate. Grain pushes bright pixels above 1 and static pulls
    dark ones below 0. Both are real information about the noise -- clipping throws
    it away.
    """
    rng = rng or np.random.default_rng()
    if L is None:
        # log-uniform: gives equal weight to "very grainy" and "barely grainy",
        # which a plain uniform draw would not
        L = float(np.exp(rng.uniform(np.log(L_RANGE[0]), np.log(L_RANGE[1]))))
    if sigma is None:
        sigma = float(rng.uniform(*SIGMA_RANGE))
    if kernel is None:
        kernel = KERNELS[rng.integers(len(KERNELS))]

    img = gt.astype(np.float64)

    if blur:
        # Occasionally soften first, as if shot on a slightly different lens
        k = float(rng.uniform(0.3, 0.8))
        small = shrink(img, (max(8, int(img.shape[0] / (1 + k))),
                             max(8, int(img.shape[1] / (1 + k)))), 'bicubic')
        img = shrink(small, img.shape, 'bicubic')

    h, w = img.shape[0] // scale, img.shape[1] // scale
    clean = shrink(img, (h, w), kernel)

    speckle = rng.gamma(shape=L, scale=1.0 / L, size=clean.shape)
    noisy = clean * speckle + rng.normal(0.0, sigma, size=clean.shape)
    return noisy, dict(L=L, sigma=sigma, kernel=kernel)


def signed_log(x):
    """Second input channel for the model.

    Grain multiplies, so taking a log turns it into ordinary additive noise, which
    networks handle far more easily. The `sign` part is what lets it cope with the
    negative pixels that static creates -- a plain log would throw those away.
    """
    return np.sign(x) * np.log1p(np.abs(x))


# ----------------------------------------------------------------------
# PyTorch dataset
# ----------------------------------------------------------------------

def make_dataset_class():
    """Defined inside a function so this file still runs without torch installed."""
    import torch
    from torch.utils.data import Dataset

    class RestorationDataset(Dataset):
        """Serves (spoiled, clean) pairs.

        mode='synthetic' : damage is regenerated fresh each time. Use for training.
        mode='fixed'     : uses the real NoisyLR files as-is. Use for validation, so
                           your score is measured against KLA's actual damage.
        """

        def __init__(self, gt_paths, lr_paths=None, mode='synthetic',
                     crop=None, augment=True, seed=None, use_log_channel=False):
            # use_log_channel defaults to False on purpose: NAFNetSR builds the
            # signed-log channel itself inside forward(). If the dataset added one
            # too, the model would receive 2 channels, log them again into 4, and
            # the first conv would fail. Leave this False unless the model has
            # use_log_channel=False.
            self.gt_paths = list(gt_paths)
            self.lr_paths = list(lr_paths) if lr_paths else None
            self.mode = mode
            self.crop = crop
            self.augment = augment
            self.use_log_channel = use_log_channel
            self.seed = seed
            if mode == 'fixed' and not self.lr_paths:
                raise ValueError("mode='fixed' needs lr_paths")

        def __len__(self):
            return len(self.gt_paths)

        def _load(self, path):
            a = np.load(path).astype(np.float64)
            # A single NaN makes the loss NaN and training silently stops. Cheap guard.
            return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

        def __getitem__(self, i):
            # Fixed seed -> identical validation every epoch, so the numbers compare
            rng = (np.random.default_rng(self.seed + i) if self.seed is not None
                   else np.random.default_rng())

            gt = self._load(self.gt_paths[i])

            if self.mode == 'fixed':
                lr = self._load(self.lr_paths[i])
            else:
                if self.crop and gt.shape[0] > self.crop:
                    y = rng.integers(0, gt.shape[0] - self.crop + 1)
                    x = rng.integers(0, gt.shape[1] - self.crop + 1)
                    gt = gt[y:y + self.crop, x:x + self.crop]
                gt = minmax(gt)
                lr, _ = degrade(gt, rng=rng, blur=bool(rng.random() < 0.1))

            if self.augment and self.mode != 'fixed':
                if rng.random() < 0.5:
                    gt, lr = gt[:, ::-1], lr[:, ::-1]
                if rng.random() < 0.5:
                    gt, lr = gt[::-1], lr[::-1]
                k = int(rng.integers(4))
                if k:
                    gt, lr = np.rot90(gt, k), np.rot90(lr, k)

            gt = np.ascontiguousarray(gt, dtype=np.float32)
            lr = np.ascontiguousarray(lr, dtype=np.float32)

            x = torch.from_numpy(lr)[None]
            if self.use_log_channel:
                x = torch.cat([x, torch.from_numpy(
                    signed_log(lr).astype(np.float32))[None]], 0)
            return x, torch.from_numpy(gt)[None]

    return RestorationDataset


# ----------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------

def _patch_stats(img, size=8):
    h, w = img.shape
    h, w = h - h % size, w - w % size
    p = img[:h, :w].reshape(h // size, size, w // size, size)
    p = p.transpose(0, 2, 1, 3).reshape(-1, size * size)
    return p.mean(1), p.var(1)


def _blind_L(images, n_bins=24, flat_pct=10):
    """Estimate grain strength from spoiled images alone (no clean version needed)."""
    m, v = [], []
    for im in images:
        a, b = _patch_stats(np.asarray(im, dtype=np.float64))
        m.append(a)
        v.append(b)
    m, v = np.concatenate(m), np.concatenate(v)
    lo, hi = np.percentile(m, [2, 98])
    idx = np.digitize(m, np.linspace(lo, hi, n_bins + 1)) - 1
    xs, ys = [], []
    for b in range(n_bins):
        s = idx == b
        if s.sum() < 40:
            continue
        xs.append(m[s].mean() ** 2)
        ys.append(np.percentile(v[s], flat_pct))
    if len(xs) < 5:
        return np.nan
    A = np.stack([np.array(xs), np.ones(len(xs))], 1)
    slope, _ = np.linalg.lstsq(A, np.array(ys), rcond=None)[0]
    return 1.0 / slope if slope > 0 else np.nan


def self_test():
    print('Does the generator reproduce the recipe it was given?\n')
    rng = np.random.default_rng(0)
    from PIL import Image as I
    for true_L in [15.0, 36.82, 90.0]:
        imgs = []
        for _ in range(40):
            base = rng.random((20, 20)).astype(np.float32)
            g = np.asarray(I.fromarray(base).resize((256, 256), I.BICUBIC)).astype(np.float64)
            g = minmax(g)
            lr, _ = degrade(g, L=true_L, sigma=0.0, kernel='bicubic', rng=rng)
            imgs.append(lr)
        est = _blind_L(imgs)
        print(f'  asked for L={true_L:<6} -> measured back {est:>6.1f}  '
              f'({100*abs(est-true_L)/true_L:.0f}% off, estimator is biased high)')

    g = minmax(rng.random((256, 256)))
    lr, meta = degrade(g, rng=rng)
    print(f'\n  shape: {g.shape} -> {lr.shape}')
    print(f'  random draw: L={meta["L"]:.1f}, sigma={meta["sigma"]:.4f}, '
          f'kernel={meta["kernel"]}')
    print(f'  output range {lr.min():.3f} to {lr.max():.3f} '
          f'(goes outside [0,1] -- correct)')


def compare_real(gt_dir, lr_dir, n=200):
    """The check that matters: do fake pairs look like KLA's real ones?"""
    print('Comparing synthetic damage against the real thing.\n')
    rng = np.random.default_rng(0)

    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.npy')))[:n]
    real_lr = [np.load(f).astype(np.float64)
               for f in sorted(glob.glob(os.path.join(lr_dir, '*.npy')))[:n]]
    if not gt_files or not real_lr:
        print('need both --gt_dir and --lr_dir with .npy files')
        return

    fake_lr = []
    for f in gt_files:
        g = minmax(np.nan_to_num(np.load(f).astype(np.float64)))
        lr, _ = degrade(g, L=L_MEASURED, sigma=SIGMA_MEASURED, kernel='bicubic', rng=rng)
        fake_lr.append(lr)

    def prof(name, ims):
        mx = np.array([x.max() for x in ims])
        mn = np.array([x.min() for x in ims])
        mu = np.array([x.mean() for x in ims])
        sd = np.array([x.std() for x in ims])
        print(f'  {name:<10} max {np.median(mx):.3f} | min {np.median(mn):+.4f} | '
              f'mean {np.median(mu):.3f} | std {np.median(sd):.3f} | '
              f'{100*(mn<0).mean():.0f}% go negative')

    print(f'  (real n={len(real_lr)}, synthetic n={len(fake_lr)})')
    prof('REAL', real_lr)
    prof('SYNTHETIC', fake_lr)
    print(f'\n  grain measured back:  real {_blind_L(real_lr):.1f}   '
          f'synthetic {_blind_L(fake_lr):.1f}')
    print('\n  Those two numbers should be close. If they are, your fake pairs')
    print('  train the model for the right exam.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--self_test', action='store_true')
    ap.add_argument('--compare_real', action='store_true')
    ap.add_argument('--gt_dir')
    ap.add_argument('--lr_dir')
    a = ap.parse_args()
    if a.compare_real:
        compare_real(a.gt_dir, a.lr_dir)
    else:
        self_test()
