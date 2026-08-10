"""
PHASE 2, PART 1 — THE DEGRADATION GENERATOR
===========================================
This is the payoff from Phase 0. Point it at ANY clean photo and it produces a
degraded version the same way KLA did. That turns 3200 fixed training pairs
into as many as you want.

TWO IDEAS THAT MAKE THIS WORK
-----------------------------
1. FRESH DAMAGE EVERY TIME.
   We do NOT pre-generate a fixed set of noisy images. Every time an image
   comes up in training it gets brand-new random damage. The model therefore
   never sees the same grainy picture twice, so it cannot memorise a
   particular pattern of grain -- it has to learn what grain IS.

2. DELIBERATELY WIDER THAN REALITY.
   We measured L around 37, ranging 22-54 between images. We train on
   L from 11 to 108 -- roughly half to double the real range. If a test image
   is noisier or cleaner than anything in training, the model panics. Training
   wider than reality means unusual test images are still familiar territory.
   Same logic for sigma, and for the shrink method (we randomise over all four
   smooth ones because Phase 0 proved we cannot tell which they used).

WHAT WE DO *NOT* RANDOMISE
--------------------------
The scale factor. It is always exactly 2x. Phase 0 confirmed it, the problem
statement confirms it. No need to be clever.
"""

import json
import numpy as np
from PIL import Image

# The four smooth shrink methods. Phase 0 ruled out "nearest" but could not
# separate these four, so we use all of them.
_KERNELS = {
    "box":      Image.BOX,
    "bilinear": Image.BILINEAR,
    "bicubic":  Image.BICUBIC,
    "lanczos":  Image.LANCZOS,
}


def _gaussian_blur(img, sigma):
    """Small separable Gaussian blur for float32 images (PIL won't do mode 'F')."""
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    pad = np.pad(img, radius, mode="reflect")
    # blur along rows, then along columns
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, pad)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, tmp)
    return out.astype(np.float32)


class Degrader:
    """Applies KLA's degradation recipe with randomised strength."""

    def __init__(self, cfg_path=None, cfg=None):
        if cfg is None:
            cfg = json.load(open(cfg_path))
        self.L_min = float(cfg["L_train_min"])
        self.L_max = float(cfg["L_train_max"])
        self.s_min = float(cfg["sigma_train_min"])
        self.s_max = float(cfg["sigma_train_max"])
        self.kernels = [k for k in cfg["downsample_kernels"] if k in _KERNELS]
        self.blur_prob = float(cfg.get("pre_blur_prob", 0.10))

    # ------------------------------------------------------------------
    def sample_params(self, rng):
        """
        Pick a random damage strength.

        NOTE ON log-uniform: we draw L uniformly in LOG space, not linear.
        Linear sampling between 11 and 108 would spend most of its time in the
        clean half (L>60), because that's where most of the number line is.
        Log sampling gives equal attention to "quite noisy" and "barely noisy".
        Since noise strength is really 1/L, log space is the natural scale.
        """
        L = float(np.exp(rng.uniform(np.log(self.L_min), np.log(self.L_max))))
        sigma = float(rng.uniform(self.s_min, self.s_max))
        kernel = self.kernels[rng.integers(len(self.kernels))]
        pre_blur = bool(rng.random() < self.blur_prob)
        return L, sigma, kernel, pre_blur

    # ------------------------------------------------------------------
    def __call__(self, gt, rng, params=None):
        """
        gt  : 2D float32 array, already min-max normalised to [0,1]
        rng : a numpy random generator
        returns: degraded array, half the size of gt
        """
        L, sigma, kernel, pre_blur = params or self.sample_params(rng)
        h, w = gt.shape

        # OPTIONAL slight blur before shrinking, ~1 image in 10.
        # Covers the possibility that some test images came off a slightly
        # different/softer instrument than the training ones.
        # (PIL's GaussianBlur refuses float images, so we do it by hand --
        #  it's just a weighted average of nearby pixels, applied to rows then
        #  columns. Doing it in two 1D passes is much cheaper than one 2D pass.)
        src = gt.astype(np.float32)
        if pre_blur:
            src = _gaussian_blur(src, sigma=float(rng.uniform(0.3, 0.8)))

        pil = Image.fromarray(src, mode="F")

        # 1. SHRINK BY HALF (this is the "resolution loss" half of the problem)
        clean = np.asarray(pil.resize((w // 2, h // 2), _KERNELS[kernel]),
                           dtype=np.float32)

        # 2. MULTIPLY IN THE SPECKLE GRAIN.
        #    Gamma(shape=L, scale=1/L) has mean exactly 1 and variance 1/L.
        #    mean 1 is why overall brightness is preserved -- which is exactly
        #    what we observed in the real data (clean mean 0.2182 vs degraded
        #    mean 0.2184). Big L -> tightly clustered around 1 -> clean image.
        g = rng.gamma(shape=L, scale=1.0 / L, size=clean.shape).astype(np.float32)

        # 3. ADD THE GAUSSIAN STATIC on top (this is what makes pixels negative)
        n = rng.normal(0.0, sigma, clean.shape).astype(np.float32)

        # NO CLIPPING. The real degraded images go below 0 and above 1.
        # Clipping here would make your synthetic data differ from the real
        # thing in exactly the way the problem statement warns about.
        return clean * g + n


# ----------------------------------------------------------------------
def minmax(x):
    """
    Per-image min-max normalisation -- squash to exactly [0,1].
    Phase 0 proved every KLA ground-truth image has exactly one pixel at 0
    and one at 1, so we must do this to our crops BEFORE degrading them,
    or our synthetic pairs won't match their pipeline.
    """
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def to_grey(img):
    """
    RGB -> greyscale. The challenge is explicitly single-channel.
    Weights 0.299/0.587/0.114 are the standard luminance weights: human eyes
    are most sensitive to green, least to blue.
    """
    if img.ndim == 2:
        return img.astype(np.float32)
    return (img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114).astype(np.float32)
