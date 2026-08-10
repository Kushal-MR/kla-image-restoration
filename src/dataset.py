"""
PHASE 2, PART 2 — THE DATASET CLASSES
=====================================
A PyTorch "Dataset" is just an object that answers two questions:
    len(ds)   -> how many samples are there?
    ds[i]     -> give me sample number i

PyTorch's DataLoader calls ds[i] over and over, in parallel worker processes,
and stacks the results into batches. So everything expensive per-sample --
cropping, degrading, converting to tensors -- happens inside __getitem__.

WE HAVE THREE SOURCES OF DATA
-----------------------------
  SyntheticDataset : DIV2K photos -> crop -> min-max -> degrade fresh each time.
                     Effectively unlimited. This is the OOD workhorse.
  RealPairDataset  : KLA's 3200 provided pairs, used as-is.
                     Anchors us to the real distribution.
  MixedDataset     : draws from both, 75% synthetic / 25% real by default.

WHY MIX RATHER THAN GO PURE SYNTHETIC
-------------------------------------
Our recipe is very good but not perfect (sigma is only known to a factor of
two, and we can't identify the exact shrink kernel). Training ONLY on our
reconstruction risks optimising for our imagined version of the problem.
Keeping a quarter of every batch as genuine KLA pairs anchors us to reality.
"""

import glob, json, os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from degrade import Degrader, minmax, to_grey


# ---------------------------------------------------------------------------
def _load_any(path):
    """
    Load a .npy or an image file (.png/.jpg) as a float32 2D greyscale array.

    The rescaling depends on the FILE TYPE, never on the pixel values.
    An earlier version used `if a.max() > 1.5: a /= 255`, which looked
    harmless but destroyed a quarter of the degraded images: speckle noise
    legitimately pushes NoisyLR above 1.5 (the problem statement says so
    explicitly), and those images were being divided by 255 into near-black.
    Never infer a file's scale from its content when the content is allowed
    to exceed the expected range -- that is the whole premise of this task.
    """
    if path.endswith(".npy"):
        a = np.asarray(np.load(path), dtype=np.float32)   # already 0-1, leave alone
    else:
        im = Image.open(path)
        a = np.asarray(im, dtype=np.float32)
        a = a / (65535.0 if im.mode == "I;16" else 255.0)  # 8- or 16-bit image file
    a = to_grey(a)
    return np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)


def _to_tensor(x):
    """2D numpy array -> torch tensor of shape (1, H, W). The 1 is the channel."""
    return torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).float()


# ---------------------------------------------------------------------------
class SyntheticDataset(Dataset):
    """
    DIV2K (or any folder of photos) -> unlimited training pairs.

    Each __getitem__:
        1. pick a photo
        2. cut a random gt_size x gt_size square out of it
        3. random flip / 90-degree rotation  (free extra variety)
        4. min-max normalise the crop      <- matches KLA's pipeline exactly
        5. apply the degradation recipe with FRESH random strength
    """

    def __init__(self, photo_dir, cfg_path, gt_size=256, length=20000,
                 exts=("npy", "png", "jpg", "jpeg")):
        self.files = []
        for e in exts:
            found = sorted(glob.glob(os.path.join(photo_dir, "**", f"*.{e}"), recursive=True))
            found = [f for f in found if not os.path.basename(f).startswith("._")]
            if found:
                # Prefer a prepared .npy cache if one exists. Mixing cached and
                # raw photos would silently reintroduce the decode cost.
                self.files = found
                break
        if not self.files:
            raise RuntimeError(f"no photos found under {photo_dir}")
        self.cached = self.files[0].endswith(".npy")
        self.gt_size = gt_size
        self.length = length          # "epoch size" -- arbitrary, since data is infinite
        self.degrader = Degrader(cfg_path)
        self._cache = {}

    def __len__(self):
        return self.length

    def _get_photo(self, idx):
        """
        Return the photo as a 2D array, as cheaply as possible.

        With a prepared cache (see scripts/prepare_photos.py) we memory-map
        the file: nothing is read until a crop is sliced out of it, so the
        cost is the crop, not the photo. Measured at ~0.1 ms against ~144 ms
        to decode the original PNG.

        Without a cache we fall back to decoding, and hold a few decoded
        photos in RAM to take the edge off.
        """
        if self.cached:
            if idx not in self._cache:
                if len(self._cache) > 256:      # memmaps are cheap; keep plenty
                    self._cache.clear()
                self._cache[idx] = np.load(self.files[idx], mmap_mode="r")
            return self._cache[idx]

        if idx not in self._cache:
            if len(self._cache) > 24:
                self._cache.clear()
            self._cache[idx] = _load_any(self.files[idx])
        return self._cache[idx]

    def __getitem__(self, i):
        # Each worker process needs its OWN random stream, otherwise every
        # worker generates identical "random" noise. Mixing in the index and
        # the worker id guarantees they differ.
        wi = torch.utils.data.get_worker_info()
        seed = (i * 9973 + (wi.id if wi else 0) * 7919 + torch.initial_seed()) % (2**31)
        rng = np.random.default_rng(seed)

        img = self._get_photo(int(rng.integers(len(self.files))))
        H, W = img.shape
        s = self.gt_size

        # Slice the crop out FIRST, then materialise it. When img is a memmap
        # this reads only the crop's bytes off disk instead of the whole photo.
        y = int(rng.integers(0, max(H - s, 0) + 1))
        x = int(rng.integers(0, max(W - s, 0) + 1))
        crop = np.asarray(img[y:y + s, x:x + s], dtype=np.float32)

        # Scale by DTYPE, never by pixel values. A cached photo is uint8;
        # a decoded one has already been divided by 255 in _load_any.
        if img.dtype == np.uint8:
            crop = crop / 255.0

        if crop.shape[0] < s or crop.shape[1] < s:     # photo smaller than the crop
            crop = np.pad(crop, ((0, max(0, s - crop.shape[0])),
                                 (0, max(0, s - crop.shape[1]))), mode="reflect")

        # cheap geometric augmentation: 8 possible orientations
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
        k = int(rng.integers(4))
        if k:
            crop = np.rot90(crop, k)
        crop = np.ascontiguousarray(crop)

        gt = minmax(crop)                        # exactly [0,1], like KLA's GT
        lr = self.degrader(gt, rng)              # fresh damage every single time
        return _to_tensor(lr), _to_tensor(gt)


# ---------------------------------------------------------------------------
class RealPairDataset(Dataset):
    """KLA's provided pairs, loaded as-is. Optionally restricted to a split."""

    def __init__(self, root, names=None, gt_size=None):
        gt_dir = os.path.join(root, "GT")
        lr_dir = os.path.join(root, "NoisyLR")
        if names is None:
            names = sorted(os.path.basename(f) for f in glob.glob(os.path.join(gt_dir, "*.npy")))
        self.pairs = [(os.path.join(gt_dir, n), os.path.join(lr_dir, n)) for n in names]
        self.gt_size = gt_size        # None = use the whole image (for validation)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        gf, lf = self.pairs[i]
        gt = _load_any(gf)
        lr = _load_any(lf)

        if self.gt_size:              # random crop for training
            s = self.gt_size
            H, W = gt.shape
            if H > s:
                rng = np.random.default_rng((i * 7717 + torch.initial_seed()) % (2**31))
                # crop must land on an EVEN pixel so the LR crop lines up exactly
                y = int(rng.integers(0, (H - s) // 2 + 1)) * 2
                x = int(rng.integers(0, (W - s) // 2 + 1)) * 2
                gt = gt[y:y + s, x:x + s]
                lr = lr[y // 2:y // 2 + s // 2, x // 2:x // 2 + s // 2]

        return _to_tensor(lr), _to_tensor(gt)


# ---------------------------------------------------------------------------
class MixedDataset(Dataset):
    """Draw mostly from synthetic, partly from the real provided pairs."""

    def __init__(self, synth, real, real_frac=0.25, length=None):
        self.synth, self.real = synth, real
        self.real_frac = real_frac
        self.length = length or len(synth)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        # Deterministic interleave: every 1/real_frac-th index comes from the
        # real set. Deterministic rather than random so each epoch has exactly
        # the intended mix, no luck involved.
        every = int(round(1.0 / self.real_frac))
        if i % every == 0:
            return self.real[(i // every) % len(self.real)]
        return self.synth[i % len(self.synth)]


# ---------------------------------------------------------------------------
def load_split(split_path):
    """Read split.json -> (train_names, val_easy_names, val_hard_names)."""
    s = json.load(open(split_path))
    return s["train"], s["val_easy"], s["val_hard"]
