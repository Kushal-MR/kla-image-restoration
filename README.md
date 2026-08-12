# AI-Based Restoration of Degraded Images

**SEMICON India Hackathon 2026 — Problem Statement 01 (KLA)**
Team **[TEAM_NAME]**

Single-stage restoration of speckle-degraded, 2×-downsampled greyscale images.
A modified NAFNet takes the degraded low-resolution image and produces a clean
full-resolution image in one pass.

| | |
|---|---|
| Architecture | NAFNet-w32, pixel-shuffle ×2 head, bicubic residual skip |
| Parameters | 29.16 M |
| Inference | 12.1 s for 400 images end-to-end (30.2 ms/image, Tesla T4, batch 16; median of six runs) |
| val_hard PSNR / SSIM / LPIPS | **26.68 dB / 0.6859 / 0.3755** |
| bicubic baseline | 22.87 dB / 0.5410 / 0.4480 |

---

## Quick start

```bash
git clone https://github.com/Kushal-MR/kla-image-restoration.git
cd kla-image-restoration

# The trained weights are stored in Git LFS. Without this step best.pt is a
# ~130 byte pointer file and the model will not load.
git lfs install && git lfs pull

# Minimal environment to run the model. Install a torch build matching your
# CUDA version first, e.g. for CUDA 12.8:
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-inference.txt

# restore a directory of degraded images
python inference.py /path/to/test_images /path/to/output_dir
```

`inference.py` takes the test-image directory and the output directory as
positional arguments, loads `weights/best.pt`, and writes one restored file
per input using the **same filename**. It requires no edits to run. Inputs may
be `.npy` (as supplied) or ordinary image files; each output is written in the
same format as its input.

### If `git lfs` is not installed

`git lfs install` fails if the Git LFS binary is missing. Install it first:

```bash
# Debian / Ubuntu
sudo apt-get install git-lfs
# macOS
brew install git-lfs
# Windows, or anything else: https://git-lfs.com
```

**Or skip LFS entirely.** GitHub resolves LFS files for direct downloads, so
the checkpoint can be fetched with an ordinary HTTP request and dropped into
`weights/`:

```bash
curl -L -o weights/best.pt \
  https://github.com/Kushal-MR/kla-image-restoration/raw/main/weights/best.pt
```

To confirm you have the real file and not the pointer: `best.pt` should be
about **108 MB**. If it is a few hundred bytes, it is still a pointer.

**Two requirements files, deliberately.** `requirements.txt` is the complete
`pip freeze` from the training environment, as the submission asks for — 933
packages from a Kaggle image, pinning `torch==2.10.0+cu128`, a CUDA build that
is not published on PyPI. It records what training ran on; it is not
installable on an arbitrary machine. `requirements-inference.txt` is the small
set actually needed to run the model.

Weights are tracked with Git LFS because `best.pt` is 108 MB, above GitHub's
100 MB single-file limit.

---

## Approach

### 1. Recover the degradation operator before training anything

The training pairs are produced by a fixed pipeline with randomised parameters.
Rather than guess it, we measured it.

For a single pixel, the error between the degraded image and the correctly
downsampled clean image is

```
r = I·(g − 1) + n         g ~ Gamma(L, 1/L)   mean 1, variance 1/L
                          n ~ Normal(0, σ²)
```

so its variance is linear in `I²`:

```
var(r) = I²·(1/L) + σ²
```

Binning pixels by intensity and regressing residual variance against `I²`
recovers `1/L` as the slope and `σ²` as the intercept. The estimator self-tests
against synthetic data with known parameters before being trusted — **L is
recovered to within 0.4%**.

**Measured across all 3,200 training pairs:**

| Parameter | Value | Confidence |
|---|---|---|
| Speckle L | **36.82** (per-image 20.6–56.8) | ±3% |
| Gaussian σ | **≈0.011** | moderate — see below |
| Downsampling | smooth (averaging), not pixel-picking | moderate — see note |
| Which smooth kernel | **not identifiable** | box / bilinear / bicubic / lanczos are indistinguishable at this scale |

The kernel result is an honest negative. Rather than guess, we randomise over
all four during synthesis, which also improves robustness to unseen content.

**On excluding `nearest`.** Two diagnostics disagree about how firmly
pixel-picking can be ruled out, so the confidence above is stated as moderate
rather than high. `scripts/day1_setup.py` compares the structure left in the
residual after assuming each kernel: at n=250 it reports `nearest` at 0.0515
against 0.0458 for the best smooth kernel, a 12% margin it declares
inconclusive, and at smaller samples the ordering flips. A second test — the
neighbour correlation of the residual — ranks `nearest` *closest* to white,
the opposite conclusion.

This does not affect the model. Training randomises over the four smooth
kernels regardless, and both validation and testing use KLA's own degraded
files rather than synthetic ones, so the measured scores already reflect
whatever the true kernel is. It is recorded here because the repository's own
tooling prints "inconclusive", and a reproduction that contradicts its README
is worse than an unresolved question stated plainly.

**σ required a correction.** The variance-law fit placed it at 0.02239, but σ²
is that fit's *intercept*, which absorbs any error not scaling with `I²`.
Checking against the real files showed 0.02239 was far too large:

| σ | median minimum pixel | % of images going negative |
|---|---|---|
| 0.0224 | −0.0269 | 74% |
| **real data** | **−0.0022** | **57%** |
| 0.0060 | +0.0036 | 42% |

Only additive noise can push a pixel below zero — multiplicative speckle
cannot — so the depth of the negative tail measures σ directly. Bracketing from
both sides gives ≈0.011. Training randomises σ over [0, 0.025] regardless, so
the correct level is encountered frequently wherever inside that range the truth
actually lies.

### 2. Manufacture training data

Knowing the operator turns 3,200 fixed pairs into an unlimited supply. We apply
it to DIV2K photographs, re-randomising the damage **every time an image is
drawn**, so the network never sees the same noise realisation twice and cannot
memorise a noise pattern.

Sampling ranges are deliberately wider than measured (L ∈ [10.3, 113.5],
log-uniform; σ ∈ [0, 0.025]; kernel uniform over four) so unusual test images
remain inside the training distribution.

Roughly a third of each epoch is genuine KLA pairs, which anchors the model to
the real distribution and guards against over-fitting to our own reconstruction
of the operator. Total 8,640 samples per epoch against 3,200 supplied pairs.

**Verification.** Applying our operator to 200 real ground-truth images and
comparing against KLA's own degraded versions of those same images:

| | real | synthetic |
|---|---|---|
| mean | 0.415 | 0.415 |
| max | 1.359 | 1.352 |
| std | 0.201 | 0.197 |
| recovered L (paired estimator) | 36.8 | 36.9 |

One discrepancy is unresolved: a reference-free variance estimator reads 57.4 on
real data against ~46 on synthetic. Two hypotheses were tested — that noise was
applied before rather than after downsampling, and that σ was inflating the
reading — and neither accounted for it. Since that estimator is known to be
biased high (14–17% on controlled data) while the paired estimator agrees to
within 0.3%, we treat the paired measurement as authoritative.

**Observation on content.** Both the training and test sets are ordinary
photographs — buildings, brick, foliage, fabric — not semiconductor imagery.
This determined the choice of DIV2K over microscopy datasets, which would have
pulled the model toward statistics absent from the challenge.

### 3. Architecture choices

Single stage — degraded LR in, clean HR out. A denoise-then-upscale cascade
doubles latency and compounds errors.

- **Signed log channel.** Speckle is multiplicative, so a logarithm converts it
  to additive noise. We use `sign(x)·log1p(|x|)` rather than
  `log1p(clamp(x, 0))` because 57% of degraded images contain negative pixels;
  clamping would discard that information.
- **Pixel-shuffle ×2 head.** The body runs at low resolution; only the final
  layer expands. Upscaling first would cost 4× the compute.
- **Bicubic residual skip, zero-initialised head.** The network predicts only
  the correction to a bicubic upsample. Residuals transfer across image sources
  far better than absolute intensities, and training begins from a sensible
  baseline rather than noise.
- **LayerNorm** normalises per instance, giving partial invariance to the global
  brightness and contrast shifts that occur between data sources.
- **Output clamped to [0,1].** Every supplied ground-truth image contains
  exactly one pixel at 0.0 and one at 1.0 — they are individually min-max
  normalised — so any predicted value outside that range is guaranteed error.
- **No adversarial training, deliberately.** A GAN would improve LPIPS by
  synthesising plausible texture. In defect inspection a hallucinated particle
  is a worse failure than a slightly soft image, because an engineer may act on
  it. Fidelity is the correct trade here.

### 4. Loss

| Term | Weight | Purpose |
|---|---|---|
| Charbonnier | 1.00 | primary reconstruction term |
| FFT L1 | 0.05 | penalises the "blur it away" shortcut |
| SSIM | 0.10 | directly targets a scored metric |

LPIPS was **not** used as a training objective — see §3 on hallucination. It
nonetheless improved by 16% over the bicubic baseline, which supports the
argument that perceptual quality did not have to be bought with invented detail.

### 5. Training

AdamW, lr 5×10⁻⁴, cosine annealing, 500-step warmup, batch 16, 128×128 LR crops,
40 epochs, fp16 mixed precision on a Tesla T4. An **exponential moving average**
of the weights (decay 0.999) is maintained and used for both validation and the
saved checkpoint.

An earlier run without EMA and with β₂ = 0.9 (NAFNet's own default) showed
validation swinging ±0.5 dB between epochs and degrading over the final ten
epochs while the learning rate annealed toward zero — behaviour inconsistent
with convergence. Setting β₂ = 0.999 and averaging the weights fixed it:
epoch-to-epoch variation fell from ±0.5 dB to ±0.02 dB.

The corrected run's peak (25.51 dB, batch-averaged) is marginally below the
unstable run's (25.77 dB), but the latter was a spike inside a noisy curve whose
neighbouring epochs read 25.43 and 25.50. The stable run is the honest number
and the one we submit.

### 6. Validation that predicts the leaderboard

Training samples are crops of a smaller number of source photographs, in runs of
consecutive indices — neighbouring samples correlate at **+0.190** against a
random-pair floor of −0.003.

A random split would therefore place near-duplicate crops on both sides and
produce a validation score that means nothing. Instead we hold out a
**contiguous block** of indices, guaranteed to contain whole source photographs
without relying on fragile cluster detection. (A thumbnail-clustering approach
was implemented but proved unreliable due to chaining, and is not used.)

| Set | Size | What it measures |
|---|---|---|
| train | 2,639 | — |
| val_easy | 81 | held-out crops of *seen* photos — confirms learning |
| **val_hard** | **480** | *unseen* photos — the only number predicting the test set |

Checkpoints are selected on `val_hard`, scored against **KLA's own degraded
files**, never synthetic ones.

### 7. Inference-time engineering

Scoring measures the whole script — startup, imports, model load, disk I/O and
inference — not just the forward pass. This matters more than it sounds:

| | ms/image |
|---|---|
| forward pass alone | 12.1 |
| **end-to-end measured (median)** | **30.2** |

**69% of the measured time is not the model.** Accordingly:

- `torch.compile` is **not** used: its 30–60 s warm-up falls inside the measured
  window and is not recovered over a 400-image test set.
- `inference.py` imports only `torch` and `numpy`. Metric libraries (`lpips`)
  and plotting are confined to training and reporting code.
- Weights load directly to device; `cudnn.benchmark`; images bucketed by size
  and batched; reads and writes issued on background threads so disk I/O
  overlaps GPU compute.
- **The memory layout and precision are chosen by measurement, not assumption.**
  `channels_last` with fp16 is normally fastest, but cuDNN's kernel coverage
  for depthwise convolutions in that layout is incomplete, and NAFNet is built
  from them — on a Tesla T4 it fails outright with *"FIND was unable to find an
  engine to execute this computation"*. `inference.py` therefore tries
  channels_last+amp, then contiguous+amp, then contiguous+fp32, and keeps the
  first that survives a real forward pass, printing which it selected. The cost
  is one small forward; the alternative is a script that runs on the machine it
  was written on and fails on the benchmarking machine.
- No test-time augmentation — ~0.15 dB for 8× the latency is a poor trade.

**Robustness.** `inference.py` handles single-image directories, mixed input
resolutions, non-power-of-two dimensions, paths containing spaces, execution
from any working directory, inputs containing NaN, and macOS resource-fork
files (`._*.npy`), which archives created on a Mac are full of and which load
as garbage. It also degrades gracefully when the GPU cannot run the preferred
kernel configuration, as described above.

---

## Results

Measured on 480 held-out samples from source photographs absent from training,
using KLA's own degraded files.

| Metric | Bicubic | Ours | Change |
|---|---|---|---|
| PSNR (dB) | 22.87 | **26.68** | +3.81 |
| SSIM | 0.5410 | **0.6859** | +0.1449 |
| LPIPS | 0.4480 | **0.3755** | −0.0726 |

Representative successes and failures at full resolution are in `outputs/`.

**Timing method:** wall-clock around the entire `inference.py` process — Python
startup, imports, model load, 400 reads, inference and 400 writes — on a Tesla
T4 (Kaggle), batch size 16.

Six runs: **9.8, 10.2, 10.4, 10.5, 14.7, 16.1 s**, median **10.4 s**. We quote
the median rather than the best. The script's own internal measurement is
far steadier (7.51–8.81 s, median 7.78), so the spread is startup and page
cache, not the model: the two slowest runs immediately followed a fresh
`git clone`, with nothing yet in the filesystem cache. Kaggle's T4s are also
shared, so I/O contention varies.

KLA benchmarks on an H100, which will be substantially faster, and which
additionally supports bf16 and the channels_last path the T4 could not run.

**Independently reproduced.** Every figure in this README was regenerated from
the checkpoint and the supplied data by `notebooks/replicate.ipynb`, using an
evaluation script written separately from the one that produced the original
numbers. PSNR, SSIM and LPIPS matched to four decimal places, and
`scripts/day1_setup.py` reproduced `configs/split.json`'s validation set
exactly, so the two runs are scored on identical data.

---

## Repository layout

```
├── inference.py             # KLA benchmarking entry point (input_dir, output_dir)
├── train.py                 # reproduces training
├── report_results.py        # PSNR/SSIM/LPIPS, baseline comparison, example figures
├── requirements.txt         # complete pip freeze from the training environment
├── src/
│   ├── nafnet_sr.py         # architecture
│   └── make_training_data.py# measured degradation operator + datasets
├── scripts/
│   ├── inspect_npy.py       # data format and sanity checks
│   ├── day1_setup.py        # recovers L and sigma, builds the source-aware split
│   ├── estimate_noise_blind.py # reference-free noise estimation (test set)
│   └── preflight.py         # repository self-check (no GPU or PyTorch needed)
├── configs/
│   ├── degradation_config.json
│   └── split.json
├── notebooks/
│   ├── kla-hackathon.ipynb  # the training run that produced weights/best.pt
│   └── replicate.ipynb      # re-derives every number in this README
├── weights/best.pt          # Git LFS
└── outputs/
    ├── results/             # metrics.json + success/failure figures
    └── test_out/            # restored test-set images (400 .npy)
```

## Reproducing

`notebooks/replicate.ipynb` runs all of the below end to end on a Kaggle T4.

```bash
# 0. check the repository is internally consistent (seconds, no GPU)
python scripts/preflight.py

# 1. what is actually in the data
python scripts/inspect_npy.py --gt_dir data/train/GT --lr_dir data/train/NoisyLR

# 2. recover the degradation operator and rebuild the split
#    --compare_to verifies the split reproduces the one used for training
python scripts/day1_setup.py --gt_dir data/train/GT --lr_dir data/train/NoisyLR \
       --n 250 --out_split split.json --compare_to configs/split.json

# 3. confirm the recipe reproduces the real damage
python src/make_training_data.py --compare_real \
       --gt_dir data/train/GT --lr_dir data/train/NoisyLR

# 4. train (~2h 52m on a Tesla T4)
python train.py --gt_dir data/train/GT --lr_dir data/train/NoisyLR \
       --split configs/split.json --extra_dir data/DIV2K_train_HR \
       --size w32 --epochs 40 --batch 16 --workers 4 --out weights

# 5. score the model and regenerate the figures
python report_results.py --gt_dir data/train/GT --lr_dir data/train/NoisyLR \
       --split configs/split.json --ckpt weights/best.pt --out outputs/results

# 6. restore the test set (this is what KLA times end to end)
python inference.py data/Test_NoisyLR outputs/test_out
```

The exact training run that produced `weights/best.pt` is preserved in
`notebooks/kla-hackathon.ipynb`.

## How the weights are stored

For anyone reproducing this repository's structure rather than just running it.
`best.pt` is 108 MB, above GitHub's 100 MB single-file limit, so a normal push
is rejected. It is committed through Git LFS:

```bash
git lfs install
git lfs track "weights/*.pt"          # writes the rule into .gitattributes
git add .gitattributes weights/best.pt
git commit -m "store weights via LFS"
git push
```

The restored test outputs are tracked the same way
(`outputs/test_out/*.npy`). Both rules are already committed in
`.gitattributes`, so a fresh clone needs only `git lfs pull` — see Quick start.

One caveat worth knowing: Git LFS on a free account allows 1 GB of bandwidth
per month, and this repository holds roughly 210 MB in LFS. If `git lfs pull`
ever fails with a quota error, use the direct download shown in Quick start.

## External resources

| Resource | Link | Licence | Reference |
|---|---|---|---|
| DIV2K (800 HR photographs) | https://data.vision.ee.ethz.ch/cvl/DIV2K/ | Free for academic research use | Agustsson & Timofte, CVPR-W 2017 |
| NAFNet architecture | https://github.com/megvii-research/NAFNet | MIT | Chen et al., ECCV 2022 |
| LPIPS (AlexNet backbone, metric only) | https://github.com/richzhang/PerceptualSimilarity | BSD-2-Clause | Zhang et al., CVPR 2018 |

No pretrained weights were used — the model is trained from random
initialisation. LPIPS is used for evaluation only, never as a training
objective.

## Limitations and further work

- The specific smooth downsampling kernel could not be identified, and two
  diagnostics disagree on how firmly pixel-picking can be excluded (§1).
- One reference-free noise measurement remains unexplained (§2).
- Checkpoints are selected on PSNR alone, though three metrics are scored. SSIM
  continued improving for several epochs after PSNR peaked; a combined selection
  criterion would be a straightforward improvement.
- LPIPS was not directly optimised. A small perceptual term applied as a
  late-stage fine-tune is the most promising remaining gain, subject to the
  fidelity constraint in §3.

## References

1. Chen et al., *Simple Baselines for Image Restoration* (NAFNet), ECCV 2022.
2. Agustsson & Timofte, *NTIRE 2017 Challenge on Single Image Super-Resolution:
   Dataset and Study* (DIV2K), CVPR Workshops 2017.
3. Zhang et al., *The Unreasonable Effectiveness of Deep Features as a
   Perceptual Metric* (LPIPS), CVPR 2018.
4. Wang et al., *Image Quality Assessment: From Error Visibility to Structural
   Similarity* (SSIM), IEEE TIP 2004.
5. Zhang et al., *Designing a Practical Degradation Model for Deep Blind Image
   Super-Resolution* (BSRGAN), ICCV 2021 — degradation randomisation.
