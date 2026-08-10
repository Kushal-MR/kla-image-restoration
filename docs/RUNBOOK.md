# Runbook — training this model on Kaggle

Written so it can be followed without prior context. Do the one-time setup
once; after that only the "Every session" section matters.

---

## Why it is set up this way

Kaggle cannot read GitHub directly at runtime unless you tell it to, so the
notebook's first cell **clones this repository** into `/kaggle/working/repo`
and imports from there.

This matters because it removes an entire loop. The old way was: change code
-> rebuild a zip -> upload it as a new Kaggle Dataset version -> wait for
Kaggle to process it -> re-import the notebook -> re-run. Now it is: `git
push` on the laptop -> re-run cell 1. Nothing to upload, nothing to wait for.

The notebook file itself is the only thing that ever needs re-importing, and
only if the notebook changes — not when the model or training code changes.

---

## One-time setup

### 1. Attach the two datasets

Right-hand panel -> **Add Input**.

| Dataset | How to find it | What it is |
|---|---|---|
| `div2k-high-resolution-images` | Search `div2k-high-resolution-images`, author `soumikrakshit` | 800 high-resolution photos. Our synthetic training corpus. |
| `kla-train` | Your own, under *Your Datasets* | KLA's provided pairs. Upload `train.zip` once via *Datasets -> New Dataset*. |

You do **not** need a dataset for the code any more. If `kla-code` or
`kla-repo` are still attached, detach them — they are stale copies and having
them attached risks importing the wrong version.

### 2. Turn Internet ON

Right-hand panel -> **Settings** (scroll down) -> **Internet** -> **On**.

Without this `git clone` fails and cell 1 errors. Kaggle may ask you to verify
your phone number the first time; that is a Kaggle account requirement.

### 3. Set the accelerator

Right-hand panel -> **Settings** -> **Accelerator** -> **GPU T4 x2**.

**Do not use GPU P100.** Kaggle still offers it, but current PyTorch builds
have no kernels for its `sm_60` architecture, so every CUDA call fails with
`no kernel image is available for execution on the device`. `train.py` now
detects this and exits with a clear message rather than a cryptic traceback.

### 4. Import the notebook

**File -> Import Notebook -> File tab -> Browse Files**, and choose
`notebooks/kla_train_kaggle.ipynb` from your clone of this repository.

(The *GitHub* tab also works but requires linking your GitHub account to
Kaggle first. The File tab needs no setup.)

---

## Every session

Run the cells in order. Do **not** use *Run All* on a CPU session — cell 4 is
the training run.

| Cell | Accelerator | Time | What you should see |
|---|---|---|---|
| 1 — fetch code | any | ~10 s | `commit: <hash> <message>`, then the two dataset paths and `all inputs found.` |
| 2 — smoke test | None is fine | ~20 s | five checks, ending `ALL CHECKS PASSED` |
| 3 — data check | None is fine | ~15 s | four images, then two lines of statistics that should roughly agree |
| 4 — train | **T4 x2** | hours | a bicubic baseline, then one line per epoch |

Cells 1–3 cost no GPU quota if the accelerator is `None`. Switch to T4 before
cell 4.

### Reading cell 3

```
synthetic  std 0.20  max 1.39  min -0.015  %neg 0.44
KLA real   std 0.20  max 1.38  min -0.003  %neg 0.13
```

These two rows are the check that the synthetic training data is the same
problem as the real data. If they diverge badly, stop — training on
mismatched synthetic data wastes the run.

### Reading cell 4

```
BICUBIC BASELINE   hard SSIM 0.60  PSNR 22.80
EPOCH  0 | loss 0.0421 | easy ... | HARD SSIM 0.63 PSNR 23.4 | vs bicubic +0.60 dB | 6.2 min
```

* **Watch the HARD column.** `easy` is held-out crops of photos the model
  trained on and always looks better; it does not predict the test set.
* **`vs bicubic` is the sanity check.** If it is not clearly positive within
  an epoch or two, something is wrong with the data path and training longer
  will not fix it. Stop and investigate.
* Reference points: bicubic is ~22.8 dB. Perfect denoising with no
  super-resolution at all would be ~32 dB. That is the realistic ceiling.

### Long runs

Use **Save Version -> Save & Run All (Commit)** rather than the interactive
session. It keeps running after you close the browser, which the interactive
session does not. Checkpoints are written to `/kaggle/working/`:
`best.pt` (selected on val_hard), `last.pt`, and `training_log.json`.

---

## When code changes

1. On the laptop: `git add -A && git commit -m "..." && git push`
2. On Kaggle: re-run **cell 1 only**. It fast-forwards the clone and clears
   stale modules from `sys.modules`, so the new code takes effect immediately.
3. Re-run whichever later cells you need.

You only re-import the notebook if the *notebook itself* changed.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Cell 1: `git clone` fails | Internet is off | Settings -> Internet -> On |
| Cell 1: `a dataset is missing` | DIV2K or kla-train not attached | Add Input, attach both |
| `no kernel image is available` | Accelerator is P100 | Switch to T4 x2 |
| Baseline PSNR ~12 instead of ~23 | Running an old copy of the code | Re-run cell 1; check the commit hash it prints |
| `KeyboardInterrupt` at `backward()` | You pressed stop | Not an error |
| `FutureWarning: torch.cuda.amp...` | Deprecated API name | Harmless, ignore |
| Cell 1 pulls but nothing changes | Modules cached from an earlier run | Cell 1 clears them; if stuck, restart the session |

### On the laptop

| Symptom | Fix |
|---|---|
| `Unable to create '.git/index.lock': File exists` | A git process died earlier. `Remove-Item .git\index.lock -Force`, then retry. |
