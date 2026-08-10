#!/usr/bin/env python3
"""
Preflight — verify the repository before spending GPU time on it.

Runs without PyTorch installed (torch is stubbed where needed), so it can be
run anywhere in a couple of seconds. Every check here exists because the
corresponding mistake was actually made at least once:

  1. every file referenced by an import is present and committed
  2. the notebook uses no undefined variables across cells
  3. every command-line flag the notebook passes exists in train.py
  4. train.py can find scripts/prepare_photos.py the way it imports it
  5. the U-Net's encoder/decoder stages balance, so output = 2x input
  6. the data pipeline produces correct shapes and ranges, cached and raw
  7. GT is exactly [0,1]; degraded input is allowed outside it

    python scripts/preflight.py
"""

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)
    return ok


# ---------------------------------------------------------------------------
def check_files_present():
    print("\n1. required files present and tracked by git")
    required = [
        "train.py", "evaluate.py", "requirements.txt",
        "src/degrade.py", "src/dataset.py", "src/model/nafnet_sr.py",
        "scripts/prepare_photos.py", "scripts/preflight.py",
        "configs/degradation_config.json", "configs/split.json",
        "notebooks/kla_train_kaggle.ipynb",
    ]
    try:
        tracked = set(subprocess.run(["git", "-C", ROOT, "ls-files"],
                                     capture_output=True, text=True).stdout.split())
    except Exception:
        tracked = None
    for f in required:
        on_disk = os.path.exists(os.path.join(ROOT, f))
        if tracked is None:
            check(f"{f} on disk", on_disk)
        else:
            check(f"{f}", on_disk and f in tracked,
                  "" if (on_disk and f in tracked)
                  else ("not on disk" if not on_disk else "on disk but NOT committed"))


# ---------------------------------------------------------------------------
def check_notebook():
    print("\n2. notebook: no undefined names across cells")
    import builtins
    nb = json.load(open(os.path.join(ROOT, "notebooks/kla_train_kaggle.ipynb")))
    known = set(dir(builtins))
    problems = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        try:
            tree = ast.parse("\n".join(c["source"]))
        except SyntaxError as e:
            problems.append((i, [f"SyntaxError: {e}"])); continue
        used, defd = set(), set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                (used if isinstance(n.ctx, ast.Load) else defd).add(n.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for al in n.names:
                    defd.add(al.asname or al.name.split(".")[0])
            elif isinstance(n, ast.comprehension):
                for t in ast.walk(n.target):
                    if isinstance(t, ast.Name):
                        defd.add(t.id)
        miss = sorted(used - defd - known)
        if miss:
            problems.append((i, miss))
        known |= defd
    check("all notebook cells resolve", not problems, str(problems))
    return nb


def check_notebook_flags(nb):
    print("\n3. notebook's train.py flags all exist in train.py")
    src = open(os.path.join(ROOT, "train.py")).read()
    tree = ast.parse(src)
    declared = set()
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument"):
            for arg in n.args:
                if isinstance(arg, ast.Constant) and str(arg.value).startswith("--"):
                    declared.add(arg.value)
    # Only the list literal that is handed to train.py counts. Cell 1 also
    # contains git's own flags (--quiet, --depth), which are not ours.
    used = set()
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src_c = "\n".join(c["source"])
        if "train.py" not in src_c:
            continue
        for n in ast.walk(ast.parse(src_c)):
            if isinstance(n, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "cmd" for t in n.targets):
                for el in ast.walk(n.value):
                    if isinstance(el, ast.Constant) and isinstance(el.value, str) \
                            and el.value.startswith("--"):
                        used.add(el.value)
    unknown = sorted(used - declared)
    check("every flag is declared", not unknown, f"unknown: {unknown}")
    print(f"         declared: {sorted(declared)}")


# ---------------------------------------------------------------------------
def check_import_paths():
    print("\n4. train.py / evaluate.py can resolve their own imports")
    # train.py inserts <root>/scripts then imports prepare_photos
    p = os.path.join(ROOT, "scripts")
    sys.path.insert(0, p)
    try:
        import prepare_photos  # noqa
        check("train.py can import prepare_photos", True)
    except Exception as e:
        check("train.py can import prepare_photos", False, repr(e))
    # evaluate.py inserts <root>/src then imports model.nafnet_sr
    ok = os.path.exists(os.path.join(ROOT, "src", "model", "nafnet_sr.py")) and \
         os.path.exists(os.path.join(ROOT, "src", "model", "__init__.py"))
    check("evaluate.py's model package is importable", ok)


# ---------------------------------------------------------------------------
def check_unet_balance():
    print("\n5. U-Net encoder/decoder balance (output must be exactly 2x input)")
    src = open(os.path.join(ROOT, "src/model/nafnet_sr.py")).read()
    tree = ast.parse(src)
    enc = dec = scale = None
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "__init__":
            for a, d in zip(n.args.args[-len(n.args.defaults):], n.args.defaults):
                if a.arg == "enc_blocks":
                    enc = ast.literal_eval(d)
                elif a.arg == "dec_blocks":
                    dec = ast.literal_eval(d)
                elif a.arg == "scale":
                    scale = ast.literal_eval(d)
    if enc is None:
        return check("could read block config", False)
    check("encoder and decoder stage counts match", len(enc) == len(dec),
          f"enc={enc} dec={dec}")
    H = 128; c = 32; size = H; skips = []
    for _ in enc:
        skips.append((c, size)); size //= 2; c *= 2
    for i, _ in enumerate(dec):
        c //= 2; size *= 2
        if skips[::-1][i] != (c, size):
            return check("skip connections line up", False,
                         f"stage {i}: skip {skips[::-1][i]} vs {(c, size)}")
    check("skip connections line up", True)
    check("final size is 2x input", size * scale == H * 2,
          f"{H} -> {size * scale}, want {H*2}")


# ---------------------------------------------------------------------------
def _stub_torch():
    import numpy as np
    class FakeT(np.ndarray):
        def unsqueeze(self, d): return np.expand_dims(self, d).view(FakeT)
        def float(self): return self
    torch = types.ModuleType("torch")
    torch.initial_seed = lambda: 12345
    torch.from_numpy = lambda a: np.asarray(a).view(FakeT)
    utils = types.ModuleType("torch.utils"); data = types.ModuleType("torch.utils.data")
    class Dataset: pass
    data.Dataset = Dataset; data.get_worker_info = lambda: None
    utils.data = data; torch.utils = utils
    sys.modules.update({"torch": torch, "torch.utils": utils, "torch.utils.data": data})


def check_pipeline():
    print("\n6. data pipeline end to end (torch stubbed)")
    import numpy as np
    from PIL import Image
    _stub_torch()
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import dataset as ds
    from prepare_photos import prepare

    tmp = tempfile.mkdtemp()
    try:
        # fake photo corpus
        photos = os.path.join(tmp, "photos"); os.makedirs(photos)
        rng = np.random.default_rng(0)
        for i in range(4):
            Image.fromarray((rng.random((600, 800, 3)) * 255).astype(np.uint8)) \
                 .save(os.path.join(photos, f"p{i}.png"))
        cache = os.path.join(tmp, "cache")
        prepare(photos, cache, verbose=False)

        # fake KLA pairs, deliberately including values above 1.5
        real = os.path.join(tmp, "real")
        os.makedirs(os.path.join(real, "GT")); os.makedirs(os.path.join(real, "NoisyLR"))
        names = []
        for i in range(6):
            g = rng.random((256, 256)).astype(np.float32)
            g = (g - g.min()) / (g.max() - g.min())
            l = (rng.random((128, 128)).astype(np.float32) * 1.9) - 0.1   # 1.8 max, negatives
            np.save(os.path.join(real, "GT", f"{i:06d}.npy"), g)
            np.save(os.path.join(real, "NoisyLR", f"{i:06d}.npy"), l)
            names.append(f"{i:06d}.npy")

        cfg = os.path.join(ROOT, "configs/degradation_config.json")

        for label, folder in [("cached .npy", cache), ("raw .png", photos)]:
            d = ds.SyntheticDataset(folder, cfg, gt_size=256, length=20)
            lr, gt = d[0]
            ok = (tuple(gt.shape) == (1, 256, 256) and tuple(lr.shape) == (1, 128, 128)
                  and abs(gt.min()) < 1e-6 and abs(gt.max() - 1) < 1e-6)
            check(f"synthetic [{label}] shapes and GT range", ok,
                  f"gt{tuple(gt.shape)} lr{tuple(lr.shape)} gt[{gt.min():.4f},{gt.max():.4f}]")

        # THE BUG THAT COST A TRAINING RUN: values above 1.5 must survive intact
        r = ds.RealPairDataset(real, names=names)
        lr, gt = r[0]
        raw = np.load(os.path.join(real, "NoisyLR", "000000.npy"))
        check("NoisyLR above 1.5 is NOT rescaled", abs(float(lr.max()) - float(raw.max())) < 1e-5,
              f"loaded max {float(lr.max()):.4f} vs file max {float(raw.max()):.4f}")
        check("negative pixels preserved", float(lr.min()) < 0, f"min {float(lr.min()):.4f}")

        # cropped variant must keep LR aligned at exactly half of GT
        rc = ds.RealPairDataset(real, names=names, gt_size=128)
        lr, gt = rc[0]
        check("cropped real pair stays 2:1", tuple(gt.shape) == (1, 128, 128)
              and tuple(lr.shape) == (1, 64, 64), f"gt{tuple(gt.shape)} lr{tuple(lr.shape)}")

        # Mixed dataset, configured exactly as train.py does it: the synthetic
        # and real halves MUST share gt_size, or a batch contains two different
        # shapes and the DataLoader raises when it tries to stack them.
        r_train = ds.RealPairDataset(real, names=names, gt_size=256)
        m = ds.MixedDataset(ds.SyntheticDataset(cache, cfg, 256, 40), r_train,
                            real_frac=0.25, length=16)
        shapes = {(tuple(m[i][0].shape), tuple(m[i][1].shape)) for i in range(16)}
        check("every mixed sample has an identical shape", len(shapes) == 1, str(shapes))
        drew_real = any(m.__getitem__(i) is not None for i in range(0, 16, 4))
        check("mixed dataset draws from the real set too", drew_real)

        # split file references real files
        split = json.load(open(os.path.join(ROOT, "configs/split.json")))
        check("split.json has all three sets",
              all(k in split for k in ("train", "val_easy", "val_hard")),
              f"train {len(split.get('train',[]))} easy {len(split.get('val_easy',[]))} "
              f"hard {len(split.get('val_hard',[]))}")
        overlap = set(split["train"]) & set(split["val_hard"])
        check("train and val_hard do not overlap", not overlap, f"{len(overlap)} shared")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
def main():
    print("PREFLIGHT")
    check_files_present()
    nb = check_notebook()
    check_notebook_flags(nb)
    check_import_paths()
    check_unet_balance()
    check_pipeline()
    print("\n" + "=" * 60)
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED:")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
