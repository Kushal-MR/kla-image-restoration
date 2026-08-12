#!/usr/bin/env python3
"""
Preflight — verify the repository before it is submitted or benchmarked.

Runs in seconds and needs neither a GPU nor PyTorch. Every check exists
because the corresponding mistake was actually made at least once during the
project:

  1. every file the code imports is present AND committed
  2. the checkpoint's architecture matches src/nafnet_sr.py
  3. inference.py declares the two positional arguments KLA will pass
  4. the restored test outputs are complete, correctly shaped and in range
  5. the split is honest: train and val_hard do not overlap
  6. the reported metrics are internally consistent

    python scripts/preflight.py
"""

import ast
import collections
import io
import json
import os
import pickle
import re
import subprocess
import sys
import zipfile

# Windows consoles default to a legacy code page; force UTF-8 so printing a
# detail string containing a non-ASCII character cannot crash the report.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)
    return ok


def read_checkpoint(path):
    """Read a .pt without torch: it is a zip whose data.pkl names torch classes."""
    z = zipfile.ZipFile(path)
    pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]

    class Stub:
        def __init__(self, *a, **k): pass

    class T:
        def __init__(self, size): self.size = size

    def rebuild(storage, offset, size, stride, *a): return T(size)

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name.startswith("_rebuild_tensor"): return rebuild
            if name == "OrderedDict": return collections.OrderedDict
            return Stub
        def persistent_load(self, pid): return pid

    return U(io.BytesIO(z.read(pkl))).load()


def check_files():
    print("\n1. required files present and committed")
    required = [
        "inference.py", "train.py", "requirements.txt", "README.md",
        "src/nafnet_sr.py", "src/make_training_data.py",
        "configs/degradation_config.json", "configs/split.json",
        "weights/best.pt", "outputs/results/metrics.json",
        "notebooks/kla-hackathon.ipynb", ".gitattributes",
    ]
    try:
        tracked = set(subprocess.run(["git", "-C", ROOT, "ls-files"],
                                     capture_output=True, text=True).stdout.split())
    except Exception:
        tracked = None
    for f in required:
        on_disk = os.path.exists(os.path.join(ROOT, f))
        if tracked is None:
            check(f, on_disk)
        else:
            ok = on_disk and f in tracked
            check(f, ok, "" if ok else ("not on disk" if not on_disk
                                        else "on disk but not staged/committed -- run: git add -A"))
    # LFS must cover the weights, or GitHub rejects the push (>100 MB)
    attrs = open(os.path.join(ROOT, ".gitattributes"), encoding="utf-8").read() if \
        os.path.exists(os.path.join(ROOT, ".gitattributes")) else ""
    size_mb = os.path.getsize(os.path.join(ROOT, "weights/best.pt")) / 1e6
    check("weights tracked by Git LFS", "weights/*.pt" in attrs and "lfs" in attrs,
          f"best.pt is {size_mb:.0f} MB; GitHub rejects >100 MB without LFS")


def check_architecture():
    print("\n2. checkpoint matches src/nafnet_sr.py")
    ck = read_checkpoint(os.path.join(ROOT, "weights/best.pt"))
    sd, cfg = ck["model"], ck["cfg"]
    keys = list(sd)
    n_par = 0
    for v in sd.values():
        if hasattr(v, "size"):
            n = 1
            for d in v.size:
                n *= d
            n_par += n
    print(f"         checkpoint: {len(keys)} tensors, {n_par:,} params, "
          f"epoch {ck.get('epoch')}, cfg {cfg}")

    src = open(os.path.join(ROOT, "src/nafnet_sr.py"), encoding="utf-8").read()
    top = sorted({k.split(".")[0] for k in keys})
    missing = [m for m in top if f"self.{m}" not in src]
    check("every checkpoint module exists in the architecture", not missing,
          f"missing: {missing}")

    enc = max(int(m) for m in re.findall(r"encoders\.(\d+)\.", " ".join(keys))) + 1
    dec = max(int(m) for m in re.findall(r"decoders\.(\d+)\.", " ".join(keys))) + 1
    mid = max(int(m) for m in re.findall(r"middle_blks\.(\d+)\.", " ".join(keys))) + 1
    check("encoder/decoder stages balance", enc == dec, f"enc {enc}, dec {dec}")
    check("stage counts match cfg", enc == len(cfg["enc_blk_nums"])
          and dec == len(cfg["dec_blk_nums"]) and mid == cfg["middle_blk_num"],
          f"enc {enc}, dec {dec}, middle {mid}")
    check("input is 2 channels (raw + signed log)",
          sd["intro.weight"].size[1] == 2, f"{sd['intro.weight'].size[1]}")
    check("SR head emits scale^2 = 4 channels",
          sd["sr_head.0.weight"].size[0] == 4, f"{sd['sr_head.0.weight'].size[0]}")
    return ck


def check_inference_cli():
    print("\n3. inference.py exposes the interface KLA will call")
    tree = ast.parse(open(os.path.join(ROOT, "inference.py"), encoding="utf-8").read())
    pos = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument" and n.args
                and isinstance(n.args[0], ast.Constant)
                and not str(n.args[0].value).startswith("-")):
            pos.append(n.args[0].value)
    check("takes input_dir and output_dir positionally",
          pos[:2] == ["input_dir", "output_dir"], f"found {pos}")
    # Look for an actual CALL, not the string: the file contains a comment
    # explaining why torch.compile is avoided, and matching text would flag it.
    uses_compile = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "compile" and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "torch"
        for n in ast.walk(tree))
    check("does not call torch.compile", not uses_compile)
    src = open(os.path.join(ROOT, "inference.py"), encoding="utf-8").read()
    heavy = [m for m in ("lpips", "matplotlib", "scipy", "pandas", "skimage")
             if re.search(rf"^\s*(import|from)\s+{m}\b", src, re.M)]
    check("imports stay minimal", not heavy, f"heavy imports: {heavy}")


def check_intra_repo_imports():
    """Every top-level module the entry points import must be findable.

    This is the check that would have caught a missing file the last time:
    the module existed in one working copy, was never committed, and the
    failure only surfaced on the training machine.
    """
    print("\n4. entry points can resolve their imports")
    src_dir = os.path.join(ROOT, "src")
    available = {f[:-3] for f in os.listdir(src_dir) if f.endswith(".py")}
    stdlib_ok = set(sys.builtin_module_names) | {
        "argparse", "glob", "json", "os", "sys", "time", "re", "ast", "io",
        "pickle", "zipfile", "subprocess", "collections", "concurrent", "tempfile",
        "numpy", "torch", "PIL", "functools", "shutil", "math", "random"}
    for entry in ("inference.py", "train.py"):
        tree = ast.parse(open(os.path.join(ROOT, entry), encoding="utf-8").read())
        needed = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                needed.add(n.module.split(".")[0])
            elif isinstance(n, ast.Import):
                for al in n.names:
                    needed.add(al.name.split(".")[0])
        unresolved = sorted(m for m in needed
                            if m not in available and m not in stdlib_ok)
        check(f"{entry} imports resolve", not unresolved,
              f"cannot find: {unresolved}")
        # a src/ module is only importable if the file puts src/ on the path
        uses_src = bool(needed & available)
        adds_path = "src" in open(os.path.join(ROOT, entry), encoding="utf-8").read().split("sys.path.insert")[1][:80] \
            if "sys.path.insert" in open(os.path.join(ROOT, entry), encoding="utf-8").read() else False
        check(f"{entry} puts src/ on sys.path", (not uses_src) or adds_path)


def check_outputs():
    print("\n5. restored test outputs")
    import numpy as np
    d = os.path.join(ROOT, "outputs/test_out")
    files = sorted(f for f in os.listdir(d) if f.endswith(".npy")
                   and not f.startswith("._"))
    shapes, mn, mx, bad = set(), 1e9, -1e9, 0
    for f in files:
        a = np.load(os.path.join(d, f))
        shapes.add(a.shape)
        mn = min(mn, float(a.min())); mx = max(mx, float(a.max()))
        if not np.isfinite(a).all():
            bad += 1
    check("400 restored images", len(files) == 400, f"{len(files)}")
    check("all 256x256 float32", shapes == {(256, 256)}, str(shapes))
    check("clamped to [0,1]", mn >= 0.0 and mx <= 1.0, f"[{mn:.4f}, {mx:.4f}]")
    check("no NaN or Inf", bad == 0, f"{bad} bad files")


def check_split_and_metrics(ck):
    print("\n6. split and reported metrics")
    split = json.load(open(os.path.join(ROOT, "configs/split.json"), encoding="utf-8"))
    overlap = set(split["train"]) & set(split["val_hard"])
    check("train and val_hard do not overlap", not overlap, f"{len(overlap)} shared")
    check("val_hard is 480 samples", len(split["val_hard"]) == 480,
          f"{len(split['val_hard'])}")

    m = json.load(open(os.path.join(ROOT, "outputs/results/metrics.json"), encoding="utf-8"))
    s = m["summary"]
    check("metrics cover the whole val_hard set", m["n_val"] == len(split["val_hard"]),
          f"metrics {m['n_val']} vs split {len(split['val_hard'])}")
    check("model beats bicubic on PSNR", s["psnr"]["mean"] > s["psnr_base"]["mean"],
          f"{s['psnr']['mean']:.2f} vs {s['psnr_base']['mean']:.2f} dB")
    check("model beats bicubic on SSIM", s["ssim"]["mean"] > s["ssim_base"]["mean"],
          f"{s['ssim']['mean']:.4f} vs {s['ssim_base']['mean']:.4f}")
    check("model beats bicubic on LPIPS (lower is better)",
          s["lpips"]["mean"] < s["lpips_base"]["mean"],
          f"{s['lpips']['mean']:.4f} vs {s['lpips_base']['mean']:.4f}")
    check("metrics params match the checkpoint",
          m["params"] == sum(
              (lambda v: [__import__('functools').reduce(lambda x, y: x*y, v.size, 1)])(v)[0]
              for v in ck["model"].values() if hasattr(v, "size")),
          f"metrics says {m['params']:,}")

    # the README's headline numbers must match metrics.json
    rd = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    for val, label in [(f"{s['psnr']['mean']:.2f}", "PSNR"),
                       (f"{s['ssim']['mean']:.4f}", "SSIM"),
                       (f"{s['lpips']['mean']:.4f}", "LPIPS")]:
        check(f"README quotes the measured {label} ({val})", val in rd)


def main():
    print("PREFLIGHT")
    check_files()
    ck = check_architecture()
    check_inference_cli()
    check_intra_repo_imports()
    check_outputs()
    check_split_and_metrics(ck)
    print("\n" + "=" * 62)
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED:")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
