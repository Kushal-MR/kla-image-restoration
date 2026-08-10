#!/usr/bin/env python3
"""
Zip this repository into a Kaggle Dataset bundle.

Kaggle cannot read from GitHub directly, so we upload the repo as a private
Dataset. Doing it this way rather than maintaining a separate copy of the
code means the notebook runs exactly what ships in the repo -- one source of
truth, no drift between "the version that trained" and "the version we
submitted".

    python scripts/make_kaggle_bundle.py            -> kla-repo.zip
    python scripts/make_kaggle_bundle.py -o out.zip

Upload the result to Kaggle as a Dataset named `kla-repo`. To update it later,
use *New Version* on the same dataset rather than creating another one.
"""

import argparse
import os
import zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "data", "outputs", ".venv"}
SKIP_EXT = {".pt", ".pth", ".onnx", ".zip", ".npy"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=os.path.join(HERE, "kla-repo.zip"))
    args = ap.parse_args()

    n = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(HERE):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                if os.path.splitext(f)[1] in SKIP_EXT or f.startswith("._"):
                    continue
                full = os.path.join(root, f)
                if os.path.abspath(full) == os.path.abspath(args.out):
                    continue
                z.write(full, os.path.relpath(full, HERE))
                n += 1

    print(f"wrote {args.out} ({n} files, {os.path.getsize(args.out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
