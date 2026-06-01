"""Stage an HF-cached model snapshot into a standalone local directory.

Why: under heavy concurrent load the shared HF cache's resolution layer
(hf_hub_download / cached_files) intermittently throws LocalEntryNotFoundError
even though the files are present. Loading transformers from a plain directory
path bypasses that resolution layer entirely. This copies the snapshot
(dereferencing the blob symlinks) into a self-contained dir.

Verifies byte counts after copy. Idempotent: skips files already present with
matching size.
"""
import hashlib
import os
import shutil
import sys

CACHE = "/home/ubuntu/.cache/huggingface/hub"


def snapshot_dir(repo_dirname: str) -> str:
    base = os.path.join(CACHE, repo_dirname)
    ref = os.path.join(base, "refs", "main")
    rev = open(ref).read().strip()
    return os.path.join(base, "snapshots", rev)


def stage(repo_dirname: str, dest: str) -> bool:
    src = snapshot_dir(repo_dirname)
    os.makedirs(dest, exist_ok=True)
    ok = True
    files = sorted(os.listdir(src))
    for fn in files:
        sp = os.path.join(src, fn)
        if not os.path.isfile(sp):  # skip subdirs (none expected)
            continue
        real = os.path.realpath(sp)  # dereference blob symlink
        if not os.path.exists(real):
            print(f"  SRC MISSING (dangling): {fn}")
            ok = False
            continue
        dp = os.path.join(dest, fn)
        ssz = os.path.getsize(real)
        if os.path.exists(dp) and os.path.getsize(dp) == ssz:
            print(f"  skip (present): {fn} ({ssz/1e9:.2f}GB)" if ssz > 1e8 else f"  skip (present): {fn}")
            continue
        print(f"  copy: {fn} ({ssz/1e9:.2f}GB)" if ssz > 1e8 else f"  copy: {fn}")
        shutil.copyfile(real, dp)
        if os.path.getsize(dp) != ssz:
            print(f"  SIZE MISMATCH after copy: {fn} ({os.path.getsize(dp)} != {ssz})")
            ok = False
    # Final verify: every source file present in dest with matching size
    n_ok, n_bad = 0, 0
    for fn in files:
        sp = os.path.join(src, fn)
        if not os.path.isfile(sp):
            continue
        real = os.path.realpath(sp)
        dp = os.path.join(dest, fn)
        if os.path.exists(dp) and os.path.exists(real) and os.path.getsize(dp) == os.path.getsize(real):
            n_ok += 1
        else:
            n_bad += 1
            print(f"  VERIFY FAIL: {fn}")
    print(f"VERIFY: {n_ok} ok, {n_bad} bad")
    return ok and n_bad == 0


if __name__ == "__main__":
    repo_dirname = sys.argv[1]
    dest = sys.argv[2]
    print(f"Staging {repo_dirname} -> {dest}")
    good = stage(repo_dirname, dest)
    print("STAGE_OK" if good else "STAGE_FAILED")
