"""Ingest a results archive made by scripts/package_results.py (e.g. from a collaborator's
GPU run): verify every checksum, then put each file at its path in this repo.

  python results/ingest.py results_20261020.tar.gz            # verify + install
  python results/ingest.py results_20261020.tar.gz --check    # verify only
  python results/ingest.py results_20261020.tar.gz --force    # overwrite differing files

Refuses to install if any checksum fails, if a path would leave the repo, or (without
--force) if an existing file differs. Prints what was installed and the next commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TOP = {"results", "lenses", "data", "config", "acts", "ANALYSIS_PLAN.md"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_member(name: str) -> bool:
    p = PurePosixPath(name)
    return not p.is_absolute() and ".." not in p.parts and (p.parts[0] in ALLOWED_TOP or name == "MANIFEST.json")


def verify(archive: Path, workdir: Path) -> dict:
    with tarfile.open(archive, "r:gz") as tar:
        bad = [m.name for m in tar.getmembers() if not safe_member(m.name) or not (m.isfile() or m.isdir())]
        if bad:
            raise SystemExit(f"refusing archive: unexpected members {bad[:5]}")
        tar.extractall(workdir, filter="data")
    manifest = json.loads((workdir / "MANIFEST.json").read_text())
    errors = []
    listed = set()
    for f in manifest["files"]:
        p = workdir / f["path"]
        listed.add(f["path"])
        if not p.is_file():
            errors.append(f"missing: {f['path']}")
        elif p.stat().st_size != f["bytes"] or sha256(p) != f["sha256"]:
            errors.append(f"checksum mismatch: {f['path']}")
    extra = [str(p.relative_to(workdir)) for p in workdir.rglob("*")
             if p.is_file() and p.name != "MANIFEST.json" and str(p.relative_to(workdir)) not in listed]
    if extra:
        errors.append(f"files not in manifest: {extra[:5]}")
    if errors:
        raise SystemExit("verification FAILED:\n  " + "\n  ".join(errors))
    return manifest


def install(manifest: dict, workdir: Path, force: bool) -> tuple[list, list, list]:
    new, same, conflict = [], [], []
    for f in manifest["files"]:
        dst = ROOT / f["path"]
        if dst.exists():
            (same if sha256(dst) == f["sha256"] else conflict).append(f["path"])
        else:
            new.append(f["path"])
    if conflict and not force:
        raise SystemExit("existing files differ (use --force to overwrite):\n  " + "\n  ".join(conflict[:20]))
    for path in new + conflict:
        dst = ROOT / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(workdir / path, dst)
    return new, same, conflict


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("archive")
    ap.add_argument("--check", action="store_true", help="verify only")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    archive = Path(args.archive)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        m = verify(archive, work)
        env = m["environment"]
        print(f"archive OK: {len(m['files'])} files, all checksums match. Created {m['created_utc']} "
              f"from commit {env.get('git_commit', '?')[:10]} (dirty={env.get('git_dirty')}), "
              f"GPU: {env.get('nvidia_smi')}, torch {env.get('torch')}, transformers {env.get('transformers')}")
        if args.check:
            return 0
        new, same, conflict = install(m, work, args.force)
    print(f"installed: {len(new)} new, {len(conflict)} overwritten, {len(same)} already identical")
    pat = re.compile(r"^results/depth_(.+)_cyc_logit_agg\.parquet$")
    models = sorted({mm.group(1) for f in m["files"] if (mm := pat.match(f["path"]))})
    if models:
        print("next:\n  python -m analysis.stats --models " + " ".join(models) +
              " --agg cyc --out results/gpu_stats\n  python analysis/figures.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
