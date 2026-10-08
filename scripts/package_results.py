"""Package GPU-run outputs into one archive with a manifest and checksums.

For running without git push access: everything the analysis needs goes into
results_<YYYYMMDD>.tar.gz at the repo root, with MANIFEST.json (git commit, environment,
and the SHA-256 and size of every file). Raw activations (acts/) and tuned-lens weights
(lenses/*.pt) are left out by default because they are large; include them with
--include-acts / --include-lens-weights. results/ingest.py verifies and unpacks it.

  python scripts/package_results.py                 # results_<today>.tar.gz
  python scripts/package_results.py --out results_lab.tar.gz
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["results", "lenses", "data/processed/patch_sample.csv", "data/processed/patch_sample_cells.csv",
           "data/processed/patch_sample_states.csv", "config", "ANALYSIS_PLAN.md"]
EXCLUDE_PARTS = ("__pycache__",)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=ROOT).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {type(e).__name__}"


def collect(include_acts: bool, include_lens_weights: bool) -> list[Path]:
    roots = list(INCLUDE) + (["acts"] if include_acts else [])
    files = []
    for r in roots:
        p = ROOT / r
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files += [f for f in sorted(p.rglob("*")) if f.is_file()]
    out = []
    for f in files:
        rel = f.relative_to(ROOT)
        if any(part in EXCLUDE_PARTS for part in rel.parts):
            continue
        if rel.parts[0] == "lenses" and f.suffix == ".pt" and not include_lens_weights:
            continue
        if rel.name.startswith("results_") and rel.name.endswith(".tar.gz"):
            continue
        out.append(f)
    return out


def environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "git_commit": _run(["git", "rev-parse", "HEAD"]), "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
           "git_dirty": bool(_run(["git", "status", "--porcelain", "--untracked-files=no"])),
           "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])}
    try:
        import torch
        import transformers
        env.update(torch=torch.__version__, cuda=torch.version.cuda, transformers=transformers.__version__)
    except ImportError:
        pass
    return env


def build(out: Path, include_acts=False, include_lens_weights=False) -> dict:
    files = collect(include_acts, include_lens_weights)
    manifest = {"created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "environment": environment(), "include_acts": include_acts,
                "include_lens_weights": include_lens_weights,
                "files": [{"path": str(f.relative_to(ROOT)), "bytes": f.stat().st_size, "sha256": sha256(f)}
                          for f in files]}
    tmp = out.with_suffix(".tmp")
    with tarfile.open(tmp, "w:gz") as tar:
        data = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(data)
        import io
        tar.addfile(info, io.BytesIO(data))
        for f in files:
            tar.add(f, arcname=str(f.relative_to(ROOT)))
    tmp.replace(out)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--include-acts", action="store_true")
    ap.add_argument("--include-lens-weights", action="store_true")
    args = ap.parse_args(argv)
    out = ROOT / (args.out or f"results_{dt.date.today():%Y%m%d}.tar.gz")
    m = build(out, args.include_acts, args.include_lens_weights)
    total = sum(f["bytes"] for f in m["files"])
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB compressed, {len(m['files'])} files, "
          f"{total / 1e6:.1f} MB uncompressed), commit {m['environment']['git_commit'][:10]}")
    print(f"sha256 of the archive: {sha256(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
