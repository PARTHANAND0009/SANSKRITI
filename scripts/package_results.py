"""Package GPU-run outputs into results_<YYYYMMDD>.tar.gz with a manifest and checksums.

MANIFEST.json records the git commit, the environment, and the size and SHA-256 of every
file. Activations and tuned-lens weights are left out unless --include-acts or
--include-lens-weights is given. results/ingest.py verifies and unpacks the archive.

  python scripts/package_results.py [--out results_lab.tar.gz]
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import logging
import platform
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crystal.io import sha256_file  # noqa: E402
from crystal.log import setup_logging  # noqa: E402

log = logging.getLogger(__name__)
INCLUDE = [
    "results",
    "lenses",
    "data/processed/patch_sample.csv",
    "data/processed/patch_sample_cells.csv",
    "data/processed/patch_sample_states.csv",
    "config",
    "ANALYSIS_PLAN.md",
]
EXCLUDE_PARTS = ("__pycache__",)


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=ROOT).stdout.strip()
    except Exception as e:  # noqa: BLE001  (e.g. no nvidia-smi or no git: recorded, not fatal)
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
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain", "--untracked-files=no"])),
        "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
    }
    try:
        import torch
        import transformers

        env.update(torch=torch.__version__, cuda=torch.version.cuda, transformers=transformers.__version__)
    except ImportError:
        pass
    return env


def build(out: Path, include_acts=False, include_lens_weights=False) -> dict:
    files = collect(include_acts, include_lens_weights)
    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": environment(),
        "include_acts": include_acts,
        "include_lens_weights": include_lens_weights,
        "files": [
            {"path": str(f.relative_to(ROOT)), "bytes": f.stat().st_size, "sha256": sha256_file(f)} for f in files
        ],
    }
    tmp = out.with_suffix(".tmp")
    with tarfile.open(tmp, "w:gz") as tar:
        data = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        for f in files:
            tar.add(f, arcname=str(f.relative_to(ROOT)))
    tmp.replace(out)
    return manifest


def main(argv=None):
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--include-acts", action="store_true")
    ap.add_argument("--include-lens-weights", action="store_true")
    args = ap.parse_args(argv)
    out = ROOT / (args.out or f"results_{dt.date.today():%Y%m%d}.tar.gz")
    m = build(out, args.include_acts, args.include_lens_weights)
    total = sum(f["bytes"] for f in m["files"])
    log.info(
        f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB compressed, {len(m['files'])} files, "
        f"{total / 1e6:.1f} MB uncompressed), commit {m['environment']['git_commit'][:10]}"
    )
    log.info(f"sha256 of the archive: {sha256_file(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
