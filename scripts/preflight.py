"""Preflight for the GPU run: checks the machine, access and the whole pipeline on a small case.

  python scripts/preflight.py               # on the GPU machine, before scripts/gpu_run.sh
  python scripts/preflight.py --skip-gpu --skip-disk --skip-token   # development machine

Checks: Python and package versions, GPU and memory, free disk, HF_TOKEN access to all three
models (config + first weight shard), data files, then the full pipeline on 10 questions with
Qwen2.5-0.5B (forward on the 4 cyclic orders + perm, depth tables with the logit and a tiny
tuned lens, aggregate, split-half, reliability, patching, smoke, results archive and its
verification), in a temporary directory, timed against a 10-minute budget. Prints one line
per check and PREFLIGHT: PASS or exactly what failed. Never prints the token.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
MODELS = {"llama31_8b": "meta-llama/Llama-3.1-8B", "qwen25_7b": "Qwen/Qwen2.5-7B", "gemma2_9b": "google/gemma-2-9b"}
STRICT = {"torch", "transformers", "tokenizers", "accelerate"}       # must match the pinned version
DISK_PASS_GB, DISK_MIN_GB = 160, 110
GPU_PASS_GIB, GPU_MIN_GIB = 40, 24
PIPELINE_BUDGET_S = 600

results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    print(f"[{status:4}] {name}" + (f": {detail}" if detail else ""), flush=True)


def check_python():
    v = sys.version_info
    record("python", "PASS" if v >= (3, 10) else "FAIL", platform.python_version()
           + ("" if v >= (3, 10) else " (need >= 3.10)"))


def check_packages():
    pins = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)==([^\s#]+)", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    bad, warn = [], []
    for name, want in pins.items():
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            bad.append(f"{name} missing (want {want})")
            continue
        base = have.split("+")[0]
        if base != want:
            (bad if name in STRICT else warn).append(f"{name} {have} (pinned {want})")
    if bad:
        record("packages", "FAIL", "; ".join(bad) + " -> pip install -r requirements.txt")
    elif warn:
        record("packages", "WARN", "; ".join(warn))
    else:
        record("packages", "PASS", f"{len(pins)} pinned packages match")


def check_gpu(skip: bool):
    if skip:
        record("gpu", "SKIP", "--skip-gpu")
        return "cpu"
    try:
        import torch
    except ImportError:
        record("gpu", "FAIL", "torch not importable")
        return "cpu"
    if not torch.cuda.is_available():
        record("gpu", "FAIL", f"no CUDA device visible (torch {torch.__version__}, built for CUDA {torch.version.cuda})")
        return "cpu"
    p = torch.cuda.get_device_properties(0)
    gib = p.total_memory / 2 ** 30
    bf16 = torch.cuda.is_bf16_supported()
    detail = f"{p.name}, {gib:.0f} GiB, bf16={'yes' if bf16 else 'no'}, CUDA {torch.version.cuda}"
    if not bf16 or gib < GPU_MIN_GIB:
        record("gpu", "FAIL", detail + f" (need bf16 and >= {GPU_MIN_GIB} GiB)")
    elif gib < GPU_PASS_GIB:
        record("gpu", "WARN", detail + " (works; keep default batch sizes)")
    else:
        record("gpu", "PASS", detail)
    return "cuda"


def check_disk(skip: bool = False):
    if skip:
        record("disk", "SKIP", "--skip-disk")
        return
    paths = {"repo": ROOT, "HF cache": Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))}
    for label, p in paths.items():
        p.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(p).free / 1e9
        if free >= DISK_PASS_GB:
            record(f"disk ({label})", "PASS", f"{free:.0f} GB free at {p}")
        elif free >= DISK_MIN_GB:
            record(f"disk ({label})", "WARN", f"{free:.0f} GB free at {p}; run with KEEP_ACTS=0")
        else:
            record(f"disk ({label})", "FAIL", f"{free:.0f} GB free at {p}; need >= {DISK_MIN_GB} GB "
                                             f"(~{DISK_PASS_GB} GB to keep activations)")


def check_token(skip: bool):
    if skip:
        record("hf access", "SKIP", "--skip-token")
        return
    if not os.environ.get("HF_TOKEN"):
        record("hf access", "FAIL", "HF_TOKEN is not set (export HF_TOKEN=hf_...; see RUN_FOR_ARIJIT.md)")
        return
    from huggingface_hub import get_hf_file_metadata, hf_hub_url, list_repo_files
    for key, repo in MODELS.items():
        try:
            files = list_repo_files(repo)
            shard = sorted(f for f in files if f.endswith(".safetensors"))[0]
            for f in ("config.json", shard):
                get_hf_file_metadata(hf_hub_url(repo, f))
            record(f"hf access {key}", "PASS", f"{repo}: config + {shard}")
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0][:160]
            hint = " -> accept the licence on the model page with this account" if (
                "gated" in msg.lower() or "403" in msg or "401" in msg) else ""
            record(f"hf access {key}", "FAIL", f"{repo}: {type(e).__name__}: {msg}{hint}")


def check_data():
    need = ["data/processed/prompts.parquet", "data/processed/entities.parquet", "data/processed/freq.parquet",
            "data/processed/swap_pool.parquet", "data/processed/patch_sample.csv"]
    missing = [p for p in need if not (ROOT / p).is_file()]
    if missing:
        record("data files", "FAIL", f"missing {missing} (git pull / git lfs?)")
        return
    import pandas as pd
    cols = set(pd.read_parquet(ROOT / "data/processed/prompts.parquet").columns)
    lack = [c for c in ("prompt_cyc0", "prompt_cyc3", "gold_idx_cyc0", "prompt_permuted") if c not in cols]
    record("data files", "FAIL" if lack else "PASS", f"prompts.parquet lacks {lack}" if lack else f"{len(need)} files")


def run(cmd, env, log, timeout=900):
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    log.write(f"$ {' '.join(cmd)}\n{r.stdout}\n{r.stderr}\n")
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
        raise RuntimeError(f"`{' '.join(cmd[2:5])} ...` exited {r.returncode}: {' | '.join(tail)}")
    return r.stdout, time.time() - t0


def check_pipeline(device: str, skip: bool):
    if skip:
        record("pipeline (Qwen2.5-0.5B, 10 q)", "SKIP", "--skip-pipeline")
        return
    m = "qwen25_05b_proxy"
    tmp = Path(tempfile.mkdtemp(prefix="crystal_preflight_"))
    env = {**os.environ, "CRYSTAL_ACTS_DIR": str(tmp / "acts"), "PYTHONPATH": str(ROOT)}
    out, lenses = str(tmp / "results"), str(tmp / "lenses")
    log_path = tmp / "preflight.log"
    t0 = time.time()
    step = "start"
    try:
        with open(log_path, "w") as log:
            for v in ("cyc0", "cyc1", "cyc2", "cyc3", "perm"):
                step = f"forward {v}"
                run([PY, "-m", "run.forward", "--model", m, "--device", device, "--sample", "10", "--variant", v],
                    env, log)
                step = f"depth table {v}"
                run([PY, "-m", "analysis.depth", "table", "--model", m, "--variant", v, "--device", device,
                     "--allow-proxy", "--out", out], env, log)
            step = "tuned lens (3 steps)"
            run([PY, "-m", "run.tune_lens", "--model", m, "--device", device, "--steps", "3", "--batch", "2",
                 "--seq-len", "64", "--tokens-per-seq", "8", "--train-tokens", "3000", "--eval-tokens", "600",
                 "--warmup", "1", "--out", lenses], env, log)
            step = "depth table (tuned)"
            run([PY, "-m", "analysis.depth", "table", "--model", m, "--variant", "cyc0", "--device", device,
                 "--allow-proxy", "--readout", "tuned", "--lens", f"{lenses}/{m}.pt", "--out", out], env, log)
            step = "aggregate"
            run([PY, "-m", "analysis.depth", "aggregate", "--model", m, "--out", out], env, log)
            step = "split-half"
            run([PY, "-m", "analysis.depth", "split-half", "--model", m, "--out", out], env, log)
            step = "reliability orig vs perm"
            run([PY, "-m", "analysis.depth", "reliability", "--model", m, "--variants", "orig", "perm",
                 "--out", out], env, log)
            step = "patching (2 questions)"
            run([PY, "-m", "run.patch", "--model", m, "--device", device, "--limit", "2", "--allow-proxy",
                 "--out", out], env, log)
            step = "smoke + lens assertion"
            so, _ = run([PY, "scripts/smoke.py", "--model", m, "--device", device, "--variant", "cyc0",
                         "--sample", "10", "--limit", "10"], env, log)
            if "assert lens(last layer) == model output: OK" not in so:
                raise RuntimeError("smoke did not report the lens assertion as OK")
            step = "results archive + verification"
            arch = tmp / "results_preflight.tar.gz"
            run([PY, "scripts/package_results.py", "--out", str(arch)], env, log)
            run([PY, "results/ingest.py", str(arch), "--check"], env, log)
    except Exception as e:  # noqa: BLE001
        record("pipeline (Qwen2.5-0.5B, 10 q)", "FAIL", f"step '{step}': {e}  (log: {log_path})")
        return
    dt = time.time() - t0
    if dt > PIPELINE_BUDGET_S and device == "cuda":
        record("pipeline (Qwen2.5-0.5B, 10 q)", "FAIL", f"took {dt:.0f}s > {PIPELINE_BUDGET_S}s budget (log: {log_path})")
    else:
        note = "" if device == "cuda" else " on CPU"
        record("pipeline (Qwen2.5-0.5B, 10 q)", "PASS", f"{dt:.0f}s{note}, all steps OK (log: {log_path})")
    shutil.rmtree(tmp / "acts", ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--skip-token", action="store_true")
    ap.add_argument("--skip-pipeline", action="store_true")
    ap.add_argument("--skip-disk", action="store_true", help="development machines only")
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    print(f"crystal preflight, repo {ROOT}")
    check_python()
    check_packages()
    device = check_gpu(args.skip_gpu)
    check_disk(args.skip_disk)
    check_token(args.skip_token)
    check_data()
    check_pipeline(device, args.skip_pipeline)
    fails = [r for r in results if r[1] == "FAIL"]
    warns = [r for r in results if r[1] == "WARN"]
    skips = [r for r in results if r[1] == "SKIP"]
    print()
    if fails:
        print(f"PREFLIGHT: FAIL ({len(fails)} check{'s' if len(fails) > 1 else ''})")
        for name, _, detail in fails:
            print(f"  - {name}: {detail}")
        return 1
    extra = (f" with {len(warns)} warning(s)" if warns else "") + (f", {len(skips)} skipped" if skips else "")
    print(f"PREFLIGHT: PASS{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
