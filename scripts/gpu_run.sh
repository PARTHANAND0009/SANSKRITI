#!/usr/bin/env bash
# One clean GPU run of stages 2-4 for the three models. Resumable: rerun the same command
# after any interruption; every step skips work that is already on disk.
#
#   export HF_TOKEN=...            # read from the environment only; never written anywhere
#   bash scripts/gpu_run.sh                      # all three models
#   MODELS="qwen25_7b" bash scripts/gpu_run.sh   # a subset
#
# Options (environment):
#   MODELS        default "llama31_8b qwen25_7b gemma2_9b"
#   VARIANTS      default "cyc0 cyc1 cyc2 cyc3 perm"
#   BATCH         forward batch size (default: config/models.yaml)
#   KEEP_ACTS=0   delete a model's activations after its depth tables are written
#   PUSH=0        do not git push results/ after each model
#   ARCHIVE=1     no git at all: after each model write results_<date>.tar.gz (manifest +
#                 checksums; no raw activations, no lens weights) for sending back by hand
#   SKIP_SETUP=1  reuse the existing .venv
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=${MODELS:-"llama31_8b qwen25_7b gemma2_9b"}
VARIANTS=${VARIANTS:-"cyc0 cyc1 cyc2 cyc3 perm"}
KEEP_ACTS=${KEEP_ACTS:-1}
ARCHIVE=${ARCHIVE:-0}
PUSH=${PUSH:-1}
[[ "$ARCHIVE" == 1 ]] && PUSH=0
ARCHIVE_NAME=${ARCHIVE_NAME:-results_$(date -u +%Y%m%d).tar.gz}
PY=.venv/bin/python
LOG=results/gpu_run_log.tsv
mkdir -p results lenses

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set (needed for meta-llama/Llama-3.1-8B and google/gemma-2-9b)" >&2
  exit 1
fi
export HF_TOKEN
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
step() {  # step <model> <name> <command...>: run, time and log one step
  local m=$1 name=$2; shift 2
  local t0=$(date +%s)
  echo "[$(stamp)] $m :: $name"
  "$@"
  printf '%s\t%s\t%s\t%s\n' "$(stamp)" "$m" "$name" "$(( $(date +%s) - t0 ))" >> "$LOG"
}

# Environment
if [[ "${SKIP_SETUP:-0}" != 1 ]]; then
  python3 -m venv .venv
  $PY -m pip install -q --upgrade pip
  $PY -m pip install -q -r requirements.txt
fi
$PY - <<'EOF'
import torch
assert torch.cuda.is_available(), "no CUDA device"
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}, {p.total_memory / 2**30:.0f} GiB")
EOF
[[ -s $LOG ]] || printf 'time\tmodel\tstep\tseconds\n' > "$LOG"
$PY -m pip freeze > results/env_pip_freeze.txt

# Data: tracked in git, rebuilt only if missing
[[ -f data/processed/prompts.parquet ]] || step all prep $PY -m data.prep
[[ -f data/processed/patch_sample.csv ]] || step all patch_sample $PY -m run.patch --build-sample
step all tokens $PY -m crystal.tokens

push_results() {
  if [[ "$ARCHIVE" == 1 ]]; then
    $PY scripts/package_results.py --out "$ARCHIVE_NAME"
    return 0
  fi
  [[ "$PUSH" == 1 ]] || return 0
  git add results/ lenses/*.eval.json 2>/dev/null || true
  git commit -q -m "GPU run: $1 results" || return 0
  git push -q origin HEAD || echo "git push failed; results are committed locally" >&2
}

# Per model
for m in $MODELS; do
  BATCH_ARG=()
  [[ -n "${BATCH:-}" ]] && BATCH_ARG=(--batch-size "$BATCH")

  step "$m" download $PY - "$m" <<'EOF'
import sys
from huggingface_hub import snapshot_download
from crystal.io import model_config
snapshot_download(model_config(sys.argv[1])["id"], allow_patterns=["*.json", "*.safetensors", "*.model"])
EOF

  for v in $VARIANTS; do
    step "$m" "forward_$v" $PY -m run.forward --model "$m" --device cuda --split analysis --variant "$v" "${BATCH_ARG[@]}"
  done

  [[ -f lenses/$m.pt ]] || step "$m" tune_lens $PY -m run.tune_lens --model "$m" --device cuda

  for readout in logit tuned; do
    for v in $VARIANTS; do
      [[ -f results/depth_${m}_${v}_${readout}.parquet ]] || \
        step "$m" "depth_${readout}_$v" $PY -m analysis.depth table --model "$m" --variant "$v" --device cuda --readout "$readout"
    done
    step "$m" "aggregate_$readout" $PY -m analysis.depth aggregate --model "$m" --variants cyc0 cyc1 cyc2 cyc3 --readout "$readout"
    step "$m" "splithalf_$readout" $PY -m analysis.depth split-half --model "$m" --readout "$readout"
    # orig is cyc{gold_idx}; orig vs perm repeats the pilot's test-retest design, and both
    # orders have balanced gold letters, so their pooled letter prior is valid
    step "$m" "reliability_$readout" $PY -m analysis.depth reliability --model "$m" --variants orig perm --readout "$readout"
  done

  step "$m" patch $PY -m run.patch --model "$m" --device cuda
  step "$m" smoke bash -c "$PY scripts/smoke.py --model $m --device cuda --variant cyc0 --limit 50 2>&1 | tee results/smoke_$m.txt; test \${PIPESTATUS[0]} -eq 0"

  [[ "$KEEP_ACTS" == 1 ]] || rm -rf "acts/$m"
  push_results "$m"
done

# Across models; output kept apart from the CPU pilot's results/prelim
step all primary_metric $PY -m analysis.depth choose-primary --models $MODELS
step all stats $PY -m analysis.stats --models $MODELS --agg cyc --out results/gpu_stats
push_results "stats"
echo "[$(stamp)] done"
