# Running the GPU stage (for Arijit)

Thanks for running this. The repo measures, layer by layer, where a model's answer to a
SANSKRITI question becomes fixed (logit lens and tuned lens, plus activation patching), for
Llama-3.1-8B, Qwen2.5-7B and gemma-2-9b. Data preparation, frequency scores and all analysis
code are done and in the repo; what needs a GPU is one script, `scripts/gpu_run.sh`. It is
resumable and writes everything into one archive you send back. You do not need push access.

## 1. Requirements

| | needed | notes |
|---|---|---|
| GPU | one NVIDIA GPU with bf16 (Ampere or newer), **>= 24 GiB**; 40 or 80 GB A100 recommended | all three models run in bf16 (16, 15, 18.5 GB weights) |
| Disk | **~160 GB** free (or **~110 GB** with `KEEP_ACTS=0`) | weights 68 GB (Gemma is published in fp32, 37 GB) + activations ~27/20/30 GB per model; the Hugging Face cache counts too (set `HF_HOME` to move it) |
| CUDA | NVIDIA driver supporting **CUDA 13.0** for the default `torch==2.14.1` wheel | `nvidia-smi` shows "CUDA Version". If it is older, see section 7 |
| Python | 3.10 to 3.13 (developed on 3.13) | |
| Network | huggingface.co (model weights, WikiText for the tuned lens) | nothing else |
| Time | ~6 to 8 h on a 40 GB A100, ~4 to 6 h on 80 GB (estimates) | run inside `tmux` or `screen` |

## 2. Hugging Face token (your own)

1. Log in (or sign up) at https://huggingface.co.
2. Open both model pages and accept the licence with that account:
   - https://huggingface.co/meta-llama/Llama-3.1-8B (a short form; approval can take from
     minutes to a few hours, and you get an email)
   - https://huggingface.co/google/gemma-2-9b (accept on the page)
   Qwen2.5-7B is not gated.
3. Create a token at https://huggingface.co/settings/tokens: type "Read" is enough (or a
   fine-grained token with "Read access to contents of all public gated repos you can access").
4. Put it in your shell only, never in a file in the repo:
   ```
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
   ```
   The scripts read it from the environment and never print or save it.

## 3. Get the code and set up

```
git clone -b claude/research-codebase-setup-f0mhpj https://github.com/PARTHANAND0009/SANSKRITI.git
cd SANSKRITI
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

(If the clone asks for access, Parth will add you to the repository or send a zip.)

## 4. Preflight (about 5 minutes)

```
export HF_TOKEN=hf_...
.venv/bin/python scripts/preflight.py
```

It checks the Python and package versions, the GPU and its memory, free disk (repo and
Hugging Face cache), that your token can read all three models, and the data files. Then it
runs the whole pipeline on 10 questions with Qwen2.5-0.5B (all option orders, depth tables,
a tiny tuned lens, patching, the lens-equals-model-output assertion, and the results archive),
which should take well under 10 minutes on a GPU. It ends with `PREFLIGHT: PASS` or a list of
exactly what failed. Please only start the full run after a PASS (a WARN is fine).

## 5. Full run

```
tmux new -s crystal
export HF_TOKEN=hf_...
ARCHIVE=1 SKIP_SETUP=1 bash scripts/gpu_run.sh 2>&1 | tee run.log
```

- `ARCHIVE=1`: no git at all. After each model it writes `results_<date>.tar.gz` in the repo
  root (tables, logs, tuned-lens evaluation reports, environment info; a `MANIFEST.json` with
  SHA-256 checksums). Raw activations and tuned-lens weights are left out (too large).
- `SKIP_SETUP=1`: reuse the `.venv` you made in step 3.
- Low on disk: add `KEEP_ACTS=0` (deletes each model's activations once its tables are written).
- Out of GPU memory: add `BATCH=8` (or `BATCH=4`).

What it does per model, in order: download the weights; forward pass on all 19,742 questions
in 5 option orders; train a tuned lens (on WikiText-103, ~10 to 15 min); depth tables for the
logit and tuned lens; reliability; activation patching on 1,200 questions; a smoke check.
Then the primary-metric decision and statistics over all models.

## 6. If it stops

Run the same command again. Every step skips what is already on disk: forward passes resume
from the last finished shard of 256 questions, the tuned lens and depth tables are skipped if
present, patching resumes by shard of 50 questions, downloads resume from the Hugging Face
cache. `results/gpu_run_log.tsv` records each finished step with its duration.

To run one model at a time: `MODELS="qwen25_7b" ARCHIVE=1 SKIP_SETUP=1 bash scripts/gpu_run.sh`.

## 7. Troubleshooting

| symptom | fix |
|---|---|
| preflight: `hf access ... 403` / gated | accept the licence on that model's page with the same account as the token; for Llama wait for the approval email |
| preflight: packages FAIL for torch | the pinned version is `torch==2.14.1`; for an older driver install the same version for your CUDA, e.g. `.venv/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126` |
| `CUDA out of memory` | rerun with `BATCH=8` (or 4); it resumes |
| `No space left on device` | free space or rerun with `KEEP_ACTS=0`; `HF_HOME=/bigdisk/hf` moves the model cache |
| slow downloads | `.venv/bin/pip install hf_transfer` then `HF_HUB_ENABLE_HF_TRANSFER=1` |
| anything else | send `run.log` and `results/gpu_run_log.tsv` |

## 8. How to check it worked

- `run.log` ends with `[...] done`.
- `results/gpu_run_log.tsv` has a line for every step of every model and `all stats`.
- `results/smoke_<model>.txt` for each model contains `assert lens(last layer) == model output: OK`.
- `results/primary_metric.json` exists.
- Verify the archive: `.venv/bin/python results/ingest.py results_<date>.tar.gz --check`
  should print `archive OK`.

## 9. Sending the results back

Send `results_<date>.tar.gz` (expected a few hundred MB) by Google Drive or any file
transfer, and paste the `sha256` line that `scripts/package_results.py` printed at the end of
`run.log`. That is all; Parth ingests it with `results/ingest.py`, which checks every file
against the manifest. The archive contains no token: only tables, logs, `pip freeze` and the
`nvidia-smi` GPU line.
