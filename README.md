# crystal

At which layer does a model's answer to an Indian cultural multiple-choice question
settle, and does that depth depend on how well documented the entity and state are?
A collaboration with Prof. Sriparna Saha's lab, IIT Patna (authors of SANSKRITI,
arXiv:2506.15355).

- **Depth.** Readout layers are 0..L: 0 is the embedding output, k is the output of block
  k. For each question we read the A-D restricted softmax at the final prompt token at
  every layer and record where the gold answer settles (ℓ\*, d = ℓ\*/L; continuous
  variants in `analysis/depth.py`).
- **Readouts.** Logit lens and tuned lens (Belrose et al. 2023), with activation patching
  as a causal check. Hugging Face transformers with PyTorch hooks.
- **Models** (base, bf16): `meta-llama/Llama-3.1-8B` (L=32), `Qwen/Qwen2.5-7B` (28),
  `google/gemma-2-9b` (42). `Qwen/Qwen2.5-0.5B` is a CPU proxy for tests and is never
  used in analysis.
- **Analysis plan:** `ANALYSIS_PLAN.md` (pre-registered). **Status:** `STATUS.md`.
  **GPU run:** `RUN_FOR_ARIJIT.md`.

## Layout

```
config/models.yaml     model registry: id, n_layers, dtype, batch_size, option_token_variant
config/run.yaml        seed, paths, API settings, analysis constants
crystal/               shared code: config and data loaders, lens readout, model loading,
                       residual hooks, option tokens, logging
data/prep.py           stage 0: prompts.parquet, data_quality.md
data/entities.py       stage 1a: entity extraction, swap pool
data/frequency.py      stage 1b: infini-gram counts, Wikipedia, pageviews (cached in data/cache/)
run/forward.py         stage 2: residual stream at the final token, every layer
run/tune_lens.py       stage 3a: tuned lens
run/patch.py           stage 3b: activation patching
analysis/depth.py      stage 4a: depth metrics, aggregation over option orders, reliability
analysis/stats.py      stage 4b: depth vs frequency
analysis/figures.py    paper figures and tables from results/
scripts/               GPU run, preflight, smoke test, results archive, bf16 conversion
paper/                 ACL-format draft
```

## Pipeline

```
make prep          # stage 0  -> data/processed/prompts.parquet, data_quality.md
make tokens        # option-token check -> config/models.yaml
make entities      # stage 1a -> entities.parquet, swap_pool.parquet, entity_audit.csv
make frequency     # stage 1b -> freq.parquet
make smoke         # stage 2 on the proxy (CPU, 50 questions) + lens checks
make test
make lint          # ruff check + ruff format --check
```

The GPU stages for all three models run from one resumable script:

```
export HF_TOKEN=...                       # environment only; Llama and Gemma are gated
python scripts/preflight.py               # checks the machine, then the pipeline on 10 questions
ARCHIVE=1 bash scripts/gpu_run.sh         # rerun the same command after an interruption
```

Options: `MODELS="qwen25_7b"` (one model), `KEEP_ACTS=0` (delete activations once tables
are written), `BATCH=8` (less GPU memory). Estimated ~6-8 h on a 40 GB A100, ~4-6 h on
80 GB; ~160 GB disk (~110 GB with `KEEP_ACTS=0`). Archives are checked and installed with
`python results/ingest.py results_<date>.tar.gz`.

Single steps:

```
python -m run.forward --model llama31_8b --device cuda --variant cyc0
python -m run.tune_lens --model llama31_8b --device cuda
python -m analysis.depth table --model llama31_8b --variant cyc0 --device cuda --readout tuned
python -m analysis.depth aggregate --model llama31_8b
python -m analysis.depth split-half --model llama31_8b
python -m run.patch --model llama31_8b --device cuda
python -m analysis.stats --models llama31_8b qwen25_7b gemma2_9b --agg cyc --out results/gpu_stats
python analysis/figures.py
```

Activations go to `acts/{model}/{variant}/shard_XXXXX.npz` (`acts` fp16 [n, L+1, d],
`qids`, `out_opt_logits` fp32 [n, 4]). Existing shards with the expected qids are skipped.

## Conventions

- Prompt template:
  ```
  Question: {stem}
  A. {opt0}
  B. {opt1}
  C. {opt2}
  D. {opt3}
  Answer:
  ```
- Option orders: `cyc0..cyc3` rotate the options so the gold answer is at A, B, C, D once
  each (the main design); `perm` is a per-question seeded shuffle (robustness).
- `load_analysis_set()` drops `ambiguous_gold` and `leaks_answer` rows (19,742 remain).
- Seed: `seed` in `config/run.yaml`, applied with `crystal.io.set_seed`.

## Sources and versions

- Dataset: HF `13ari/Sanskriti` (CC0), revision `8d8523795491a1484834d5cb85ed898b04839077`,
  21,853 rows.
- Corpus counts: infini-gram API, index `v4_dolma-v1_7_llama` (Dolma v1.7, Llama-2
  tokenizer); exact, case-sensitive token-sequence matches.
- Wikipedia: MediaWiki API and Wikimedia pageviews REST API, 2024 totals, `user` agent.
- Python 3.13; pinned packages in `requirements.txt`, dev tools in `requirements-dev.txt`.
