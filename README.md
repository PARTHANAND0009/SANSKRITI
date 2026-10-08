# crystal

Where does a model's answer to an Indian cultural MCQ crystallize, and does that
depth depend on how well documented the entity and state are? Collaboration with
Prof. Sriparna Saha's lab, IIT Patna (authors of SANSKRITI, arXiv:2506.15355).

- **Depth ℓ\***: first readout layer at which the correct option is top-1 under a
  softmax restricted to the option tokens A–D at the final prompt token, and stays
  top-1 through the last layer. Readout layers are 0..L: **0 = embedding output**,
  k = output of the k-th block (1-indexed blocks), so **d = ℓ\*/L**.
- **Readouts**: logit lens and tuned lens (Belrose et al. 2023), with activation
  patching as a causal check. HuggingFace transformers + PyTorch forward hooks; no
  TransformerLens.
- **Models** (base): `meta-llama/Llama-3.1-8B` (32 layers), `Qwen/Qwen2.5-7B` (28),
  `google/gemma-2-9b` (42). `Qwen/Qwen2.5-0.5B` is a CPU proxy for tests
  (`proxy: true`); it is never used in analysis.

## Layout

```
config/models.yaml     model registry: id, n_layers, dtype, batch_size, option_token_variant, proxy
config/run.yaml        seeds, paths, sample sizes, API settings
crystal/io.py          config + data loaders (load_analysis_set, load_acts, ...)
crystal/lens.py        lens_logits(model, resid, layer) and restricted-softmax helpers
crystal/tokens.py      option-token check; writes option_token_variant
data/prep.py           stage 0: prompts.parquet, data_quality.md
data/entities.py       stage 1a: deterministic entity extraction + swap pool
data/frequency.py      stage 1b: Wikipedia / pageviews / infini-gram scores (cached in data/cache/)
run/forward.py         stage 2: resid_post at the final token for every layer
run/lens.py, run/patch.py, analysis/depth.py, analysis/stats.py    stubs (stages 3-4)
scripts/smoke.py       proxy smoke test on stage 2 output
tests/                 CPU tests, no big-model downloads
```

## Processed data (tracked)

| file | content |
|---|---|
| `data/processed/prompts.parquet` | stage 0 rows (+ `ambiguous_gold`, `duplicate_options`, per-model option ids as JSON) |
| `data/processed/data_quality.md` | dropped / ambiguous / duplicate-option rows, for the dataset authors |
| `data/processed/entities.parquet` | entity, source, stem span, rule, confidence, `entity_mentions_state` |
| `data/processed/templates.md` | stem templates per question_type with coverage and examples |
| `data/processed/entity_audit.csv` | 100 random rows for hand checking |
| `data/processed/swap_pool.parquet` | per attribute: entities with a stem span and their state |
| `data/processed/freq*.parquet` | stage 1b scores per question / entity / state |

## Pipeline

```
make prep          # stage 0  -> data/processed/prompts.parquet, data_quality.md
make tokens        # option-token check -> config/models.yaml
make entities      # stage 1a -> data/processed/entities.parquet, entity_audit.csv
make frequency     # stage 1b -> data/processed/freq.parquet
make smoke         # stage 2 on the proxy (CPU, 50 questions) + smoke checks
make test
```

## GPU run (stages 2-4, all three models)

One script does everything, resumably, and pushes `results/` after each model:

```
git clone -b claude/research-codebase-setup-f0mhpj <repo> && cd SANSKRITI
export HF_TOKEN=...                 # read from the environment only (Llama and Gemma are gated)
bash scripts/gpu_run.sh             # rerun the same command after any interruption
MODELS="qwen25_7b" bash scripts/gpu_run.sh      # one model
KEEP_ACTS=0 bash scripts/gpu_run.sh             # delete each model's activations when done
```

Per model: download weights; `run.forward` on the full analysis set (19,742 questions) for
the four cyclic option orders `cyc0..cyc3` and the random permutation `perm`; `run.tune_lens`;
depth tables for the logit and tuned lens; the 4-rotation aggregate; split-half and original-vs-perm
reliability (the original order is cyc{gold_idx}); `run.patch` on the stratified sample (1,248 questions); smoke. Then
`analysis.stats --agg cyc`. Step durations are logged to `results/gpu_run_log.tsv`.

Estimates (not measured on a GPU; derived from FLOP counts and the CPU pilot; the log will
give real numbers after the first model):

| | A100 40 GB | A100 80 GB |
|---|---|---|
| forward, 5 variants x 19,742 questions (~10 M tokens) | ~1 h / model (Gemma ~1.5 h) | ~0.7 h / model (Gemma ~1 h) |
| tuned lens (250 steps, 8 x 512 tokens) + eval | ~15 min / model | ~10 min / model |
| depth tables (2 readouts x 5 variants) | ~20 min / model | ~15 min / model |
| patching (1,248 questions, L+1 patched runs each) | ~15 min / model | ~10 min / model |
| **total, three models** | **~6-8 h** | **~4-6 h** |

Both cards hold every model in bf16 (16, 15, 18.5 GB); on 40 GB keep the default batch
sizes (16 / 16 / 8), on 80 GB `BATCH=32` is safe for Llama and Qwen.
Disk: weights 68 GB (Gemma ships fp32, 37 GB; cast to bf16 at load), activations fp16
~27 / 20 / 30 GB per model for 5 variants, env ~10 GB: **~160 GB with KEEP_ACTS=1, ~110 GB
with KEEP_ACTS=0** (peak = all weights + one model's activations).

Single steps:

```
python -m run.forward --model llama31_8b --device cuda --split analysis --variant cyc0
python -m run.tune_lens --model llama31_8b --device cuda
python -m analysis.depth table --model llama31_8b --variant cyc0 --device cuda --readout tuned
python -m analysis.depth aggregate --model llama31_8b --variants cyc0 cyc1 cyc2 cyc3
python -m analysis.depth split-half --model llama31_8b
python -m run.patch --model llama31_8b --device cuda
python -m analysis.stats --models llama31_8b qwen25_7b gemma2_9b --agg cyc --out results/gpu_stats
```

Shards land in `acts/{model}/{variant}/shard_XXXXX.npz` (`acts` fp16 [n, L+1, d], layer 0 = embeddings,
`qids`, `out_opt_logits` fp32 [n, 4] from the model's own output). Re-running skips
shards that already exist with the expected qids, so an interrupted run resumes.

## Conventions

- Prompt template (exact):
  ```
  Question: {stem}
  A. {opt0}
  B. {opt1}
  C. {opt2}
  D. {opt3}
  Answer:
  ```
- `prompt_permuted`: options shuffled with a per-question seed derived from
  `sha256(qid)` and `prep.permute_seed`.
- `ambiguous_gold` rows stay in `prompts.parquet`; `load_analysis_set()` drops them.
- Seed: `seed` in `config/run.yaml`, applied via `crystal.io.set_seed`.

## Versions

Python 3.13.16, CPU only during setup. See `requirements.txt` (torch 2.14.1,
transformers 5.18.0, tokenizers 0.23.2, datasets 5.1.0, pandas 3.0.6,
pyarrow 25.0.1, numpy 2.5.3, scipy 1.18.1).

External sources:
- Dataset: HF `13ari/Sanskriti` (CC0), revision `8d8523795491a1484834d5cb85ed898b04839077`
  (file `Merged_Dataset_english_SANSKRITI.csv`, 21,853 rows, single `train` split).
- infini-gram: `https://api.infini-gram.io/`, index `v4_dolma-v1_7_llama` (Dolma v1.7,
  Llama-2 tokenizer). A count query returns `{"approx", "count", "latency", "token_ids",
  "tokens"}`; the query string is tokenized with a leading-space prefix (SentencePiece),
  and counts are case-sensitive exact token-sequence matches.
- Wikipedia: MediaWiki action API (en, hi) and Wikimedia pageviews REST API,
  2024 totals, `user` agent type.

## Status

See `STATUS.md`.
