# crystal

Where does a model's answer to an Indian cultural MCQ crystallize, and does that
depth depend on how well documented the entity and state are? Collaboration with
Prof. Sriparna Saha's lab, IIT Patna (authors of SANSKRITI, arXiv:2506.15355).

- **Depth ℓ\***: first decoder layer at which the correct option is top-1 under a
  softmax restricted to the option tokens A–D at the final prompt token, and stays
  top-1 through the last layer. ℓ\* is 1-indexed (ℓ\* = k is the output of the k-th
  block), so **d = ℓ\*/L ∈ (0, 1]**.
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
data/entities.py       stage 1a: entity extraction        (not written: blocked on data)
data/frequency.py      stage 1b: frequency scores         (not written: blocked on network)
run/forward.py         stage 2: resid_post at the final token for every layer
run/lens.py, run/patch.py, analysis/depth.py, analysis/stats.py    stubs (stages 3-4)
scripts/smoke.py       proxy smoke test on stage 2 output
tests/                 CPU tests, no big-model downloads
```

## Pipeline

```
make prep          # stage 0  -> data/processed/prompts.parquet, data_quality.md
make tokens        # option-token check -> config/models.yaml
make entities      # stage 1a -> data/processed/entities.parquet, entity_audit.csv
make frequency     # stage 1b -> data/processed/freq.parquet
make smoke         # stage 2 on the proxy (CPU, 50 questions) + smoke checks
make test
```

Stage 2 on a real model:

```
python -m run.forward --model llama31_8b --device cuda --split analysis
python -m run.forward --model llama31_8b --device cuda --split analysis --variant prompt_permuted
```

Shards land in `acts/{model}/{variant}/shard_XXXXX.npz` (`acts` fp16 [n, L, d],
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
- Dataset: HF `13ari/Sanskriti` (CC0). Revision: not yet pinned (unreachable at setup).
- infini-gram index: `v4_dolma-v1_7_llama` (Dolma v1.7), set in `config/run.yaml`;
  not yet queried.
- Wikipedia: MediaWiki action API (en, hi) and Wikimedia pageviews REST API,
  2024 totals, `user` agent type.

## Status

See `STATUS.md`.
