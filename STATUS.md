# Status (2026-10-09)

## Environment
- Python 3.13.16, torch 2.14.1 (CPU), transformers 5.18.0. No CUDA (expected). 4 CPUs, 15 GB RAM.
- Hugging Face token: read only from the `HF_TOKEN` environment variable (the stored
  credential file was deleted). Not set in this session; gated models need it on the GPU host.
- Network: huggingface.co (+ CDN), en/hi.wikipedia.org, wikimedia.org, api.infini-gram.io all reachable.
- Weights: Llama-3.1-8B 4 shards 16.06 GB bf16; Qwen2.5-7B ~15.2 GB bf16; gemma-2-9b ships
  **fp32, 8 shards, 36.97 GB**, converted to a local bf16 copy (18 GB) by
  `scripts/convert_bf16.py` (range requests, no fp32 on disk).

## Stage status
| stage | state | notes |
|---|---|---|
| 0 prep | done | 21,853 raw → 127 dropped, 10 ambiguous_gold, 1,974 leaks_answer; default analysis set 19,742 |
| tokens | done | all four models `space` (Llama [362, 426, 356, 423]) |
| 1a entities | done | 84.4% entity, 68.3% stem span (analysis set) |
| 1b frequency | done except pageviews | corpus_count and pageviews complete (`pageviews_complete: true`); Spearman(corpus_count, pageviews) 0.672, n=826; length-adjusted frequency added |
| lens | done | lens(last layer) == model output, bit-exact on proxy |
| design | done | 4 cyclic option orders per question (`cyc0..cyc3`, gold at A..D) + `perm` |
| 2 forward | done; CPU pilot run | `--variant cyc0..3/perm`, `--cpu-layers`, `--sample N` |
| 3a tuned lens | done, CPU-tested | `run/tune_lens.py`; Qwen2.5-0.5B: beats logit lens at every layer < L, identity at L |
| 3b patching | done, CPU-tested | `run/patch.py`; sample 1,014 questions (`patch_sample_cells.csv`) |
| 4a depth | done | l_star, l_star_cal, l_star_cyc, d_margin, d_soft; aggregate, reliability, split-half |
| 4b stats | PRELIMINARY | `analysis/stats.py`, `results/prelim/` (Llama + Qwen pilot) |
| GPU runbook | ready | `scripts/gpu_run.sh`, README section with estimates |

## CPU pilot (no GPU available): 2000 sampled questions per model
Seeded random sample (seed 1234) of the analysis set (ambiguous + leaks excluded), bf16 on a
4-core Xeon with AMX, decoder blocks partly streamed from disk. Results in `results/`.
Activations (`acts/`, ~1 GB per model) are not in git.

| model | variant | accuracy | d mean (raw) | l*_cal median | d_cal mean |
|---|---|---|---|---|---|
| Llama-3.1-8B (L=32) | original | 0.834 | 0.636 | 18 | 0.564 |
| Llama-3.1-8B | permuted | 0.832 | 0.638 | 18 | 0.567 |
| Qwen2.5-7B (L=28) | original | 0.811 | 0.735 | 20 | 0.743 |
| Qwen2.5-7B | permuted | 0.824 | 0.752 | 20 | 0.739 |
| gemma-2-9b (L=42) | original, 1536 q* | 0.848 | 0.599 | 28 | 0.672 |
| gemma-2-9b | permuted | not run | | | |

\* Gemma stopped at 6 of 8 shards: a container restart moved the session to a host without
AMX / AVX-512-bf16, where bf16 matmuls ran >8x slower (an hour without finishing one shard).
Shards run in qid order, so these 1536 are the first 1536 of the sample by qid and include
only 38 State Prediction questions; compare models on the matched set below.

Matched comparison, same 1536 questions, original prompts:

| model | accuracy | l*_cal median | d_cal q25 / median / q75 | raw d mean (correct) |
|---|---|---|---|---|
| Llama-3.1-8B | 0.845 | 18 / 32 | 0.53 / 0.56 / 0.56 | 0.640 |
| Qwen2.5-7B | 0.828 | 20 / 28 | 0.71 / 0.71 / 0.79 | 0.731 |
| gemma-2-9b | 0.848 | 28 / 42 | 0.64 / 0.67 / 0.69 | 0.599 |

In every model the calibrated gold-top-1 share first exceeds 0.5 exactly at the median
l*_cal: crystallisation is a sharp, model-specific event (d = 0.56 / 0.71 / 0.67).

Checks on all three real models (Llama, Qwen, Gemma incl. softcap): lens at the last layer reproduces the model's full-vocabulary
output exactly (with disk offload); stored-fp16 readout agrees with model output on all
checked questions; no fp16 overflow (Llama max |x| 28).

Findings that matter for stage 4:
1. **Early l* is a letter prior.** Uncalibrated early layers rank one letter first for every
   question (B at layer 3 in both models). All 273 Llama questions with l* <= 13 have gold B;
   all 271 Qwen questions with l* <= 12 have gold A. `l_star_cal` removes each layer's mean
   letter log-prob; with it early layers sit at chance and crystallisation is a narrow event
   (Llama layers 17-18; Qwen layers 19-22).
2. **Per-question l* is not stable across option orders.** Raw l* identical in 28% of
   questions (Llama and Qwen), within one layer 35-38%; calibrated within one layer 61-62%,
   Spearman 0.14 (Llama) / 0.27 (Qwen). Aggregate depth is stable. Per-question depth should
   be averaged over option orders (e.g. all 4 cyclic rotations) before stage 4.
3. Exploratory (Llama, raw d, correct only): entity log_freq vs d Spearman -0.07 (p=0.008);
   low vs high tier d 0.651 vs 0.624. Accuracy unrelated to log_freq. Not controlled.

## Session 2026-10-09: design, metrics, preliminary stats, tuned lens, patching
- **Primary depth metric: `d_soft`** (expected layer under the normalised increase in debiased
  gold probability), chosen on test-retest reliability (original vs permuted order, questions
  correct in both): Spearman 0.39 (Llama) / 0.32 (Qwen) vs l_star 0.10 / 0.28 and l_star_cal
  0.17 / 0.24. All single-order reliabilities are low; Spearman-Brown projects 0.72 / 0.65 for
  4 orders. The 4-rotation metrics (l_star_cyc, d_margin) are measured by `split-half` on the
  GPU run.
- Letter-prior calibration must use a prior pooled over balanced rows (all 4 rotations): in a
  single cyclic variant every gold is the same letter and its own prior erases the signal.
- PRELIMINARY effects (`results/prelim/README.md`), per +1 SD log_freq, d_soft, layers:
  entity Llama -0.18 [-0.32, -0.03], Qwen +0.17 [+0.01, +0.34] (opposite signs); state
  -0.06 [-0.22, 0.09] / -0.03 [-0.16, 0.10], permutation p 0.50 / 0.73. State random
  effect at the boundary (variance ~0) in every mixed fit; OLS with clustered SEs reported.
- Tuned lens: per-layer input scaling was needed (residual norms 0.4 -> 84 across Qwen2.5-0.5B's
  layers made plain SGD diverge at the last layer).
- Patching: Country Prediction (answer always India) and stems naming a state are excluded,
  since an entity swap cannot change their answer; 30 dangling-tail entities dropped from the
  swap pool. On Qwen2.5-0.5B only 1 of 20 corruptions moved the answer by >= 1 logit (12 clean
  answers already wrong); mechanics verified (recovery 0 at layer 0, 1 at layer L).

## Data quality findings (data/processed/data_quality.md, for the SANSKRITI authors)
1. 127 rows: answer matches no option (dropped).
2. 10 rows: gold text appears in two options (excluded).
3. 73 more rows: a distractor is duplicated (flagged `duplicate_options`, kept).
4. 1,974 rows: stem reveals the answer (`leaks_answer`, excluded by default; `leaks.csv`).
5. 1,891 Association rows: gold option is the row's own state rather than a region
   (913 of them also name the state in the stem). 15 rows: gold repeats the item.
Also: 1,634 duplicated stems; mojibake in some stems (e.g. "stateâs").

## Stage 1b caveats
- `corpus_count` is the exact entity string on `v4_dolma-v1_7_llama`. It anti-correlates
  with entity length (Spearman −0.57 vs word count): many entities are descriptive phrases
  ("Art form showcased during Mysuru Dasara") with zero exact matches. 21.4% of entities
  count 0. This confounds state-level comparisons (Karnataka median 0.00).
- 3 entities contain double spaces and count 0 because of it ("Mamluk  Dynasty" 0 vs
  "Mamluk Dynasty" 1,237). Left as exact strings.
- 129 of 841 English articles are reached via a redirect that changes the concept
  (`redirect_changed_concept`), e.g. "Hemis Festival" → "Hemis Monastery".
- Commit f1a7cad carried freq*.parquet from a run where 1,719 infini-gram
  403 throttling responses had been cached as answers; those files are superseded.

## Resuming pageviews
```
python -m data.frequency                         # fetches only the uncached pageviews (125 left)
python -m data.frequency --pageviews cached-only # rebuild outputs from cache, no requests
```
The cache (`data/cache/`, 8.8 MB, force-added to git) holds every answered request. The
pageviews API rate-limits this cloud IP to about 4 requests/min (429, Retry-After ~48 s),
so the remaining 125 take about 30-45 minutes.

## Next
- `HF_TOKEN=... bash scripts/gpu_run.sh` on an A100 (README: ~6-8 h on 40 GB, ~4-6 h on 80 GB,
  ~160 GB disk). Then check split-half reliability before fixing the primary metric.
- Decide whether `corpus_count` should be normalised (whitespace, phrase length) before
  stage 4; hand-check `entity_audit.csv`.
