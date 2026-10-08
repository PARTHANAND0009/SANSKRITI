# PRELIMINARY: depth vs documentation (CPU pilot, Llama-3.1-8B and Qwen2.5-7B)

**Everything here is preliminary.** CPU pilot, 2,000 sampled questions per model, two option
orders only (original + seeded permutation), logit lens. The planned design (4 cyclic option
rotations, full analysis set, tuned lens, patching, Gemma) has not been run yet. Produced by
`python -m analysis.stats` from `results/depth_{model}_prompt+prompt_permuted_logit_agg.parquet`.

## Data and model
- Rows: questions answered correctly under the content-averaged readout, with an entity
  `log_freq` (exact-string infini-gram count, Dolma v1.7). n = 1,265–1,401 per model and
  metric, 36 states.
- Depth as a fraction of L, averaged over the two orders by answer content (letter bias
  only partly cancels with two orders).
- `d ~ z(entity freq) + z(state log_freq) + acc + z(stem words) + z(log entity tokens) + z(entity words)
  + attribute + question_type`, fitted once per entity frequency measure: `raw` (log1p of the
  exact-string count) and `lenadj` (residual of that on log entity token length).
  Mixed model with a state random intercept: **every fit hit the boundary** (state variance
  ≈ 0 after controls), so per protocol OLS with state-clustered SEs is reported.
- Control: state labels shuffled across states (1,000 permutations) for the state coefficient.

## Primary metric
**`d_soft`** (expected layer under the normalised increase in debiased gold probability).
Chosen on test-retest reliability alone (`results/reliability_*.csv`, original vs permuted,
questions correct in both):

| metric (single order) | Llama Spearman / within 1 layer | Qwen Spearman / within 1 layer |
|---|---|---|
| l_star | 0.10 / 35% | 0.28 / 38% |
| l_star_cal | 0.17 / 62% | 0.24 / 62% |
| d_soft (calibrated) | **0.39** / 52% | **0.32** / 63% |

All single-order reliabilities are low; questions differ in depth by only ~2–3 layers (SD).
Spearman-Brown projects 0.72 (Llama) / 0.65 (Qwen) for d_soft averaged over 4 orders. The
4-rotation metrics (`l_star_cyc`, `d_margin`) cannot be tested on the pilot; the GPU run's
`analysis.depth split-half` will measure them.

## Effects, per +1 SD of entity frequency (layers; 95% CI; OLS, state-clustered)

Raw log count, with entity token length and word count controlled:

| model | metric | entity freq | Holm p | state log_freq | state perm. p |
|---|---|---|---|---|---|
| Llama (L=32) | d_soft | −0.24 [−0.42, −0.06] | 0.028 | −0.05 [−0.21, 0.11] | 0.58 |
| Llama | l_star_cyc | −0.35 [−0.54, −0.16] | 0.001 | −0.27 [−0.63, 0.10] | 0.059 |
| Llama | d_margin | −0.18 [−0.38, 0.02] | 0.077 | −0.22 [−0.46, 0.03] | 0.029 |
| Qwen (L=28) | d_soft | +0.18 [−0.07, 0.44] | 0.82 | −0.03 [−0.17, 0.11] | 0.71 |
| Qwen | l_star_cyc | +0.10 [−0.31, 0.50] | 1.00 | −0.21 [−0.53, 0.11] | 0.23 |
| Qwen | d_margin | −0.03 [−0.16, 0.09] | 1.00 | −0.08 [−0.15, −0.01] | 0.10 |

Length-adjusted frequency gives the same tests (same t and p) because the model already
controls log entity token length, and the adjusted measure is the raw one minus a linear
function of that control; only the per-SD scale changes (Llama d_soft −0.19 [−0.34, −0.05]
per SD of the adjusted measure). The adjusted measure matters where length is not
controlled (tier splits, state medians). Holm p: across the five depth metrics within model,
estimator, frequency measure and term. All rows: `PRELIMINARY_regression.csv`.

## Reading
- Effects are **small**: a few tenths of a layer per SD, against a depth SD of 2–6 layers.
- **Llama**: with entity length controlled, better-documented entities crystallise slightly
  earlier on most metrics (d_soft Holm p 0.028). **Qwen**: no entity effect once length is
  controlled (the uncontrolled +0.17 on d_soft in the previous version came partly from length).
  The two models still disagree in direction.
- **State frequency**: negative point estimates throughout, CIs mostly include 0; no
  permutation p survives Holm across metrics (smallest Holm-adjusted 0.12).
- Per-state medians with bootstrap CIs: `PRELIMINARY_state_medians.csv` (n per state 4–150;
  CIs mostly overlap).

## Caveats
- `log_freq` is an exact-string count and falls with entity length (log count vs log token
  length r = −0.59); entity token length and word count are now controlled.
- Two orders, not four: letter bias is only partly removed.
- Sample: seeded 2,000 of 19,742; Gemma excluded (incomplete).
