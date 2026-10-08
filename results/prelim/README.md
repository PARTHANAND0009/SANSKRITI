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
- `d ~ z(entity log_freq) + z(state log_freq) + acc + z(stem words) + attribute + question_type`.
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

## Effects, per +1 SD of log_freq (layers; 95% CI; OLS, state-clustered)

| model | metric | entity log_freq | state log_freq | state perm. p |
|---|---|---|---|---|
| Llama (L=32) | d_soft | −0.18 [−0.32, −0.03] | −0.06 [−0.22, 0.09] | 0.50 |
| Llama | l_star_cyc | −0.30 [−0.48, −0.12] | −0.28 [−0.61, 0.06] | 0.045 |
| Llama | d_margin | −0.07 [−0.20, 0.05] | −0.24 [−0.48, 0.00] | 0.020 |
| Qwen (L=28) | d_soft | +0.17 [+0.01, +0.34] | −0.03 [−0.16, 0.10] | 0.73 |
| Qwen | l_star_cyc | +0.29 [−0.06, +0.65] | −0.25 [−0.57, 0.08] | 0.12 |
| Qwen | d_margin | +0.04 [−0.05, +0.12] | −0.09 [−0.16, −0.03] | 0.053 |

Other metrics, all terms and full CIs: `PRELIMINARY_regression.csv`.

## Reading
- Effects are **small**: a few tenths of a layer per SD, against a depth SD of 2–6 layers.
- **Entity frequency has opposite signs** in the two models on the primary metric (Llama
  earlier, Qwen later). No consistent entity effect.
- **State frequency**: point estimates are negative (better-documented states slightly
  earlier) for every metric and model, but the primary metric's CIs include 0 and its
  permutation p is 0.50 / 0.73. Some event-style metrics pass the permutation control at
  p < 0.05; with 10 model × metric tests and no correction, that is weak evidence.
- Per-state medians with bootstrap CIs: `PRELIMINARY_state_medians.csv` (n per state 4–150;
  CIs mostly overlap).

## Caveats
- `log_freq` is an exact-string count and falls with entity length (Spearman −0.57 vs word
  count); stem length is controlled, entity length is not.
- Two orders, not four: letter bias is only partly removed.
- Sample: seeded 2,000 of 19,742; Gemma excluded (incomplete).
