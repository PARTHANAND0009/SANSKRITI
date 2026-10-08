# Analysis plan (pre-registered)

Written 2026-10-08 (registered by commit c05260c at 2026-10-08T17:16Z), before any results from the full GPU run exist. The commit that
adds this file is the registration. After GPU results arrive this file is not edited; any
change of plan goes in the Deviations section at the end, dated, with the reason.

What exists at the time of writing: a CPU pilot (2,000 sampled questions, Llama-3.1-8B and
Qwen2.5-7B, original and one permuted option order, logit lens). The pilot was used to design
the metrics and is exploratory. It is not part of the confirmatory analysis below.

## 1. Data

- Source: SANSKRITI, `13ari/Sanskriti` at revision `8d8523795491a1484834d5cb85ed898b04839077`.
- Analysis set (`crystal.io.load_analysis_set`, 19,742 questions): all rows whose answer field
  matches an option, minus `ambiguous_gold` (gold text in two options) and minus
  `leaks_answer` (the stem contains the gold text, the gold state's name form, demonym or
  language, or "Indian" when the gold is India).
- Models (base, bf16): Llama-3.1-8B (L=32), Qwen2.5-7B (L=28), gemma-2-9b (L=42).
- Each question is run in four option orders, the cyclic rotations `cyc0..cyc3` that put the
  gold answer at A, B, C and D once each, plus one seeded random permutation `perm`
  (robustness only).
- Readout: logit lens is primary. Tuned lens is a secondary readout, reported in full.

## 2. Depth metrics

Readout layers are 0..L (0 = embeddings, k = output of block k); a depth in layers divided
by L gives d. All confirmatory metrics are computed on the four rotations together, with
option probabilities aligned by answer content (`analysis.depth aggregate`):

| metric | definition |
|---|---|
| `d_soft` | expected layer under the normalised positive increments of the gold content probability averaged across the four rotations |
| `l_star_cyc` | first layer where the gold content is top-1 under the rotation-averaged probabilities and stays top-1 through L |
| `d_margin` | first layer where the rotation-mean of the calibrated gold-minus-best-distractor log-prob margin is > 0 and stays > 0 (letter prior pooled over the four rotations) |
| `l_star_cal_mean` | mean over rotations of the letter-calibrated first-top-1-and-stays layer |
| `l_star_mean` | mean over rotations of the original l_star (first layer where the gold letter is top-1 and stays) |

### Primary metric: decision rule

The primary metric is **`d_soft`** unless the split-half reliability of the four-rotation
metrics shows a challenger is clearly more reliable. The rule (coded in
`analysis.depth.choose_primary`, applied once, on the logit lens, before any frequency
association is computed):

1. For each model, split-half reliability: aggregate `cyc0+cyc2` and `cyc1+cyc3` separately,
   Spearman correlation over questions correct in both halves, Spearman-Brown corrected to
   four rotations, with a 95% bootstrap CI (1,000 resamples).
2. A challenger (`l_star_cyc` or `d_margin`) replaces `d_soft` only if, in at least two of
   the three models, its corrected reliability exceeds `d_soft`'s by at least 0.10 **and**
   its CI lower bound is above `d_soft`'s CI upper bound.
3. If both challengers qualify, the one with the higher mean corrected reliability wins.

Reliabilities are reported for all five metrics regardless.

## 3. Hypotheses

**H1 (entity, primary).** Within a model, questions about better-documented entities
crystallise earlier: the coefficient of entity frequency on the primary depth metric is
negative.

**H2 (state, secondary).** Questions about better-documented states crystallise earlier:
the coefficient of state frequency is negative.

**H3 (causal check, secondary).** Patching depth (first layer where recovery >= 0.5) is
positively correlated with lens depth (`d_soft`) on the patching sample.

## 4. Model and controls

Per model and per depth metric, on questions answered correctly under the rotation-averaged
readout at layer L, with a defined depth and an entity frequency:

```
d ~ z(entity freq) + z(state log_freq) + acc + z(stem words)
    + z(log entity tokens) + z(entity words) + C(attribute) + C(question_type)
```

- Entity frequency, primary measure: `log_freq = log1p(exact-string count)` on infini-gram
  `v4_dolma-v1_7_llama`. Secondary measure: `log_freq_lenadj`, the residual of `log_freq` on
  log entity token length (identical test when log token length is a control; reported for
  uncontrolled summaries).
- `acc`: mean correctness across the four rotations.
- Estimator: linear mixed model with a random intercept per state (statsmodels MixedLM, REML).
  If it does not converge cleanly (non-convergence or a singular random-effect covariance),
  OLS with standard errors clustered by state is the reported estimator. Both are saved.
- Predictors are z-scored; effects are reported in layers per SD and as standardised
  coefficients (depth SDs per SD).

## 5. Tests and multiple comparisons

- H1: two-sided test of the entity-frequency coefficient. Holm-Bonferroni across the five
  depth metrics within each model (`p_holm`); the decision uses the primary metric's
  Holm-adjusted p.
- H2: state-label permutation test, 1,000 permutations (state -> state frequency shuffled
  across states, model refitted); Holm across the five metrics within each model.
- H3: Spearman correlation between patching depth and `d_soft` on valid patched questions
  (clean answer correct and corruption effect >= 1 logit), per model.
- alpha = 0.05 throughout.

## 6. What counts as which result

Let beta_std be the primary metric's standardised entity-frequency coefficient and SESOI =
0.10 (depth SDs per SD of frequency), the smallest effect of interest.

- **Positive (H1 supported):** Holm-adjusted p < 0.05 with a negative coefficient in at least
  two of the three models, and no model significant in the opposite direction.
- **Null:** in at least two of the three models the 90% CI of beta_std lies within
  [-0.10, 0.10] (`equivalent_null` in the regression table), and no model meets the positive
  criterion.
- **Opposite / mixed:** significant coefficients with opposite signs across models, or a
  significant positive coefficient (later crystallisation for better-documented entities)
  in at least two models. Reported as such.
- **Inconclusive:** anything else (e.g. non-significant but CIs wider than the SESOI).

The same categories are applied to H2 with the permutation p-values.

## 7. Exclusions and robustness (reported, not confirmatory)

- Question exclusions: ambiguous gold, leaks, wrong at layer L, no entity, no frequency.
- Patching rows with clean answer wrong or corruption effect < 1 logit are excluded from H3.
- Robustness: tuned lens readout; the `perm` order (original vs permuted test-retest);
  length-adjusted frequency; Wikipedia pageviews as an alternative frequency measure on the
  subset with an English article; per-question-type and per-attribute breakdowns; per-state
  medians with bootstrap CIs.

## 8. Software

Analysis code at the registration commit: `analysis/depth.py` (metrics, aggregation,
split-half, `choose_primary`), `analysis/stats.py` (models, Holm, permutation, equivalence),
`run/patch.py` (patching), `run/tune_lens.py` (tuned lens). Seed 1234.

## Deviations

Each deviation is appended here with a date, what changed and why.

- 2026-10-08: Disclosure, not a change of plan. This plan was written after the 2,000-question CPU pilot (Llama-3.1-8B, Qwen2.5-7B, partial Gemma-2-9B), whose results informed the metric definitions, controls and decision rules; the pilot is therefore exploratory, and only the full GPU run is confirmatory.
