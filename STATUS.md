# Status (2026-10-08)

## Environment
- Python 3.13, torch 2.14.1 (CPU), transformers 5.18.0; no GPU in this container.
- `HF_TOKEN` is read from the environment only; it is not set here. Llama and Gemma are gated.
- Gemma-2-9b ships fp32 (36.97 GB); `scripts/convert_bf16.py` makes an 18 GB bf16 copy.

## Stages
| stage | state | notes |
|---|---|---|
| 0 prep | done | 21,853 rows: 127 dropped, 10 ambiguous_gold, 1,974 leaks_answer; analysis set 19,742 |
| tokens | done | " A".." D" single tokens in all four models (Llama ids 362, 426, 356, 423) |
| 1a entities | done | 84.4% of the analysis set has an entity, 68.3% a stem span |
| 1b frequency | done | corpus counts and pageviews complete; Spearman(count, pageviews) 0.67, n = 826 |
| lens | done | the lens at layer L reproduces the model output on all three models |
| 2 forward | done; CPU pilot | variants `cyc0..cyc3` (gold at A..D) and `perm` |
| 3a tuned lens | done, CPU-tested | per-layer input scaling needed; beats the logit lens below L on the proxy |
| 3b patching | done, CPU-tested | stratified sample of 1,200 questions, 36 states (Maharashtra 57, Bihar 52) |
| 4a depth | done | five metrics; aggregate over orders; reliability; split-half; primary-metric rule |
| 4b stats | preliminary | `results/prelim/` (Llama and Qwen pilot) |
| GPU run | ready | `RUN_FOR_ARIJIT.md`, `scripts/preflight.py`, `scripts/gpu_run.sh` |
| paper | draft | `paper/`, ACL format, 12 TODO-RESULT markers |

## CPU pilot (exploratory)
2,000 sampled questions per model, original and one permuted order, logit lens. Gemma
stopped at 1,536 questions after the container moved to a host without bf16 matrix units.

On the 1,536 questions all three models share, the letter-calibrated ℓ\* median is
18/32 (Llama), 20/28 (Qwen) and 28/42 (Gemma): d = 0.56, 0.71, 0.67.

Findings:
1. **Letter bias.** Uncalibrated early layers rank one letter first for nearly every
   question: every Llama question with ℓ\* <= 13 has gold B, and every Qwen question with
   ℓ\* <= 12 has gold A. Hence the four cyclic orders, and a letter prior pooled over
   orders with balanced gold letters.
2. **Single-order depths are unreliable.** Test-retest Spearman across two orders is 0.10
   to 0.28 for ℓ\*, and 0.39 (Llama) and 0.32 (Qwen) for the continuous `d_soft`.
3. **Frequency, with length controls** (d_soft, layers per SD): Llama -0.24 [-0.42, -0.06],
   Holm p 0.028; Qwen +0.18 [-0.07, 0.44]. No state effect (permutation p 0.58, 0.71).
   Every mixed model hit the boundary, so OLS with state-clustered SEs is reported.

## Data quality (`data/processed/data_quality.md`, for the SANSKRITI authors)
1. 127 rows: the answer matches no option (dropped).
2. 10 rows: the gold text appears in two options (excluded).
3. 73 more rows: a distractor is duplicated (flagged, kept).
4. 1,974 rows: the stem reveals the answer (`leaks_answer`, excluded; `leaks.csv`).
5. 1,891 Association rows: the gold option is the row's own state rather than a region;
   969 of them also name the state in the stem. 15 rows: the gold repeats the item.

## Frequency caveats
- Exact-string counts fall with entity length (slope -4.95, r -0.59 on log tokens); 21.4% of
  entities count 0. Hence the length controls and `log_freq_lenadj`.
- 3 entities contain double spaces and count 0 because of it; kept as exact strings.
- 129 of 841 English articles are reached through a redirect that changes the concept
  (`redirect_changed_concept`).

## Next
- Arijit runs `scripts/preflight.py`, then `ARCHIVE=1 SKIP_SETUP=1 bash scripts/gpu_run.sh`.
- `python results/ingest.py results_<date>.tar.gz`, then `python analysis/figures.py`, then
  fill the TODO-RESULT markers.
