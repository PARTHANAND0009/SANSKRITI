# Status (2026-10-06)

## Environment
- Python 3.13.16, torch 2.14.1 (CPU), transformers 5.18.0. No CUDA (expected). 4 CPUs, 15 GB RAM.
- `HF_TOKEN` not set in this session.
- Network: huggingface.co (+ CDN), en/hi.wikipedia.org, wikimedia.org, api.infini-gram.io all reachable.
- Gated models: `google/gemma-2-9b` config, tokenizer, weight index and shard headers download
  (no token needed from this session). Weights are **fp32, 8 shards, 36.97 GB**.
  `meta-llama/Llama-3.1-8B` still 403 (needs `HF_TOKEN` with accepted licence).

## Stage status
| stage | state | notes |
|---|---|---|
| 0 prep | done | 21,853 raw → 127 dropped, 10 ambiguous_gold, 1,974 leaks_answer; default analysis set 19,742 |
| tokens | 3/4 | Qwen2.5-7B, Qwen2.5-0.5B, gemma-2-9b: `space`; Llama-3.1-8B pending (gated) |
| 1a entities | done | 84.4% entity, 68.3% stem span (analysis set) |
| 1b frequency | done except pageviews | corpus_count complete; pageviews 108/877 (`pageviews_complete: false`) |
| lens | done | lens(last layer) == model output, bit-exact on proxy |
| 2 forward | done, proxy-tested | layer 0 = embeddings; ready for GPU |
| 3–4 | stubs | |

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
python -m data.frequency                         # fetches only the 769 uncached pageviews
python -m data.frequency --pageviews cached-only # rebuild outputs from cache, no requests
```
The cache (`data/cache/`, 8.8 MB, force-added to git) holds every answered request. The
pageviews API rate-limits this cloud IP to about 4 requests/min (429, Retry-After ~48 s),
so the remaining 769 take roughly 3 hours.

## Next
- Set `HF_TOKEN` (Llama licence accepted), then `make tokens` and `make prep`.
- GPU run of stage 2 for the three models (see README).
- Decide whether `corpus_count` should be normalised (whitespace, phrase length) before
  stage 4; hand-check `entity_audit.csv`.
