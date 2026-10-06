# Status (setup session, 2026-10-06)

## Environment
- Python 3.13.16, torch 2.14.1 (CPU wheel from PyPI), CUDA not available (expected).
- 4 CPUs, 15 GB RAM.
- `HF_TOKEN` not set.

## Network, re-check 2026-10-06 (after allowlist update)
All required endpoints answer 200: HF dataset API, en.wikipedia REST, hi.wikipedia
action API, wikimedia pageviews, api.infini-gram.io (POST count on
`v4_dolma-v1_7_llama` -> `{"approx", "count", "latency", "token_ids", "tokens"}`).
`HF_TOKEN` not set: Llama-3.1-8B and gemma-2-9b config.json -> 403 (gated, manual);
Qwen2.5-7B and Qwen2.5-0.5B accessible.

## Network, first check (one request each, through the session egress proxy)
| endpoint | result |
|---|---|
| huggingface.co | blocked (proxy 403 on CONNECT) |
| en.wikipedia.org (REST) | blocked (403) |
| wikimedia.org (pageviews) | blocked (403) |
| api.infini-gram.io | blocked (403) |
| hi.wikipedia.org | blocked (403) |
| pypi.org / files.pythonhosted.org | ok |

Domains to allow: `huggingface.co`, `cdn-lfs.huggingface.co`, `cas-bridge.xethub.hf.co`
(HF file/LFS/Xet downloads), `en.wikipedia.org`, `hi.wikipedia.org`, `wikimedia.org`,
`api.infini-gram.io`.

## Stages
| stage | code | tests | run |
|---|---|---|---|
| 0 prep (`data/prep.py`) | done | offline fixtures pass | **blocked**: dataset unreachable |
| token check (`crystal/tokens.py`) | done | logic tests pass; per-model tests skip | **blocked**: all 4 `pending` |
| 1a entities | not written | – | blocked: needs the data to derive templates |
| 1b frequency | not written | – | blocked: needs 1a + Wikipedia/infini-gram |
| lens (`crystal/lens.py`) | done | lens == output logits, Llama/Qwen2/Gemma2, tied+untied | – |
| 2 forward (`run/forward.py`) | done | batched == single (default fp32 tol), resume, end-to-end smoke on tiny models | **blocked**: proxy run needs prompts + weights |

## Once the network is open, in order
```
make prep          # check the printed counts: expect ~127 dropped, ~83 ambiguous
make tokens        # records option_token_variant (needs HF_TOKEN for Llama/Gemma)
make test
make smoke         # proxy, CPU, 50 questions
```
Then stage 1a (inspect templates first), then 1b.
