"""Stage 1a: deterministic entity extraction. NOT YET WRITTEN.

Blocked: the extraction rules have to come from the real stem templates
(step 1 of the stage is to inspect templates per question_type), and the
dataset could not be downloaded in the setup session (huggingface.co was not
reachable). Rules will not be guessed ahead of seeing the data.

Planned output: data/processed/entities.parquet with qid, entity,
entity_source (stem|answer), span_start, span_end, extraction_rule,
confidence; data/processed/entity_audit.csv (random 100); swap-pool report.
No LLM API is used.
"""


def main():
    raise SystemExit("stage 1a not written yet: needs data/processed/prompts.parquet (see docstring)")


if __name__ == "__main__":
    main()
