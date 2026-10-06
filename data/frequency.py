"""Stage 1b: documentation / frequency scores. NOT YET WRITTEN.

Blocked: depends on stage 1a entities, and en/hi.wikipedia.org, wikimedia.org
and api.infini-gram.io were not reachable from the setup session, so response
formats could not be verified.

Planned output: data/processed/freq.parquet with qid, entity, wiki_exists_en,
wiki_exists_hi, wiki_bytes_en, wiki_pageviews_en (2024 total), corpus_count
(infini-gram, index in config/run.yaml), log_freq + log_freq_source, tier.
Every response cached under data/cache/, rate-limited, retried with backoff.
"""


def main():
    raise SystemExit("stage 1b not written yet: needs stage 1a and network access (see docstring)")


if __name__ == "__main__":
    main()
