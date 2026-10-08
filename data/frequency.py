"""Stage 1b: documentation scores per entity and per state.

Per distinct entity string:
  wiki_exists_en         an en.wikipedia article under that title, after normalisation and
                         redirects; disambiguation pages count as missing (wiki_disambig_en)
  wiki_title_en          the resolved title
  redirect_changed_concept  the title differs beyond case, punctuation and plural
                         ("Hemis Festival" -> "Hemis Monastery"); null without an article
  wiki_exists_hi         the en article has a Hindi interlanguage link (null without one)
  wiki_bytes_en          article length in bytes
  wiki_pageviews_en      2024 total views (agent=user); null when the API has no data
  corpus_count           infini-gram exact count on frequency.infinigram_index
  log_freq               log1p(corpus_count), else log1p(pageviews) (log_freq_source says which)
  entity_n_tokens        length in the index tokenizer; entity_n_words: word count
  log_freq_lenadj        residual of log1p(corpus_count) on log(entity_n_tokens): exact-string
                         counts fall with length, so this is frequency among same-length entities
  tier                   high when log_freq is above its median within the attribute
States get the same scores for their own article (STATE_TITLES).

2xx and 404 responses are cached in data/cache/{service}/ and never requested again, so a
rerun resumes. 403/429/5xx and network errors are retried with backoff and never cached.
Outputs: data/processed/freq.parquet (per qid), freq_entities.parquet, freq_state.parquet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import time
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

from crystal.io import load_analysis_set, load_entities, load_prompts, load_run_config, resolve, set_seed
from crystal.log import setup_logging

log = logging.getLogger(__name__)

# state -> (en.wikipedia title of its article, string counted in the corpus)
STATE_TITLES = {
    "Andaman_and_Nicobar": ("Andaman and Nicobar Islands", "Andaman and Nicobar"),
    "Andhra_Pradesh": ("Andhra Pradesh", "Andhra Pradesh"),
    "Arunachal_Pradesh": ("Arunachal Pradesh", "Arunachal Pradesh"),
    "Assam": ("Assam", "Assam"),
    "Bihar": ("Bihar", "Bihar"),
    "Chandigarh": ("Chandigarh", "Chandigarh"),
    "Chhattisgarh": ("Chhattisgarh", "Chhattisgarh"),
    "Dadra_and_Nagar_Haveli_and_Daman_and_Diu": (
        "Dadra and Nagar Haveli and Daman and Diu",
        "Dadra and Nagar Haveli and Daman and Diu",
    ),
    "Delhi": ("Delhi", "Delhi"),
    "Goa": ("Goa", "Goa"),
    "Gujarat": ("Gujarat", "Gujarat"),
    "Haryana": ("Haryana", "Haryana"),
    "Himachal_Pradesh": ("Himachal Pradesh", "Himachal Pradesh"),
    "Jammu_kashmir": ("Jammu and Kashmir (union territory)", "Jammu and Kashmir"),
    "Jharkhand": ("Jharkhand", "Jharkhand"),
    "Karnataka": ("Karnataka", "Karnataka"),
    "Kerala": ("Kerala", "Kerala"),
    "Ladakh": ("Ladakh", "Ladakh"),
    "Lakshadweep": ("Lakshadweep", "Lakshadweep"),
    "Madhya_Pradesh": ("Madhya Pradesh", "Madhya Pradesh"),
    "Maharashtra": ("Maharashtra", "Maharashtra"),
    "Manipur": ("Manipur", "Manipur"),
    "Meghalaya": ("Meghalaya", "Meghalaya"),
    "Mizoram": ("Mizoram", "Mizoram"),
    "Nagaland": ("Nagaland", "Nagaland"),
    "Odisha": ("Odisha", "Odisha"),
    "Puducherry": ("Puducherry (union territory)", "Puducherry"),
    "Punjab": ("Punjab, India", "Punjab"),
    "Rajasthan": ("Rajasthan", "Rajasthan"),
    "Sikkim": ("Sikkim", "Sikkim"),
    "Tamil_Nadu": ("Tamil Nadu", "Tamil Nadu"),
    "Telangana": ("Telangana", "Telangana"),
    "Tripura": ("Tripura", "Tripura"),
    "Uttar_Pradesh": ("Uttar Pradesh", "Uttar Pradesh"),
    "Uttarakhand": ("Uttarakhand", "Uttarakhand"),
    "West_Bengal": ("West Bengal", "West Bengal"),
}

MW_API = "https://en.wikipedia.org/w/api.php"
PV_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{title}/monthly/{start}/{end}"
IG_API = "https://api.infini-gram.io/"
MW_BATCH = 50
CACHEABLE = {404}
TRANSIENT = {403, 429}


class CachedClient:
    def __init__(
        self,
        cache_dir: Path,
        user_agent: str,
        min_interval: float,
        max_retries: int = 8,
        service_intervals: dict | None = None,
        max_retry_after: float = 300,
    ):
        self.cache_dir = cache_dir
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.min_interval = min_interval
        self.service_intervals = service_intervals or {}
        self.max_retries = max_retries
        self.max_retry_after = max_retry_after
        self._last: dict[str, float] = {}
        self.hits = self.misses = 0
        self.throttled: dict[str, int] = {}
        self.not_fetched: dict[str, int] = {}

    def _path(self, service: str, key: dict) -> Path:
        h = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        return self.cache_dir / service / f"{h}.json"

    def request(
        self, service: str, method: str, url: str, params=None, body=None, cache_only: bool = False
    ) -> tuple[int | None, object]:
        """(status, parsed JSON or None). cache_only returns (None, None) for anything not cached."""
        key = {"method": method, "url": url, "params": params, "body": body}
        path = self._path(service, key)
        if path.exists():
            self.hits += 1
            c = json.loads(path.read_text())
            return c["status"], c["json"]
        if cache_only:
            self.not_fetched[service] = self.not_fetched.get(service, 0) + 1
            return None, None
        self.misses += 1
        interval = self.service_intervals.get(service, self.min_interval)
        for attempt in range(self.max_retries):
            wait = interval - (time.monotonic() - self._last.get(service, 0.0))
            if wait > 0:
                time.sleep(wait)
            self._last[service] = time.monotonic()
            backoff = min(2**attempt, 60)
            try:
                r = self.session.request(method, url, params=params, json=body, timeout=60)
            except requests.RequestException as e:
                err = f"{type(e).__name__}: {e}"
            else:
                # Only 2xx and 404 are answers about the query. infini-gram's gateway answers
                # 403 under load, so 403/429/5xx are treated as throttling and never cached.
                if r.status_code in TRANSIENT or r.status_code >= 500:
                    err = f"HTTP {r.status_code}"
                    self.throttled[service] = self.throttled.get(service, 0) + 1
                    ra = r.headers.get("Retry-After", "")
                    if ra.isdigit():
                        backoff = min(max(float(ra), backoff), self.max_retry_after)
                else:
                    try:
                        js = r.json()
                    except ValueError:
                        js = None
                    if not (200 <= r.status_code < 300 or r.status_code in CACHEABLE):
                        return r.status_code, js  # other 4xx: an answer for this run only
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(".tmp")
                    tmp.write_text(json.dumps({"request": key, "status": r.status_code, "json": js}))
                    tmp.replace(path)
                    return r.status_code, js
            time.sleep(backoff)
        raise RuntimeError(f"{service}: giving up on {url} {params or body}: {err}")


def mw_lookup(client: CachedClient, titles: list[str]) -> dict:
    """{input string: dict(wiki_exists_en, wiki_title_en, wiki_disambig_en, wiki_bytes_en, wiki_exists_hi)}"""
    out = {}
    for i in range(0, len(titles), MW_BATCH):
        batch = titles[i : i + MW_BATCH]
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "redirects": "1",
            "prop": "info|langlinks|pageprops",
            "lllang": "hi",
            "lllimit": "max",
            "ppprop": "disambiguation",
            "titles": "|".join(batch),
        }
        pages, norm, redir = {}, {}, {}
        cont = {}
        while True:
            status, js = client.request("mediawiki", "GET", MW_API, params={**params, **cont})
            if status != 200 or js is None or "query" not in js:
                raise RuntimeError(f"mediawiki: unexpected answer {status}: {str(js)[:300]}")
            q = js["query"]
            norm.update({n["from"]: n["to"] for n in q.get("normalized", [])})
            redir.update({r["from"]: r["to"] for r in q.get("redirects", [])})
            for p in q.get("pages", []):
                prev = pages.get(p["title"], {})
                if prev.get("langlinks"):
                    p = {**p, "langlinks": prev["langlinks"] + p.get("langlinks", [])}
                pages[p["title"]] = {**prev, **p}
            if "continue" not in js:
                break
            cont = js["continue"]
        for t in batch:
            title = norm.get(t, t)
            title = redir.get(title, title)
            p = pages.get(title)
            ok = p is not None and not p.get("missing") and not p.get("invalid")
            disamb = bool(ok and "disambiguation" in p.get("pageprops", {}))
            exists = ok and not disamb
            out[t] = {
                "wiki_exists_en": exists,
                "wiki_title_en": title if exists else None,
                "wiki_disambig_en": disamb,
                "wiki_bytes_en": p.get("length") if exists else None,
                "wiki_exists_hi": bool(p.get("langlinks")) if exists else None,
            }
    return out


def pageviews(client: CachedClient, title: str, year: int, cache_only: bool = False):
    """(total views or None, status), status one of fetched, no_data (404), not_fetched."""
    t = urllib.parse.quote(title.replace(" ", "_"), safe="")
    status, js = client.request(
        "pageviews", "GET", PV_API.format(title=t, start=f"{year}0101", end=f"{year}1231"), cache_only=cache_only
    )
    if status is None:
        return None, "not_fetched"
    if status == 404:
        return None, "no_data"
    if status != 200 or js is None:
        raise RuntimeError(f"pageviews: {status} for {title}: {str(js)[:200]}")
    return int(sum(it["views"] for it in js.get("items", []))), "fetched"


def corpus_count(client: CachedClient, index: str, text: str):
    """(count, approx flag, query length in the index tokenizer); Nones when unavailable."""
    status, js = client.request(
        "infinigram", "POST", IG_API, body={"index": index, "query_type": "count", "query": text}
    )
    if status != 200 or js is None or "error" in js or "count" not in js:
        return None, None, None
    toks = js.get("token_ids")
    return int(js["count"]), bool(js.get("approx")), (len(toks) if toks is not None else None)


def score_strings(
    client, strings_wiki: list[str], strings_corpus: list[str], cfg: dict, pageviews_cache_only: bool = False
) -> pd.DataFrame:
    """Scores aligned with the inputs: strings_wiki looked up as titles, strings_corpus counted.

    One pass per service, so a throttled service does not hold up the others.
    """
    fcfg = cfg["frequency"]
    wiki = mw_lookup(client, sorted(set(strings_wiki)))
    rows = [dict(wiki[w]) for w in strings_wiki]
    log.info(f"  wikipedia metadata done ({client.misses} requests, {client.hits} cached)")
    for n, (r, c) in enumerate(zip(rows, strings_corpus)):
        r["corpus_count"], r["corpus_count_approx"], r["corpus_n_tokens"] = corpus_count(
            client, fcfg["infinigram_index"], c
        )
        if (n + 1) % 250 == 0:
            log.info(f"  corpus counts {n + 1}/{len(rows)}")
    todo = [r for r in rows if r["wiki_exists_en"]]
    for n, r in enumerate(todo):
        r["wiki_pageviews_en"], r["pageviews_status"] = pageviews(
            client, r["wiki_title_en"], fcfg["pageviews_year"], cache_only=pageviews_cache_only
        )
        if (n + 1) % 50 == 0 and not pageviews_cache_only:
            log.info(f"  pageviews {n + 1}/{len(todo)} (429s so far: {client.throttled.get('pageviews', 0)})")
    for r in rows:
        r.setdefault("wiki_pageviews_en", None)
        r.setdefault("pageviews_status", "no_article")
    return pd.DataFrame(rows)[
        [
            "wiki_exists_en",
            "wiki_title_en",
            "wiki_disambig_en",
            "wiki_bytes_en",
            "wiki_exists_hi",
            "wiki_pageviews_en",
            "pageviews_status",
            "corpus_count",
            "corpus_count_approx",
            "corpus_n_tokens",
        ]
    ]


def _concept_key(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", str(s).replace("_", " ").lower())
    words = []
    for w in s.split():
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 4 and w.endswith(("ses", "xes", "zes", "ches", "shes")):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        words.append(w)
    return " ".join(words)


def redirect_changed_concept(entity: str, title) -> bool | None:
    if title is None or (isinstance(title, float) and math.isnan(title)):
        return None
    return _concept_key(entity) != _concept_key(title)


def add_length_adjusted(fe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add entity_n_tokens, entity_n_words and log_freq_lenadj (OLS over distinct entities).

    Returns:
        (fe with the new columns, fit summary: slope, intercept, r, n)
    """
    fe = fe.copy()
    fe["entity_n_tokens"] = pd.to_numeric(fe["corpus_n_tokens"], errors="coerce")
    fe["entity_n_words"] = fe["entity"].str.split().str.len()
    y = np.log1p(pd.to_numeric(fe["corpus_count"], errors="coerce").astype(float))
    x = np.log(fe["entity_n_tokens"].astype(float))
    ok = y.notna() & x.notna() & np.isfinite(x)
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    fe["log_freq_lenadj"] = np.where(ok, y - (intercept + slope * x), np.nan)
    r = np.corrcoef(x[ok], y[ok])[0, 1]
    return fe, {"slope": float(slope), "intercept": float(intercept), "r": float(r), "n": int(ok.sum())}


def add_log_freq(df: pd.DataFrame) -> pd.DataFrame:
    cc = pd.to_numeric(df["corpus_count"], errors="coerce").astype(float)
    pv = pd.to_numeric(df["wiki_pageviews_en"], errors="coerce").astype(float)
    df["log_freq"] = np.where(cc.notna(), np.log1p(cc), np.log1p(pv))
    src = [("corpus_count" if c else "pageviews" if v else None) for c, v in zip(cc.notna(), pv.notna())]
    df["log_freq_source"] = pd.Series(src, index=df.index, dtype="object")
    return df


def median_tier(df: pd.DataFrame, by="attribute") -> pd.Series:
    lf = df["log_freq"].astype(float)
    med = lf.groupby(df[by]).transform("median")
    tier = pd.Series(np.where(lf > med, "high", "low"), index=df.index, dtype="object")
    return tier.where(df["log_freq"].notna(), None)


def main(argv=None):
    setup_logging()
    ap = argparse.ArgumentParser(description="stage 1b: frequency scores (resumable from data/cache/)")
    ap.add_argument(
        "--pageviews",
        choices=["fetch", "cached-only"],
        default="fetch",
        help="cached-only: make no pageview requests; uncached pageviews stay null",
    )
    args = ap.parse_args(argv)
    pv_cache_only = args.pageviews == "cached-only"
    cfg = load_run_config()
    set_seed(cfg["seed"])
    if "CHANGE_ME" in cfg["frequency"]["user_agent"]:
        raise SystemExit("set frequency.user_agent contact in config/run.yaml first")
    client = CachedClient(
        resolve(cfg["paths"]["cache"]),
        cfg["frequency"]["user_agent"],
        cfg["frequency"]["min_interval_s"],
        service_intervals=cfg["frequency"].get("service_intervals_s"),
    )
    out = resolve(cfg["paths"]["processed"])
    prompts = load_prompts()
    ent = load_entities()

    ents = sorted(ent.entity.dropna().unique())
    log.info(f"scoring {len(ents)} distinct entity strings (index {cfg['frequency']['infinigram_index']})")
    fe = pd.concat(
        [pd.DataFrame({"entity": ents}), score_strings(client, ents, ents, cfg, pageviews_cache_only=pv_cache_only)],
        axis=1,
    )
    fe["redirect_changed_concept"] = [redirect_changed_concept(e, t) for e, t in zip(fe.entity, fe.wiki_title_en)]
    fe = add_log_freq(fe)
    fe, lenfit = add_length_adjusted(fe)
    fe.to_parquet(out / "freq_entities.parquet", index=False)

    keys = sorted(prompts.state.unique())
    missing = set(keys) - set(STATE_TITLES)
    if missing:
        raise SystemExit(f"no title mapping for states {missing}")
    fs = pd.concat(
        [
            pd.DataFrame(
                {
                    "state": keys,
                    "state_title": [STATE_TITLES[k][0] for k in keys],
                    "state_corpus_string": [STATE_TITLES[k][1] for k in keys],
                }
            ),
            score_strings(
                client,
                [STATE_TITLES[k][0] for k in keys],
                [STATE_TITLES[k][1] for k in keys],
                cfg,
                pageviews_cache_only=pv_cache_only,
            ),
        ],
        axis=1,
    )
    fs = add_log_freq(fs)
    bad = fs[~fs.wiki_exists_en.astype(bool)]
    if len(bad):
        raise SystemExit(f"state titles not resolving to an article: {bad.state_title.tolist()}")
    fs.to_parquet(out / "freq_state.parquet", index=False)

    q = prompts[["qid", "state", "attribute", "question_type"]].merge(ent[["qid", "entity"]], on="qid")
    q = q.merge(fe, on="entity", how="left")
    q["tier"] = median_tier(q)
    q = q.merge(fs[["state", "log_freq"]].rename(columns={"log_freq": "state_log_freq"}), on="state")
    cols = [
        "qid",
        "entity",
        "wiki_exists_en",
        "wiki_exists_hi",
        "wiki_bytes_en",
        "wiki_pageviews_en",
        "pageviews_status",
        "corpus_count",
        "corpus_count_approx",
        "wiki_title_en",
        "wiki_disambig_en",
        "redirect_changed_concept",
        "entity_n_tokens",
        "entity_n_words",
        "log_freq",
        "log_freq_source",
        "log_freq_lenadj",
        "tier",
        "state_log_freq",
    ]
    q[cols].to_parquet(resolve(cfg["paths"]["freq"]), index=False)

    pv_needed = int(fe.wiki_exists_en.sum() + fs.wiki_exists_en.sum())
    pv_missing = int((fe.pageviews_status == "not_fetched").sum() + (fs.pageviews_status == "not_fetched").sum())
    meta = {
        "infinigram_index": cfg["frequency"]["infinigram_index"],
        "pageviews_year": cfg["frequency"]["pageviews_year"],
        "pageviews_mode": args.pageviews,
        "pageviews_complete": pv_missing == 0,
        "pageviews_needed": pv_needed,
        "pageviews_not_fetched": pv_missing,
        "n_entities": len(fe),
        "n_states": len(fs),
        "log_freq_primary": "corpus_count",
        "log_freq_fallback": "wiki_pageviews_en",
        "length_adjustment": {"model": "log1p(corpus_count) ~ log(entity_n_tokens)", **lenfit},
        "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "freq_meta.json").write_text(json.dumps(meta, indent=2))
    log.info(f"\nrun metadata: {meta}")
    log.info(
        f"\nHTTP: {client.misses} requests, {client.hits} cache hits, throttled (429/5xx): {client.throttled}, "
        f"not fetched (cache-only): {client.not_fetched}"
    )
    an = q[q.qid.isin(set(load_analysis_set().qid))]
    has_ent = an[an.entity.notna()]
    log.info(
        f"\nmissing rate per column (analysis set excl. ambiguous + leaks, {len(an)} questions; "
        f"{len(has_ent)} with an entity)"
    )
    for c in [
        "entity",
        "wiki_exists_en",
        "wiki_exists_hi",
        "wiki_bytes_en",
        "wiki_pageviews_en",
        "corpus_count",
        "log_freq",
        "tier",
    ]:
        base = an if c == "entity" else has_ent
        log.info(
            f"  {c:<20} {base[c].isna().mean():6.1%} missing"
            + (f"   (True: {base[c].eq(True).mean():.1%})" if c.startswith("wiki_exists") else "")
        )
    log.info(
        f"  log_freq_source: {has_ent.log_freq_source.value_counts(dropna=False).to_dict()} "
        f"(pageviews fallback rows: {(has_ent.log_freq_source == 'pageviews').sum()})"
    )
    rc = fe[fe.wiki_exists_en.astype(bool)]
    log.info(
        f"  redirect_changed_concept: {int(rc.redirect_changed_concept.sum())} of {len(rc)} entities with an en "
        f"article ({has_ent.redirect_changed_concept.eq(True).sum()} analysis questions)"
    )
    log.info("\nper distinct entity")
    log.info(
        f"  wiki_exists_en {fe.wiki_exists_en.mean():.1%}, disambiguation {fe.wiki_disambig_en.mean():.1%}, "
        f"wiki_exists_hi True {fe.wiki_exists_hi.eq(True).mean():.1%}, "
        f"corpus_count==0 {(fe.corpus_count == 0).mean():.1%}, approx {fe.corpus_count_approx.fillna(False).mean():.1%}"
    )
    log.info(f"  pageviews_status: {fe.pageviews_status.value_counts().to_dict()}")
    if meta["pageviews_complete"]:
        both = fe.dropna(subset=["corpus_count", "wiki_pageviews_en"])
        rho, p = spearmanr(both.corpus_count, both.wiki_pageviews_en)
        log.info(f"  Spearman(corpus_count, wiki_pageviews_en) = {rho:.3f} (p = {p:.2g}, n = {len(both)} entities)")
    else:
        log.info(
            f"  Spearman(corpus_count, wiki_pageviews_en): PENDING (pageviews incomplete: "
            f"{pv_missing}/{pv_needed} not fetched)"
        )

    med = an.groupby("state").log_freq.median().sort_values()
    st = fs.set_index("state")
    tab = pd.DataFrame(
        {
            "median_entity_log_freq": med,
            "n_with_score": an.groupby("state").log_freq.count(),
            "state_log_freq": st.log_freq,
            "state_corpus_count": st.corpus_count,
            "state_bytes": st.wiki_bytes_en,
        }
    ).loc[med.index]
    log.info(
        "\nmedian entity log_freq per state (analysis set), sorted\n" + tab.to_string(float_format=lambda x: f"{x:.2f}")
    )
    ne = ["Arunachal_Pradesh", "Assam", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Sikkim", "Tripura"]
    low, high = [*ne, "Bihar", "Jharkhand"], ["Delhi", "Maharashtra"]
    hmin = med[high].min()
    log.info("\nsanity: NE states + Bihar + Jharkhand vs Delhi, Maharashtra (entity median log_freq)")
    log.info(f"  Delhi {med['Delhi']:.2f}, Maharashtra {med['Maharashtra']:.2f}")
    for s in low:
        log.info(f"  {s:<20} {med[s]:.2f}  {'below both' if med[s] < hmin else 'NOT below both'}")
    log.info(f"  {sum(med[s] < hmin for s in low)}/{len(low)} below both")
    log.info(
        f"  state-article log_freq: low group median {fs.set_index('state').log_freq[low].median():.2f} "
        f"vs Delhi {st.log_freq['Delhi']:.2f}, Maharashtra {st.log_freq['Maharashtra']:.2f}"
    )


if __name__ == "__main__":
    main()
