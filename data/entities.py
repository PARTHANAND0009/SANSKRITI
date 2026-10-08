"""Stage 1a: deterministic entity extraction (no LLM).

SANSKRITI instantiates each seed entity into a fixed set of stem templates, so for most
rows the entity is read off the template. Rules, in priority order:

  template (high)  entity in the stem (with its character span) or, for location
                   templates, the gold option
  quoted (medium)  the first double-quoted name in a free-form stem
  lexicon (low)    the longest template-extracted entity of the same state that occurs
                   word-bounded in the stem

Outputs in data/processed/: entities.parquet, entity_audit.csv (rows to hand-check),
swap_pool.parquet (patching swap targets), templates.md (coverage and examples).
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from crystal.io import load_analysis_set, load_prompts, load_run_config, resolve, set_seed
from crystal.log import setup_logging
from crystal.text import WS, word_pattern

log = logging.getLogger(__name__)


def _t(pattern: str) -> re.Pattern:
    """Template literal whose single spaces match any run of whitespace."""
    return re.compile(pattern.replace(" ", WS))


# (rule, question types, regex). Group E is the entity in the stem; group L is a location,
# in which case the entity is the answer.
TEMPLATES = [
    ("assoc_regions_home_to", {"Association"}, _t(r"^Which of the given regions is home to the (?P<E>.+?)\s*\?$")),
    ("assoc_where_famous", {"Association"}, _t(r"^Where is the (?P<E>.+?) famous(?: within (?P<S>.+?))?\s*\?$")),
    (
        "assoc_or_country_associated_to",
        {"Association", "Country Prediction"},
        _t(r"^(?:The )?(?P<E>.+?) is associated to which (?:region|country|state)(?: of (?P<S>.+?))?\s*\?$"),
    ),
    ("country_home_to", {"Country Prediction"}, _t(r"^Which country is the home to (?P<E>.+?)\s*\?$")),
    ("country_famous_for", {"Country Prediction"}, _t(r"^Which country is famous for the (?P<E>.+?)\s*\?$")),
    ("ga_which_one_belongs", {"General Awareness"}, _t(r"^Which one belongs to (?P<L>.+?)\s*\?$")),
    (
        "ga_closely_associated",
        {"General Awareness"},
        _t(r"^According to you, which of the following is closely associated to (?P<L>.+?)\s*\??$"),
    ),
    (
        "state_houses",
        {"State Prediction"},
        _t(r"^According to you, which of the following states houses the (?P<E>.+?)\s*\?$"),
    ),
    (
        "state_options_associated",
        {"State Prediction"},
        _t(r"^Which of the states given in the options is associated to (?P<E>.+?)\s*\?$"),
    ),
    ("state_famous_for", {"State Prediction"}, _t(r"^Which state is famous for (?P<E>.+?)\s*\?$")),
]

# First words too generic to identify a state on their own.
_GENERIC_FIRST = {"west", "uttar", "tamil", "madhya", "jammu", "dadra", "andaman", "himachal", "arunachal"}


def state_name_regex(states) -> re.Pattern:
    forms = set()
    for s in states:
        full = s.replace("_", " ")
        forms.add(full)
        first = full.split()[0]
        if len(first) >= 5 and first.lower() not in _GENERIC_FIRST:
            forms.add(first)
    forms |= {
        "Jammu and Kashmir",
        "Kashmir",
        "Andaman",
        "Nicobar",
        "Daman",
        "Diu",
        "Dadra",
        "Nagar Haveli",
        "Arunachal",
        "Himachal",
        "Madhya Pradesh",
        "Uttar Pradesh",
        "Tamil Nadu",
        "West Bengal",
    }
    alt = "|".join(sorted((re.escape(f).replace(r"\ ", WS) for f in forms), key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alt})(?!\w)", re.IGNORECASE)


QUOTED = re.compile(r"[\"“](?P<E>[^\"”]{2,80}?)[,.]?[\"”]")
MIN_LEXICON_LEN = 4
DANGLING_TAIL = r"(?i)\s(?:state|to|in|of|the|for|houses|celebrated in)\s*$"


def clean(s: str) -> str:
    return re.sub(WS, " ", s).strip().strip(",.;:")


def match_template(stem: str, qtype: str):
    """(rule, kind 'E'|'L', match) for the first template that fits, else None."""
    for name, types, rx in TEMPLATES:
        if qtype not in types:
            continue
        m = rx.match(stem)
        if m:
            return name, ("E" if m.groupdict().get("E") is not None else "L"), m
    return None


def stem_span(stem: str, m: re.Match, group="E"):
    """Character span of the matched group after trimming whitespace and trailing punctuation."""
    s, e = m.span(group)
    raw = stem[s:e]
    lead = len(raw) - len(raw.lstrip())
    stripped = raw.strip()
    trail = len(stripped) - len(stripped.rstrip(",.;:"))
    return s + lead, s + lead + len(stripped) - trail


def extract_row(stem: str, qtype: str, gold: str):
    """Entity fields from the template and quoted rules (the lexicon rule needs all rows)."""
    hit = match_template(stem, qtype)
    if hit:
        rule, kind, m = hit
        if kind == "E":
            s, e = stem_span(stem, m)
            return dict(
                entity=stem[s:e],
                entity_source="stem",
                span_start=s,
                span_end=e,
                extraction_rule=rule,
                confidence="high",
            )
        return dict(
            entity=clean(gold),
            entity_source="answer",
            span_start=None,
            span_end=None,
            extraction_rule=rule,
            confidence="high",
        )
    q = QUOTED.search(stem)
    if q:
        s, e = stem_span(stem, q)
        return dict(
            entity=stem[s:e],
            entity_source="stem",
            span_start=s,
            span_end=e,
            extraction_rule="quoted",
            confidence="medium",
        )
    return None


def lexicon_match(stem: str, lexicon: list[str]):
    """Longest lexicon entry occurring word-bounded in the stem: (entity, start, end) or None."""
    best = None
    for ent in lexicon:
        m = re.search(word_pattern(ent, flexible_space=True), stem)
        if m and (best is None or len(ent) > len(best[0])):
            best = (ent, m.start(), m.end())
    return best


def extract(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        out = extract_row(r.stem, r.question_type, r.options[r.gold_idx])
        rows.append({"qid": r.qid, **(out or {})})
    ent = pd.DataFrame(rows)
    ent = df[["qid", "state", "stem"]].merge(ent, on="qid")

    lex = (
        ent[(ent.extraction_rule.isin([t[0] for t in TEMPLATES])) & (ent.entity_source == "stem")]
        .groupby("state")
        .entity.agg(lambda s: sorted({x for x in s if len(x) >= MIN_LEXICON_LEN}))
    )
    todo = ent.entity.isna()
    for i in ent.index[todo]:
        m = lexicon_match(ent.at[i, "stem"], lex.get(ent.at[i, "state"], []))
        if m:
            ent.loc[i, ["entity", "entity_source", "span_start", "span_end", "extraction_rule", "confidence"]] = [
                m[0],
                "stem",
                m[1],
                m[2],
                "lexicon",
                "low",
            ]
    ent["extraction_rule"] = ent.extraction_rule.fillna("none")
    for c in ("span_start", "span_end"):
        ent[c] = ent[c].astype("Int64")
    sp = ent.span_start.notna()
    bad = [
        i
        for i in ent.index[sp]
        if clean(ent.at[i, "stem"][ent.at[i, "span_start"] : ent.at[i, "span_end"]]) != clean(ent.at[i, "entity"])
    ]
    if bad:
        raise AssertionError(
            f"{len(bad)} spans do not reproduce their entity, e.g. {ent.loc[bad[:3]].to_dict('records')}"
        )
    rx = state_name_regex(df.state.unique())
    ent["entity_mentions_state"] = ent.entity.map(lambda e: bool(rx.search(e)) if isinstance(e, str) else False)
    return ent[
        [
            "qid",
            "entity",
            "entity_source",
            "span_start",
            "span_end",
            "extraction_rule",
            "confidence",
            "entity_mentions_state",
        ]
    ]


def template_report(df: pd.DataFrame, ent: pd.DataFrame, n_examples=3, seed=0) -> str:
    j = df.merge(ent, on="qid")
    lines = [
        "# Stem templates by question_type",
        "",
        f"{len(df)} rows (stage 0 output). Coverage = share of the question type's rows.",
        "",
    ]
    for qt, g in j.groupby("question_type"):
        lines += [f"## {qt} ({len(g)} rows)", "", "| rule | rows | coverage |", "|---|---|---|"]
        vc = g.extraction_rule.value_counts()
        for rule, n in vc.items():
            lines.append(f"| {rule} | {n} | {n / len(g):.1%} |")
        lines.append("")
        for rule in vc.index:
            pat = next((t[2].pattern.replace(WS, " ") for t in TEMPLATES if t[0] == rule), None)
            lines.append(f"**{rule}**" + (f"  `{pat}`" if pat else ""))
            for r in (
                g[g.extraction_rule == rule]
                .sample(min(n_examples, (g.extraction_rule == rule).sum()), random_state=seed)
                .itertuples()
            ):
                lines.append(
                    f"- {r.stem}  →  "
                    + (f"entity: `{r.entity}` ({r.entity_source})" if isinstance(r.entity, str) else "no entity")
                )
            lines.append("")
    return "\n".join(lines)


def swap_pool(df: pd.DataFrame, ent: pd.DataFrame):
    """Swap targets for patching: per attribute, distinct entities with a stem span and their state.

    Returns:
        (pool, per-question count of usable candidates, for questions with a span)
    """
    j = df[["qid", "state", "attribute"]].merge(ent, on="qid")
    sp = j[j.span_start.notna()].copy()
    sp["key"] = sp.entity.str.lower()
    pool = (
        sp.groupby(["attribute", "key", "state"])
        .agg(entity=("entity", "first"), n_questions=("qid", "size"))
        .reset_index()
    )
    # An entity seen under several states, naming a state, or cut from a stem with trailing
    # words ("Sohrai painting state") would not cleanly change the answer; kept but not used.
    pool["n_states_for_entity"] = pool.groupby(["attribute", "key"]).state.transform("nunique")
    pool = pool.merge(
        sp.groupby(["attribute", "key", "state"]).entity_mentions_state.first().reset_index(),
        on=["attribute", "key", "state"],
    )
    pool["dangling_tail"] = pool.entity.str.contains(DANGLING_TAIL)
    pool["swap_ok"] = (pool.n_states_for_entity == 1) & ~pool.entity_mentions_state & ~pool.dangling_tail
    clean_pool = pool[pool.swap_ok]
    per_attr_state = clean_pool.groupby(["attribute", "state"]).size()
    per_attr = clean_pool.groupby("attribute").size()

    def n_candidates(r):
        total = per_attr.get(r.attribute, 0)
        same_state = per_attr_state.get((r.attribute, r.state), 0)
        return int(total - same_state)

    sp["n_swap_candidates"] = sp.apply(n_candidates, axis=1)
    return pool.drop(columns="key"), sp[["qid", "attribute", "state", "n_swap_candidates"]]


def main():
    setup_logging()
    cfg = load_run_config()
    set_seed(cfg["seed"])
    out = resolve(cfg["paths"]["processed"])
    df = load_prompts()
    ent = extract(df)
    ent.to_parquet(resolve(cfg["paths"]["entities"]), index=False)
    (out / "templates.md").write_text(template_report(df, ent))

    j = df[["qid", "question_type", "state", "attribute", "stem", "options", "gold_idx"]].merge(ent, on="qid")
    audit = j.sample(cfg["sample_sizes"]["entity_audit"], random_state=cfg["seed"]).sort_values("qid")
    audit = audit.assign(
        gold=audit.apply(lambda r: r.options[r.gold_idx], axis=1),
        span_text=audit.apply(lambda r: r.stem[r.span_start : r.span_end] if pd.notna(r.span_start) else "", axis=1),
        options=audit.options.map(lambda o: " | ".join(o)),
        entity_correct="",
        span_correct="",
        notes="",
    )[
        [
            "qid",
            "question_type",
            "state",
            "attribute",
            "stem",
            "options",
            "gold",
            "entity",
            "entity_source",
            "span_start",
            "span_end",
            "span_text",
            "extraction_rule",
            "confidence",
            "entity_correct",
            "span_correct",
            "notes",
        ]
    ]
    audit.to_csv(out / "entity_audit.csv", index=False)

    pool, cand = swap_pool(df, ent)
    pool.to_parquet(out / "swap_pool.parquet", index=False)

    an_qids = set(load_analysis_set().qid)
    an = j[j.qid.isin(an_qids)]
    log.info(f"rows: {len(j)} (analysis set {len(an)}: ambiguous_gold and leaks_answer excluded)")
    cov = an.groupby("question_type").agg(
        n=("qid", "size"),
        has_entity=("entity", lambda s: s.notna().mean()),
        has_stem_span=("span_start", lambda s: s.notna().mean()),
    )
    cov.loc["ALL"] = [len(an), an.entity.notna().mean(), an.span_start.notna().mean()]
    log.info("\ncoverage (analysis set)\n" + cov.to_string(float_format=lambda x: f"{x:.3f}"))
    by_rule = an.groupby(["extraction_rule", "confidence"], dropna=False).size()
    log.info("\nextraction_rule x confidence\n" + by_rule.to_string())
    cand = cand.merge(df[["qid", "question_type"]], on="qid")
    cand = cand[cand.qid.isin(an_qids)]
    has_cand = cand.n_swap_candidates > 0
    log.info(
        f"\nswap pool: {len(pool)} (attribute, entity, state) entries, "
        f"{(~pool.swap_ok).sum()} not usable as candidates ({pool.dangling_tail.sum()} dangling tail, "
        f"{(pool.n_states_for_entity > 1).sum()} seen under >1 state, "
        f"{pool.entity_mentions_state.sum()} name a state)"
    )
    log.info(
        f"entities naming a state (entity_mentions_state): {an.entity_mentions_state.sum()} rows; "
        f"in State Prediction: {an[an.question_type == 'State Prediction'].entity_mentions_state.sum()}"
    )
    log.info(
        f"questions with a stem span: {len(cand)}; with >=1 same-attribute different-state candidate: "
        f"{has_cand.sum()} ({has_cand.mean():.1%} of spans, {has_cand.sum() / len(an):.1%} of analysis set)"
    )
    by_type = cand.groupby("question_type").n_swap_candidates.agg(
        lambda s: f"{(s > 0).mean():.1%} have >=1, median {int(s.median())}"
    )
    log.info(by_type.to_string())
    per_attr = cand.groupby("attribute").n_swap_candidates.agg(["size", "min", "median", "max"])
    per_attr.columns = ["questions_with_span", "min", "median", "max"]
    log.info("\nswap candidates per attribute (questions with a stem span)\n" + per_attr.to_string())

    overall_e, overall_s = an.entity.notna().mean(), an.span_start.notna().mean()
    st = an.groupby("state").agg(
        n=("qid", "size"),
        entity_cov=("entity", lambda s: s.notna().mean()),
        span_cov=("span_start", lambda s: s.notna().mean()),
    )
    st["flag"] = np.where(
        (st.entity_cov < overall_e - 0.15) | (st.span_cov < overall_s - 0.15), ">15 pts below overall", ""
    )
    table = st.sort_values("span_cov").to_string(
        formatters={"entity_cov": "{:.1%}".format, "span_cov": "{:.1%}".format}
    )
    log.info(
        f"\ncoverage per state (analysis set), sorted by span coverage; overall entity "
        f"{overall_e:.1%}, span {overall_s:.1%}\n{table}"
    )


if __name__ == "__main__":
    main()
