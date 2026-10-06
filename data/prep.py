"""Stage 0: build prompts.parquet from the SANSKRITI release (HF 13ari/Sanskriti, CC0).

Outputs
  data/processed/prompts.parquet   qid, state, attribute, question_type, stem,
                                   options, gold_idx, prompt, prompt_permuted,
                                   gold_idx_permuted, option_token_ids, ambiguous_gold
  data/processed/data_quality.md   every dropped and every ambiguous row, for the
                                   dataset authors

Rules
  - gold matching is on strip + lowercase text.
  - no option matches the answer field  -> row dropped (listed in data_quality.md)
  - more than one option matches        -> ambiguous_gold = True, kept; gold_idx is
                                           the first match. load_analysis_set()
                                           excludes these rows.
  - any two options identical           -> duplicate_options = True (superset of
                                           ambiguous_gold: also catches duplicated
                                           distractors). Flagged only, not excluded.
  - the stem reveals the gold answer    -> leaks_answer = True, leak_rule says how
                                           (see leak_rule()). load_analysis_set()
                                           excludes these unless include_leaks=True.
  - qid = "sk%05d" over the concatenated splits in their published order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

from crystal.io import load_models_config, load_run_config, resolve, set_seed

TEMPLATE = "Question: {stem}\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"
LETTERS = ("A", "B", "C", "D")

# Header names in the release. Each target field maps to the first candidate
# present; if none is present we stop and print the actual columns rather than
# guess. Confirm against the real release on first run.
COLUMN_CANDIDATES = {
    "state": ["state", "State"],
    "attribute": ["attribute", "Attribute", "category", "Category"],
    "question_type": ["question_type", "Question Type", "question type", "type"],
    "stem": ["question", "Question"],
    "answer": ["answer", "Answer"],
    "options": [
        ["option1", "option2", "option3", "option4"],
        ["Option 1", "Option 2", "Option 3", "Option 4"],
        ["option_a", "option_b", "option_c", "option_d"],
        ["A", "B", "C", "D"],
    ],
}


def norm(s) -> str:
    return str(s).strip().lower()


def build_prompt(stem: str, options) -> str:
    o = [str(x).strip() for x in options]
    if len(o) != 4:
        raise ValueError(f"expected 4 options, got {len(o)}")
    return TEMPLATE.format(stem=str(stem).strip(), o0=o[0], o1=o[1], o2=o[2], o3=o[3])


def match_gold(answer, options) -> tuple[int | None, bool]:
    """(gold_idx, ambiguous). gold_idx is None when no option matches."""
    a = norm(answer)
    hits = [i for i, o in enumerate(options) if norm(o) == a]
    if not hits:
        return None, False
    return hits[0], len(hits) > 1


# Distinctive name forms and demonyms / languages that identify one state.
# Generic first words ("West", "Uttar", "Madhya", "Tamil" as a first word of a
# state name is handled via the language list) are not used on their own;
# Telugu is omitted because it does not single out one state.
STATE_ALIASES = {
    "andaman and nicobar": ["andaman", "nicobar", "andamanese", "nicobarese"],
    "andhra pradesh": ["andhra"],
    "arunachal pradesh": ["arunachal"],
    "assam": ["assamese"],
    "bihar": ["bihari"],
    "chhattisgarh": ["chhattisgarhi"],
    "dadra and nagar haveli and daman and diu": ["dadra", "nagar haveli", "daman", "diu"],
    "goa": ["goan", "goans"],
    "gujarat": ["gujarati"],
    "haryana": ["haryanvi"],
    "himachal pradesh": ["himachal", "himachali"],
    "jammu kashmir": ["jammu and kashmir", "jammu & kashmir", "kashmir", "kashmiri", "jammu"],
    "jharkhand": ["jharkhandi"],
    "karnataka": ["kannada", "kannadiga"],
    "kerala": ["malayali", "malayalam", "keralite"],
    "ladakh": ["ladakhi"],
    "lakshadweep": [],
    "madhya pradesh": [],
    "maharashtra": ["marathi"],
    "manipur": ["manipuri"],
    "meghalaya": [],
    "mizoram": ["mizo"],
    "nagaland": ["naga", "nagas"],
    "odisha": ["odia", "oriya", "orissa", "odishan"],
    "puducherry": ["pondicherry"],
    "punjab": ["punjabi"],
    "rajasthan": ["rajasthani"],
    "sikkim": ["sikkimese"],
    "tamil nadu": ["tamil"],
    "telangana": [],
    "tripura": ["tripuri"],
    "uttar pradesh": [],
    "uttarakhand": ["uttaranchal", "garhwali", "kumaoni"],
    "west bengal": ["bengal", "bengali"],
    "delhi": [],
    "chandigarh": [],
}
COUNTRY_ALIASES = {"india": ["indian", "indians"]}


def _norm_text(s) -> str:
    return re.sub(r"\s+", " ", str(s).replace("_", " ").lower()).strip()


def _contains_word(text: str, phrase: str) -> bool:
    return bool(phrase) and re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def leak_rule(stem: str, gold: str) -> str | None:
    """How the stem reveals the gold option, or None.

    gold_text_in_stem   the gold option text itself occurs in the stem (word-bounded)
    gold_state_name     gold is a state/UT and a distinctive name form, demonym or
                        language of it occurs in the stem
    gold_country_name   gold is India and "Indian" occurs in the stem
    """
    s, g = _norm_text(stem), _norm_text(gold)
    if len(g) >= 3 and _contains_word(s, g):
        return "gold_text_in_stem"
    if g in STATE_ALIASES and any(_contains_word(s, a) for a in STATE_ALIASES[g]):
        return "gold_state_name"
    if g in COUNTRY_ALIASES and any(_contains_word(s, a) for a in COUNTRY_ALIASES[g]):
        return "gold_country_name"
    return None


def has_duplicate_options(options) -> bool:
    return len({norm(o) for o in options}) < len(options)


def qid_seed(qid: str, base_seed: int) -> int:
    h = hashlib.sha256(f"{base_seed}:{qid}".encode()).hexdigest()
    return int(h[:16], 16)


def permute(options, gold_idx: int, qid: str, base_seed: int):
    """Shuffle options with a fixed per-qid seed. Returns (options_perm, gold_idx_perm)."""
    order = list(range(len(options)))
    random.Random(qid_seed(qid, base_seed)).shuffle(order)
    return [options[i] for i in order], order.index(gold_idx)


def map_columns(columns) -> dict:
    cols = set(columns)
    out, missing = {}, []
    for field, cands in COLUMN_CANDIDATES.items():
        hit = next((c for c in cands if (set(c) <= cols if isinstance(c, list) else c in cols)), None)
        if hit is None:
            missing.append(field)
        out[field] = hit
    if missing:
        raise SystemExit(
            f"cannot map fields {missing} from dataset columns {sorted(cols)}; "
            "update COLUMN_CANDIDATES in data/prep.py after checking the release"
        )
    return out


def load_raw(hf_id: str, revision: str, raw_dir: Path) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(hf_id, revision=revision, cache_dir=str(raw_dir / "hf"))
    frames = []
    for split in ds:  # published order
        df = ds[split].to_pandas()
        df["_split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def option_token_ids_by_model(models: dict) -> dict:
    """{model_key: [id_A, id_B, id_C, id_D] or None}, from option_token_variant.
    Models whose variant is still 'pending' get None (tokenizer not checked)."""
    from crystal.tokens import option_ids

    out = {}
    for key, m in models.items():
        variant = m.get("option_token_variant", "pending")
        if variant == "pending":
            out[key] = None
            continue
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(m["id"])
        out[key] = option_ids(tok, variant)
    return out


def build(raw: pd.DataFrame, base_seed: int, tok_ids: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (kept rows, dropped rows)."""
    cm = map_columns(raw.columns)
    rows, dropped = [], []
    tok_json = json.dumps(tok_ids, sort_keys=True)
    for i, r in raw.reset_index(drop=True).iterrows():
        qid = f"sk{i:05d}"
        options = [str(r[c]).strip() for c in cm["options"]]
        base = {
            "qid": qid,
            "state": str(r[cm["state"]]).strip(),
            "attribute": str(r[cm["attribute"]]).strip(),
            "question_type": str(r[cm["question_type"]]).strip(),
            "stem": str(r[cm["stem"]]).strip(),
            "options": options,
        }
        gold_idx, ambiguous = match_gold(r[cm["answer"]], options)
        if gold_idx is None:
            dropped.append({**base, "answer": str(r[cm["answer"]])})
            continue
        opts_p, gold_p = permute(options, gold_idx, qid, base_seed)
        rows.append(
            {
                **base,
                "answer": str(r[cm["answer"]]),
                "gold_idx": gold_idx,
                "prompt": build_prompt(base["stem"], options),
                "prompt_permuted": build_prompt(base["stem"], opts_p),
                "gold_idx_permuted": gold_p,
                "option_token_ids": tok_json,
                "ambiguous_gold": ambiguous,
                "duplicate_options": has_duplicate_options(options),
                "leak_rule": leak_rule(base["stem"], options[gold_idx]),
            }
        )
    kept = pd.DataFrame(rows)
    if len(kept):
        kept["leaks_answer"] = kept["leak_rule"].notna()
    return kept, pd.DataFrame(dropped)


def _md_cell(x) -> str:
    if isinstance(x, (list, tuple)):
        x = "<br>".join(f"{LETTERS[i]}. {v}" for i, v in enumerate(x))
    return str(x).replace("|", "\\|").replace("\n", " ")


def _md_table(df: pd.DataFrame, cols) -> str:
    if df.empty:
        return "_none_\n"
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_md_cell(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _leak_summary(kept: pd.DataFrame) -> str:
    if kept.empty or not kept["leaks_answer"].any():
        return "_none_\n"
    tab = kept[kept["leaks_answer"]].groupby(["question_type", "leak_rule"]).size().unstack(fill_value=0)
    cols = list(tab.columns)
    lines = ["| question_type | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for qt, r in tab.iterrows():
        lines.append(f"| {qt} | " + " | ".join(str(int(r[c])) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_quality_report(kept: pd.DataFrame, dropped: pd.DataFrame, path: Path, meta: dict) -> None:
    cols = ["qid", "state", "attribute", "stem", "answer", "options"]
    amb = kept[kept["ambiguous_gold"]] if len(kept) else kept
    dup = kept[kept["duplicate_options"] & ~kept["ambiguous_gold"]] if len(kept) else kept
    text = f"""# SANSKRITI data quality notes

Source: `{meta['hf_id']}` (revision `{meta['revision']}`), {meta['n_raw']} rows.
`qid` is the 0-based row position over the published splits, formatted `sk00000`.
Matching compares the answer field with each option after trimming whitespace and
lowercasing.

## 1. Answer matches none of the four options ({len(dropped)} rows)

These rows were dropped from our analysis.

{_md_table(dropped, cols)}
## 2. Answer matches more than one option ({len(amb)} rows)

The same text appears in two or more options, so the gold letter is not
determined. These rows are kept in our files but excluded from analysis.

{_md_table(amb, cols)}
## 3. Two options identical, gold still unique ({len(dup)} rows)

A distractor appears twice, so the question effectively has three distinct
choices. Kept and flagged (`duplicate_options`); not excluded.

{_md_table(dup, cols)}
## 4. Stem reveals the answer ({int(kept["leaks_answer"].sum()) if len(kept) else 0} rows)

The gold option, or a name form / demonym of the gold state, appears in the
question text. Kept and flagged (`leaks_answer`); excluded from our depth
analysis by default. Full list: `leaks.csv`.

{_leak_summary(kept)}"""
    path.write_text(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--revision", default=None)
    args = ap.parse_args(argv)

    cfg = load_run_config()
    set_seed(cfg["seed"])
    hf_id = cfg["dataset"]["hf_id"]
    revision = args.revision or cfg["dataset"]["revision"]
    raw_dir, out_dir = resolve(cfg["paths"]["raw"]), resolve(cfg["paths"]["processed"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw = load_raw(hf_id, revision, raw_dir)
    except Exception as e:  # network / auth: stop, don't fall back
        sys.exit(f"cannot load {hf_id}@{revision}: {type(e).__name__}: {e}")

    tok_ids = option_token_ids_by_model(load_models_config())
    kept, dropped = build(raw, cfg["prep"]["permute_seed"], tok_ids)
    kept.drop(columns=["answer"]).to_parquet(resolve(cfg["paths"]["prompts"]), index=False)
    leaks = kept[kept["leaks_answer"]].assign(gold=lambda d: [o[i] for o, i in zip(d.options, d.gold_idx)])
    leaks[["qid", "state", "attribute", "question_type", "leak_rule", "stem", "gold"]].to_csv(
        out_dir / "leaks.csv", index=False)
    write_quality_report(
        kept, dropped, out_dir / "data_quality.md",
        {"hf_id": hf_id, "revision": revision, "n_raw": len(raw)},
    )

    n_amb = int(kept["ambiguous_gold"].sum())
    print(f"raw rows: {len(raw)}  kept: {len(kept)}  dropped (no match): {len(dropped)}  "
          f"ambiguous_gold: {n_amb}  analysis set: {len(kept) - n_amb}  "
          f"duplicate_options (incl. ambiguous): {int(kept['duplicate_options'].sum())}  "
          f"leaks_answer: {int(kept['leaks_answer'].sum())}")
    pending = [k for k, v in tok_ids.items() if v is None]
    if pending:
        print(f"option_token_ids pending for: {pending}")
    for col in ("state", "attribute", "question_type"):
        print(f"\n== counts by {col} (kept rows)")
        print(kept[col].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
