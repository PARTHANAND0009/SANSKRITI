"""Option-token check: are " A".." D" / "A".."D" single tokens for each model?

`python -m crystal.tokens --write` records the working variant per model in
config/models.yaml (`space` preferred, since the prompt ends in "Answer:" and the
next token is " A"). Tokenizers that cannot be loaded stay `pending`.
"""
from __future__ import annotations

import argparse
import re

from crystal.io import ROOT, load_models_config

VARIANTS = {"space": [" A", " B", " C", " D"], "bare": ["A", "B", "C", "D"]}


def encode(tok, s: str) -> list[int]:
    return tok.encode(s, add_special_tokens=False)


def check_tokenizer(tok) -> dict:
    """{variant: [ids] if every letter is a single token (and distinct), else None}."""
    out = {}
    for name, strs in VARIANTS.items():
        ids = [encode(tok, s) for s in strs]
        ok = all(len(i) == 1 for i in ids) and len({i[0] for i in ids}) == 4
        out[name] = [i[0] for i in ids] if ok else None
    return out


def choose_variant(result: dict) -> str | None:
    for name in ("space", "bare"):
        if result.get(name) is not None:
            return name
    return None


def option_ids(tok, variant: str) -> list[int]:
    ids = check_tokenizer(tok).get(variant)
    if ids is None:
        raise ValueError(f"variant {variant!r} is not single-token for this tokenizer")
    return ids


def check_all(models: dict) -> dict:
    """{key: {'status': ok|pending|none, 'space': ids|None, 'bare': ids|None,
    'variant': str, 'error': str|None}}"""
    from transformers import AutoTokenizer

    out = {}
    for key, m in models.items():
        try:
            tok = AutoTokenizer.from_pretrained(m["id"])
        except Exception as e:
            out[key] = {"status": "pending", "space": None, "bare": None, "variant": "pending",
                        "error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}
            continue
        r = check_tokenizer(tok)
        v = choose_variant(r)
        out[key] = {"status": "ok" if v else "none", **r, "variant": v or "none", "error": None}
    return out


def write_variants(results: dict, path=None) -> None:
    """Rewrite option_token_variant lines in models.yaml in place (keeps comments)."""
    path = path or ROOT / "config" / "models.yaml"
    lines = path.read_text().splitlines(keepends=True)
    key = None
    for i, line in enumerate(lines):
        m = re.match(r"^  (\w+):\s*$", line)
        if m:
            key = m.group(1)
        elif (key in results and results[key]["status"] != "pending"
              and re.match(r"^    option_token_variant:", line)):
            # an unreachable tokenizer never overwrites an earlier result
            lines[i] = f"    option_token_variant: {results[key]['variant']}\n"
    path.write_text("".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="record variants in config/models.yaml")
    args = ap.parse_args(argv)
    res = check_all(load_models_config())
    print(f"{'model':<20} {'space':<28} {'bare':<28} variant")
    for k, r in res.items():
        print(f"{k:<20} {str(r['space']):<28} {str(r['bare']):<28} {r['variant']}"
              + (f"   ({r['error']})" if r["error"] else ""))
    if args.write:
        write_variants(res)
        print("wrote config/models.yaml")


if __name__ == "__main__":
    main()
