"""Option-token checks.

The per-model tests load each real tokenizer (small download, no weights). If a
tokenizer is unreachable (network policy, gated without token) the test is
skipped and the model stays `pending` in config/models.yaml.
"""
import pytest

from crystal.io import load_models_config
from crystal.tokens import VARIANTS, check_tokenizer, choose_variant
from tests.fixtures import char_tokenizer

MODELS = load_models_config()


class _FakeTok:
    def __init__(self, table):
        self.table = table

    def encode(self, s, add_special_tokens=False):
        return self.table[s]


def test_check_logic():
    t = _FakeTok({" A": [10], " B": [11], " C": [12], " D": [13],
                  "A": [1], "B": [2], "C": [3, 4], "D": [5]})
    r = check_tokenizer(t)
    assert r == {"space": [10, 11, 12, 13], "bare": None}
    assert choose_variant(r) == "space"
    t = _FakeTok({**{s: [7, 8] for s in VARIANTS["space"]}, "A": [1], "B": [2], "C": [3], "D": [4]})
    assert choose_variant(check_tokenizer(t)) == "bare"


def test_char_tokenizer_fixture():
    r = check_tokenizer(char_tokenizer())
    assert r["bare"] is not None and r["space"] is None


@pytest.mark.network
@pytest.mark.parametrize("key", sorted(MODELS))
def test_model_option_tokens(key):
    from transformers import AutoTokenizer

    m = MODELS[key]
    try:
        tok = AutoTokenizer.from_pretrained(m["id"])
    except Exception as e:
        pytest.skip(f"{m['id']} tokenizer unavailable -> pending ({type(e).__name__})")
    r = check_tokenizer(tok)
    v = choose_variant(r)
    assert v is not None, f"{key}: neither ' A' nor 'A' is single-token: {r}"
    recorded = m["option_token_variant"]
    assert recorded in (v, "pending"), f"{key}: models.yaml says {recorded}, tokenizer says {v}"
    if recorded == "pending":
        pytest.skip(f"{key}: works with {v!r}; run `make tokens` to record it")
