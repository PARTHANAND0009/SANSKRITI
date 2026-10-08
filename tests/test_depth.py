"""Depth metrics on synthetic probabilities (no model)."""
import numpy as np
import pandas as pd

from analysis.depth import aggregate, reliability, soft_layer, spearman_brown, table_from_probs
from data.prep import option_order

L = 10
SEED = 1234


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _prompts(n, rng):
    return pd.DataFrame({"qid": [f"q{i:03d}" for i in range(n)], "gold_idx": rng.integers(0, 4, n)}).set_index("qid")


def _variant_probs(prompts, variant, cryst, bias_letter=1, bias=4.0):
    """Content logits: gold gets +6 from layer `cryst` on. Every layer adds a
    letter bias toward `bias_letter` (every layer, letter space), so raw l_star is fooled."""
    n = len(prompts)
    P = np.empty((n, L + 1, 4))
    for i, (qid, g) in enumerate(zip(prompts.index, prompts.gold_idx)):
        order = option_order(variant, int(g), qid, SEED)          # letter j shows content order[j]
        content = np.zeros((L + 1, 4))
        content[:cryst[i], (g + 1) % 4] += 1.0                  # a distractor leads before cryst
        content[cryst[i]:, g] += 6.0
        letters = content[:, order]
        letters[:, bias_letter] += bias                     # constant letter prior at every layer
        P[i] = _softmax(letters)
    return P


def test_soft_layer():
    p = np.array([[0.25, 0.25, 0.25, 0.9, 0.9], [0.25, 0.5, 0.75, 1.0, 1.0], [0.5, 0.5, 0.4, 0.4, 0.4]])
    s = soft_layer(p)
    assert s[0] == 3 and s[1] == 2 and np.isnan(s[2])


def test_aggregate_cancels_letter_bias():
    rng = np.random.default_rng(0)
    pr = _prompts(40, rng)
    cryst = rng.integers(3, 7, len(pr))
    variants = ["cyc0", "cyc1", "cyc2", "cyc3"]
    tables = {v: table_from_probs(pr.index.tolist(), _variant_probs(pr, v, cryst), pr.gold_idx.to_numpy() * 0 + int(v[3]))
              for v in variants}
    # raw single-order l_star is fooled for gold-at-B (bias letter) rows in cyc1: crystallizes at 0
    assert (tables["cyc1"].l_star.astype(float) == 0).all()
    agg = aggregate(tables, pr, SEED)
    assert agg.correct_cyc_final.all()
    np.testing.assert_array_equal(agg.l_star_cyc.astype(int).to_numpy(), cryst)
    np.testing.assert_array_equal(agg.d_margin.astype(int).to_numpy(), cryst)
    np.testing.assert_allclose(agg.d_soft.to_numpy(), cryst, atol=0.5)
    assert (agg.d_frac_l_star_cyc.astype(float) == cryst / L).all()


def test_aggregate_with_original_and_permuted_orders():
    rng = np.random.default_rng(1)
    pr = _prompts(30, rng)
    cryst = rng.integers(3, 7, len(pr))
    perm_gold = np.array([option_order("perm", int(g), q, SEED).index(int(g)) for q, g in zip(pr.index, pr.gold_idx)])
    tables = {"prompt": table_from_probs(pr.index.tolist(), _variant_probs(pr, "prompt", cryst, bias=0.0), pr.gold_idx.to_numpy()),
              "perm": table_from_probs(pr.index.tolist(), _variant_probs(pr, "perm", cryst, bias=0.0), perm_gold)}
    agg = aggregate(tables, pr, SEED)
    np.testing.assert_array_equal(agg.l_star_cyc.astype(int).to_numpy(), cryst)


def test_reliability_identical_orders():
    rng = np.random.default_rng(2)
    pr = _prompts(50, rng)
    cryst = rng.integers(2, 8, len(pr))
    t = table_from_probs(pr.index.tolist(), _variant_probs(pr, "prompt", cryst, bias=0.0), pr.gold_idx.to_numpy())
    r = reliability(t, t).set_index("metric")
    assert (r.within_1_layer == 1).all() and (r.spearman > 0.99).all()
    assert abs(spearman_brown(0.5, 4) - 0.8) < 1e-12


def test_single_cyclic_table_does_not_self_calibrate():
    rng = np.random.default_rng(3)
    pr = _prompts(20, rng)
    t = table_from_probs(pr.index.tolist(), _variant_probs(pr, "cyc1", rng.integers(3, 7, 20)), np.ones(20, int))
    assert t.l_star_cal.isna().all() and np.isnan(t.d_soft_cal).all()


def test_split_half_runs_and_is_high_for_noiseless_data():
    from analysis.depth import split_half

    rng = np.random.default_rng(4)
    pr = _prompts(60, rng)
    cryst = rng.integers(2, 9, len(pr))
    tables = {v: table_from_probs(pr.index.tolist(), _variant_probs(pr, v, cryst, bias=0.0), np.full(len(pr), int(v[3])))
              for v in ["cyc0", "cyc1", "cyc2", "cyc3"]}
    r = split_half(tables, pr, SEED).set_index("metric")
    assert r.loc["d_soft", "spearman"] > 0.95 and r.loc["l_star_cyc", "within_1_layer"] == 1.0


def test_orig_from_cyclic_matches_original_order():
    from analysis.depth import orig_from_cyclic

    rng = np.random.default_rng(5)
    pr = _prompts(30, rng)
    cryst = rng.integers(2, 8, len(pr))
    cyc = {v: table_from_probs(pr.index.tolist(), _variant_probs(pr, v, cryst), np.full(len(pr), int(v[3])))
           for v in ["cyc0", "cyc1", "cyc2", "cyc3"]}
    orig = table_from_probs(pr.index.tolist(), _variant_probs(pr, "prompt", cryst), pr.gold_idx.to_numpy())
    got = orig_from_cyclic(cyc, pr).set_index("qid").loc[orig.qid]
    np.testing.assert_allclose(np.stack(got.option_probs_by_layer.to_numpy()), np.stack(orig.option_probs_by_layer.to_numpy()))
