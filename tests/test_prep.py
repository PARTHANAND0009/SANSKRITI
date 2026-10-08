"""Stage 0 logic on hand-written fixtures (not dataset rows)."""
import pandas as pd
import pytest

from data.prep import build, build_prompt, map_columns, match_gold, permute, write_quality_report


def test_template_exact():
    p = build_prompt(" Which X? ", ["a", "b ", " c", "d"])
    assert p == "Question: Which X?\nA. a\nB. b\nC. c\nD. d\nAnswer:"


def test_match_gold():
    assert match_gold("Foo", ["bar", " foo ", "baz", "qux"]) == (1, False)
    assert match_gold("foo", ["FOO", "bar", "foo", "x"]) == (0, True)
    assert match_gold("nope", ["a", "b", "c", "d"]) == (None, False)


def test_permute_deterministic_and_tracks_gold():
    opts = ["w", "x", "y", "z"]
    for q in range(50):
        qid = f"sk{q:05d}"
        p1, g1 = permute(opts, 2, qid, 1234)
        p2, g2 = permute(opts, 2, qid, 1234)
        assert (p1, g1) == (p2, g2)
        assert sorted(p1) == opts and p1[g1] == "y"
    # different qids give different orders at least sometimes
    assert len({tuple(permute(opts, 0, f"sk{q:05d}", 1234)[0]) for q in range(50)}) > 1


def test_map_columns_fails_loudly():
    with pytest.raises(SystemExit, match="cannot map"):
        map_columns(["foo", "bar"])


def _raw():
    return pd.DataFrame({
        "state": ["S1", "S2", "S3"], "attribute": ["food"] * 3, "question_type": ["t"] * 3,
        "question": ["q0?", "q1?", "q2?"],
        "option1": ["a", "a", "a"], "option2": ["b", "A ", "b"],
        "option3": ["c", "c", "c"], "option4": ["d", "d", "d"],
        "answer": ["c", "a", "zzz"],
    })


def test_build_and_report(tmp_path):
    kept, dropped = build(_raw(), 1234, {"m": None})
    assert list(kept.qid) == ["sk00000", "sk00001"] and list(dropped.qid) == ["sk00002"]
    assert kept.gold_idx.tolist() == [2, 0]
    assert kept.ambiguous_gold.tolist() == [False, True]
    assert kept.duplicate_options.tolist() == [False, True]
    for _, r in kept.iterrows():
        perm_opts = r.prompt_permuted.split("\n")[1:5]
        assert perm_opts[r.gold_idx_permuted][3:] == r.options[r.gold_idx]
    out = tmp_path / "dq.md"
    write_quality_report(kept, dropped, out, {"hf_id": "x", "revision": "y", "n_raw": 3})
    text = out.read_text()
    assert "(1 rows)" in text and "sk00002" in text and "sk00001" in text


def test_leak_rule():
    from data.prep import leak_rule

    assert leak_rule("Which state is famous for Eluru carpets Andhra?", "Andhra_Pradesh") == "gold_state_name"
    assert leak_rule("Which state is famous for Punjabi folk music?", "Punjab") == "gold_state_name"
    assert leak_rule("Which country is home to the Indian rhino?", "India") == "gold_country_name"
    assert leak_rule("Where is the Hemis Festival famous within Ladakh?", "Hemis") == "gold_text_in_stem"
    assert leak_rule("Which of the given regions is home to the Kumaoni?", "Kumaon") is None  # word-bounded
    assert leak_rule("Which state is famous for Bengali sweets?", "Odisha") is None
    assert leak_rule("Which state is famous for Jaapi?", "Assam") is None


def test_cyclic_rotations():
    from data.prep import rotate

    opts = ["w", "x", "y", "z"]
    for g in range(4):
        rots = [rotate(opts, g, k)[0] for k in range(4)]
        for k, r in enumerate(rots):
            assert r[k] == opts[g]                       # gold lands at letter k
            i = r.index("w")
            assert [r[(i + j) % 4] for j in range(4)] == opts  # cyclic order preserved
        assert len({tuple(r) for r in rots}) == 4        # four distinct orders
        assert opts in rots                              # the original order is one of them


def test_build_has_cyclic_prompts():
    kept, _ = build(_raw(), 1234, {"m": None})
    r = kept.iloc[0]
    for k in range(4):
        lines = r[f"prompt_cyc{k}"].split("\n")[1:5]
        assert lines[k][3:] == r.options[r.gold_idx] and r[f"gold_idx_cyc{k}"] == k
