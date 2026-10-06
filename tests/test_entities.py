"""Stage 1a rules on hand-written stems (template forms copied from the release)."""
import pandas as pd

from data.entities import extract, extract_row, lexicon_match, state_name_regex


def test_stem_templates_and_spans():
    s = "Which of the given regions is home to the  Gale skirts ?"
    r = extract_row(s, "Association", "Arunachal_Pradesh")
    assert r["entity"] == "Gale skirts" and s[r["span_start"]:r["span_end"]] == "Gale skirts"
    assert r["extraction_rule"] == "assoc_regions_home_to" and r["confidence"] == "high"
    r = extract_row("The Muga silk is associated to which country?", "Country Prediction", "India")
    assert r["entity"] == "Muga silk"
    r = extract_row("Which of the states given in the options is associated to Jaapi?", "State Prediction", "Assam")
    assert r["entity"] == "Jaapi"


def test_answer_templates_have_no_span():
    r = extract_row("Which one belongs to Ladakh?", "General Awareness", " Hemis Festival ")
    assert r["entity"] == "Hemis Festival" and r["entity_source"] == "answer" and r["span_start"] is None


def test_template_restricted_to_question_type():
    # same text under a type the template does not belong to falls through to no match
    assert extract_row("Which one belongs to Ladakh?", "State Prediction", "Ladakh") is None


def test_quoted_and_lexicon():
    r = extract_row('The "Nagaji Fair," a grand gathering, is held in which country?', "Country Prediction", "India")
    assert r["entity"] == "Nagaji Fair" and r["confidence"] == "medium"
    assert lexicon_match("Known for the Hundru Falls, one of", ["Hundru Falls", "Falls"])[0] == "Hundru Falls"
    assert lexicon_match("Hundru Fallsx", ["Hundru Falls"]) is None


def test_extract_end_to_end_and_state_flag():
    df = pd.DataFrame({
        "qid": ["a", "b", "c"], "state": ["Jharkhand", "Jharkhand", "Andhra_Pradesh"],
        "question_type": ["State Prediction", "Association", "State Prediction"],
        "stem": ["Which state is famous for Hundru Falls?", "Tourists like the Hundru Falls in winter.",
                 "Which state is famous for Eluru carpets Andhra?"],
        "options": [["Jharkhand", "b", "c", "d"]] * 3, "gold_idx": [0, 0, 0],
    })
    e = extract(df).set_index("qid")
    assert e.loc["b", "extraction_rule"] == "lexicon" and e.loc["b", "entity"] == "Hundru Falls"
    assert not e.loc["a", "entity_mentions_state"] and e.loc["c", "entity_mentions_state"]
    assert state_name_regex(["West_Bengal"]).search("west  bengal sweets")


def test_redirect_changed_concept():
    from data.frequency import redirect_changed_concept as r

    assert r("Bihu dances", "Bihu dance") is False
    assert r("gale skirts", "Gale skirt") is False
    assert r("Ghoomar.", "Ghoomar") is False
    assert r("Pottery", "Potteries") is False
    assert r("Bastar_district", "Bastar district") is False
    assert r("Hemis Festival", "Hemis Monastery") is True
    assert r("Kalbelia", "Kalbelia dance") is True
    assert r("anything", None) is None


def test_log_freq_and_tier():
    import numpy as np
    import pandas as pd

    from data.frequency import add_log_freq, median_tier

    df = pd.DataFrame({"corpus_count": [0, 10, None, None], "wiki_pageviews_en": [5, None, 99, None],
                       "attribute": ["a", "a", "a", "a"]})
    df = add_log_freq(df)
    assert df.log_freq_source.tolist() == ["corpus_count", "corpus_count", "pageviews", None]
    assert np.isclose(df.log_freq[1], np.log1p(10)) and np.isnan(df.log_freq[3])
    assert median_tier(df).tolist() == ["low", "low", "high", None]


def test_client_caches_only_answers(tmp_path, monkeypatch):
    """403/429 are retried and never cached; 200 and 404 are cached; other 4xx are returned uncached."""
    import data.frequency as fq

    class R:
        def __init__(self, status, js):
            self.status_code, self._js, self.headers = status, js, {}

        def json(self):
            return self._js

    seq = {"a": [R(403, {"message": "Forbidden"}), R(200, {"count": 3})], "b": [R(404, {})], "c": [R(400, {"e": 1})]}
    c = fq.CachedClient(tmp_path, "test", 0.0)
    monkeypatch.setattr(c.session, "request", lambda m, u, params=None, json=None, timeout=None: seq[json["q"]].pop(0))
    monkeypatch.setattr(fq.time, "sleep", lambda s: None)
    assert c.request("s", "POST", "u", body={"q": "a"}) == (200, {"count": 3})
    assert c.throttled == {"s": 1}
    assert c.request("s", "POST", "u", body={"q": "b"}) == (404, {})
    assert c.request("s", "POST", "u", body={"q": "c"}) == (400, {"e": 1})
    assert len(list((tmp_path / "s").glob("*.json"))) == 2          # a and b only
    assert c.request("s", "POST", "u", body={"q": "a"}) == (200, {"count": 3}) and c.hits == 1
    assert c.request("s", "POST", "u", body={"q": "x"}, cache_only=True) == (None, None)
