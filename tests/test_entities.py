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
