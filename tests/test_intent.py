import pandas as pd

from safety_dial.intent import IntentRating, summarize, validate


def test_intent_rating_ordinal():
    assert IntentRating("benign", "").ordinal == 0
    assert IntentRating("sensitive", "").ordinal == 1
    assert IntentRating("disallowed", "").ordinal == 2


def _rated(level_to_ord):
    # Build a frame where every (safeguard, level) gets the given ordinal.
    rows = []
    for sg in ("privacy", "fraud"):
        for level, ordv in level_to_ord.items():
            for _ in range(4):
                rows.append(dict(safeguard=sg, level=level, ordinal=ordv))
    return pd.DataFrame(rows)


def test_summarize_modal_and_mean():
    tbl = summarize(_rated({0: 0, 4: 2}))
    l0 = tbl[(tbl.safeguard == "privacy") & (tbl.level == 0)].iloc[0]
    assert l0["modal_label"] == "benign" and l0["mean_ordinal"] == 0.0
    l4 = tbl[(tbl.safeguard == "privacy") & (tbl.level == 4)].iloc[0]
    assert l4["modal_label"] == "disallowed" and l4["mean_ordinal"] == 2.0


def test_validate_passes_on_clean_ladder():
    v = validate(_rated({0: 0, 1: 0, 2: 1, 3: 2, 4: 2}))
    assert v["l0_benign"] and v["l4_disallowed"] and v["monotone"] and v["passed"]


def test_validate_flags_non_monotone_and_bad_endpoints():
    v = validate(_rated({0: 2, 1: 1, 2: 1, 3: 0, 4: 0}))  # inverted ladder
    assert not v["l0_benign"]
    assert not v["l4_disallowed"]
    assert not v["monotone"]
    assert not v["passed"]
