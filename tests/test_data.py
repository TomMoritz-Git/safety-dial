import json

import pytest

from safety_dial import config
from safety_dial.data import all_items, load_anchors, load_gold, load_ladders


def _write(path, obj):
    path.write_text(json.dumps(obj))
    return path


def _mini_ladders():
    return {
        sg: {
            "escalation": "x",
            "scenarios": [
                {"id": f"{sg}_00", "surface": "s", "levels": [f"L{i} {sg}" for i in range(5)]}
            ],
        }
        for sg in config.SAFEGUARDS
    }


def test_load_ladders_and_flatten(tmp_path):
    p = _write(tmp_path / "ladders.json", _mini_ladders())
    ladders = load_ladders(p)
    assert set(ladders) == set(config.SAFEGUARDS)
    items = all_items(ladders)
    assert len(items) == len(config.SAFEGUARDS) * 5
    assert items[0].uid.endswith("/L0")


def test_load_ladders_rejects_wrong_level_count(tmp_path):
    bad = _mini_ladders()
    bad[config.SAFEGUARDS[0]]["scenarios"][0]["levels"] = ["only", "four", "lvls", "here"]
    p = _write(tmp_path / "ladders.json", bad)
    with pytest.raises(ValueError, match="levels"):
        load_ladders(p)


def test_load_ladders_rejects_unknown_safeguard(tmp_path):
    bad = _mini_ladders()
    bad["not_a_safeguard"] = bad.pop(config.SAFEGUARDS[0])
    p = _write(tmp_path / "ladders.json", bad)
    with pytest.raises(ValueError, match="safeguards"):
        load_ladders(p)


def test_load_anchors_requires_balance(tmp_path):
    p = _write(tmp_path / "anchors.json", {"benign": ["a", "b"], "harmful": ["c"]})
    with pytest.raises(ValueError, match="balanced"):
        load_anchors(p)


def test_load_gold_validates_labels(tmp_path):
    ok = [{"id": "g0", "request": "r", "response": "x", "label": "full_refuse"}]
    assert load_gold(_write(tmp_path / "gold.json", ok))[0].label == "full_refuse"
    bad = [{"id": "g0", "request": "r", "response": "x", "label": "nope"}]
    with pytest.raises(ValueError, match="bad label"):
        load_gold(_write(tmp_path / "bad.json", bad))
