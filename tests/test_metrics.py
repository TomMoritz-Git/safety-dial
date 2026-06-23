import numpy as np
import pandas as pd

from safety_dial import metrics
from safety_dial.extraction import random_matched


def _graded():
    # Two models x two safeguards; within each scenario the refused item projects
    # higher, so within-topic AUC should be ~1 with very different baselines.
    rows = []
    for model in ("m1", "m2"):
        shift = 0.0 if model == "m1" else 50.0
        for sg in ("privacy", "fraud"):
            base = 0.0 if sg == "privacy" else 20.0
            for sc in range(5):
                for level in range(5):
                    refused = level >= 3
                    proj = base + shift + level + (2.0 if refused else 0.0)
                    rows.append(
                        dict(
                            model=model,
                            safeguard=sg,
                            scenario_id=f"{sg}_{sc}",
                            level=level,
                            projection=proj,
                            refused=refused,
                        )
                    )
    return pd.DataFrame(rows)


def test_random_matched_preserves_norm_and_is_deterministic():
    ref = np.array([3.0, 4.0, 0.0, 12.0])
    r = random_matched(ref, seed=1)
    assert np.isclose(np.linalg.norm(r), np.linalg.norm(ref))
    assert np.allclose(r, random_matched(ref, seed=1))
    assert not np.allclose(r, random_matched(ref, seed=2))


def test_monitor_table_high_within_auc():
    tbl = metrics.monitor_table(_graded(), n_resamples=200)
    assert set(tbl["model"]) == {"m1", "m2"}
    assert (tbl["within_auc"] > 0.9).all()
    assert (tbl["within_lo"] <= tbl["within_auc"]).all()
    assert (tbl["within_auc"] <= tbl["within_hi"]).all()
    assert (tbl["n"] == 25).all()


def test_pooled_monitor_table_has_ci_bracketing_point():
    tbl = metrics.pooled_monitor_table(_graded(), n_resamples=200)
    assert set(tbl["model"]) == {"m1", "m2"}
    assert (tbl["within_auc"] > 0.9).all()
    assert (tbl["within_lo"] <= tbl["within_auc"]).all()
    assert (tbl["within_auc"] <= tbl["within_hi"]).all()
    assert (tbl["n"] == 50).all()  # 2 safeguards x 5 scenarios x 5 levels


def test_ramp_table_is_monotone_in_level():
    tbl = metrics.ramp_table(_graded())
    cell = tbl[(tbl["model"] == "m1") & (tbl["safeguard"] == "privacy")].sort_values("level")
    assert list(cell["level"]) == [0, 1, 2, 3, 4]
    assert cell["refuse_rate"].iloc[0] == 0.0
    assert cell["refuse_rate"].iloc[-1] == 1.0


def test_dial_table_and_gap():
    rows = []
    for c in (0.0, 0.5, 1.0):
        for control in ("real", "random"):
            rate_true = {0.0: 0, 0.5: 6, 1.0: 10}[c] if control == "real" else 0
            for i in range(10):
                rows.append(dict(model="m1", coeff=c, control=control, refused=i < rate_true))
    dial = pd.DataFrame(rows)
    tbl = metrics.dial_table(dial)
    gap = metrics.dial_gap(tbl)
    top = gap[gap["coeff"] == 1.0].iloc[0]
    assert np.isclose(top["gap"], 1.0)  # real 100% - random 0%
    assert (gap["coeff"] > 0).all()  # the dead c=0 baseline is dropped


def test_intent_calibration_scores_against_ground_truth():
    # Build a graded frame with a known operating point: model "cautious" refuses
    # half its L0 (legitimate) items and all L4; model "permissive" never refuses
    # L0 but complies with half its L4 (disallowed) items.
    rows = []
    for sc in range(10):
        # cautious: L0 refused on even scenarios (5/10 over-refusal), L4 always refused.
        rows.append(dict(model="cautious", safeguard="privacy", level=0, refused=sc % 2 == 0))
        rows.append(dict(model="cautious", safeguard="privacy", level=4, refused=True))
        # permissive: L0 never refused, L4 complied on even scenarios (5/10 under-refusal).
        rows.append(dict(model="permissive", safeguard="privacy", level=0, refused=False))
        rows.append(dict(model="permissive", safeguard="privacy", level=4, refused=sc % 2 == 1))
    graded = pd.DataFrame(rows)

    tbl = metrics.intent_calibration(graded)
    cautious = tbl[tbl["model"] == "cautious"].iloc[0]
    permissive = tbl[tbl["model"] == "permissive"].iloc[0]
    assert cautious["n_legit"] == 10 and cautious["n_disallowed"] == 10
    assert np.isclose(cautious["over_refusal"], 0.5)
    assert np.isclose(cautious["under_refusal"], 0.0)
    assert np.isclose(permissive["over_refusal"], 0.0)
    assert np.isclose(permissive["under_refusal"], 0.5)
    # CIs bracket the point estimate.
    assert (tbl["over_lo"] <= tbl["over_refusal"]).all()
    assert (tbl["over_refusal"] <= tbl["over_hi"]).all()

    pooled = metrics.pooled_intent_calibration(graded)
    assert set(pooled["model"]) == {"cautious", "permissive"}
    assert np.isclose(pooled[pooled["model"] == "cautious"]["over_refusal"].iloc[0], 0.5)


def test_label_mix_fractions_sum_to_one():
    rows = []
    # L0: 6 partial, 3 comply, 1 refuse (10 items); L4: all refuse.
    labels_l0 = ["partial"] * 6 + ["full_comply"] * 3 + ["full_refuse"] * 1
    for lbl in labels_l0:
        rows.append(dict(model="m1", safeguard="privacy", level=0, label=lbl))
    for _ in range(10):
        rows.append(dict(model="m1", safeguard="privacy", level=4, label="full_refuse"))
    tbl = metrics.label_mix(pd.DataFrame(rows))
    l0 = tbl[tbl["level"] == 0].iloc[0]
    assert np.isclose(l0["frac_partial"], 0.6)
    assert np.isclose(l0["frac_full_comply"], 0.3)
    assert np.isclose(l0["frac_full_refuse"], 0.1)
    # Fractions over the three labels partition each cell.
    frac_cols = [f"frac_{lbl}" for lbl in ("full_refuse", "partial", "full_comply")]
    assert np.allclose(tbl[frac_cols].sum(axis=1), 1.0)
    l4 = tbl[tbl["level"] == 4].iloc[0]
    assert np.isclose(l4["frac_full_refuse"], 1.0)


def test_operating_band_edge():
    # random control: clean (<=0.1) at c=0.25,0.5; saturates at c=1.0.
    rates = {0.25: 0.0, 0.5: 0.1, 1.0: 0.8}
    rows = []
    for c, rr in rates.items():
        for i in range(10):
            rows.append(dict(model="m1", coeff=c, control="random", refused=i < rr * 10))
    tbl = metrics.dial_table(pd.DataFrame(rows))
    band = metrics.operating_band(tbl, spec_threshold=0.1)
    assert float(band["band_edge"].iloc[0]) == 0.5
