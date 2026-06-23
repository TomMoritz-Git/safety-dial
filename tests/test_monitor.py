import numpy as np

from safety_dial.monitor import (
    calibration_gap,
    cross_fit_gap,
    overall_auc,
    within_topic_auc,
    youden_threshold,
)


def test_overall_auc_matches_separation():
    proj = np.array([0.0, 1.0, 2.0, 3.0])
    refused = np.array([False, False, True, True])
    assert overall_auc(proj, refused) == 1.0


def test_within_topic_auc_controls_for_topic():
    # Two topics with very different baselines but the refused item is always
    # the higher-projecting one *within* its topic.
    proj = np.array([0.0, 1.0, 100.0, 101.0])
    refused = np.array([False, True, False, True])
    sids = np.array(["a", "a", "b", "b"])
    val, n_pairs = within_topic_auc(proj, refused, sids)
    assert val == 1.0
    assert n_pairs == 2  # one refuse/comply pair per topic


def test_within_topic_auc_no_pairs_is_nan():
    proj = np.array([1.0, 2.0])
    refused = np.array([True, True])  # no complies -> no pairs
    val, n_pairs = within_topic_auc(proj, refused, np.array(["a", "a"]))
    assert np.isnan(val) and n_pairs == 0


def test_youden_threshold_separates_classes():
    proj = np.array([0.0, 1.0, 5.0, 6.0])
    refused = np.array([False, False, True, True])
    t = youden_threshold(proj, refused)
    assert 1.0 < t <= 5.0


def test_calibration_gap_counts_disagreements():
    proj = np.array([0.0, 1.0, 2.0, 3.0])
    refused = np.array([True, False, True, False])  # item0 over, item3 under
    gap = calibration_gap(proj, refused, threshold=1.5)
    assert gap.over_refusal == 1  # refused at proj 0.0 < 1.5
    assert gap.under_refusal == 1  # complied at proj 3.0 >= 1.5
    assert gap.n == 4
    assert np.isclose(gap.over_rate, 0.25)


def test_cross_fit_gap_zero_on_separable_data():
    # Perfectly separable with no within-class spread: every cross-fitted
    # threshold lands in the gap, so there are no out-of-sample disagreements.
    proj = np.array([0.0] * 40 + [1.0] * 40)
    refused = np.array([False] * 40 + [True] * 40)
    gap = cross_fit_gap(proj, refused, k=5)
    assert gap.over_refusal == 0 and gap.under_refusal == 0
    assert gap.n == 80
    assert np.isfinite(gap.threshold)


def test_cross_fit_gap_is_deterministic_and_not_optimistic():
    rng = np.random.default_rng(1)
    # Overlapping classes => some out-of-sample disagreements are unavoidable.
    proj = np.concatenate([rng.normal(0, 1.0, 50), rng.normal(1.0, 1.0, 50)])
    refused = np.array([False] * 50 + [True] * 50)
    g1 = cross_fit_gap(proj, refused, k=5, seed=7)
    g2 = cross_fit_gap(proj, refused, k=5, seed=7)
    assert (g1.over_refusal, g1.under_refusal) == (g2.over_refusal, g2.under_refusal)
    # Cross-fitted error is >= the in-sample optimum on the same threshold.
    tau = youden_threshold(proj, refused)
    insample = calibration_gap(proj, refused, tau)
    assert g1.over_refusal + g1.under_refusal >= insample.over_refusal + insample.under_refusal


def test_cross_fit_gap_single_class_is_safe():
    proj = np.array([1.0, 2.0, 3.0])
    refused = np.array([True, True, True])
    gap = cross_fit_gap(proj, refused)
    assert gap.over_refusal == 0 and gap.under_refusal == 0 and gap.n == 3
