import numpy as np

from safety_dial.stats import auc, bootstrap_ci, cohens_d, wilson_ci


def test_auc_perfect_separation():
    assert auc([3, 4, 5], [0, 1, 2]) == 1.0
    assert auc([0, 1, 2], [3, 4, 5]) == 0.0


def test_auc_ties_half_credit():
    # Fully overlapping identical values -> chance.
    assert auc([1, 1], [1, 1]) == 0.5
    # 3 clean wins (2>1, 2>0, 1>0) + 1 tie (1==1) over 4 pairs -> 3.5/4.
    assert auc([2, 1], [1, 0]) == 0.875


def test_auc_empty_is_nan():
    assert np.isnan(auc([], [1, 2]))


def test_cohens_d_sign_and_scale():
    a = np.array([10.0, 12.0, 11.0, 13.0])
    b = np.array([0.0, 2.0, 1.0, 3.0])
    d = cohens_d(a, b)
    assert d > 5  # large, positive
    assert cohens_d(b, a) == -d


def test_wilson_ci_brackets_point_and_clamps():
    ci = wilson_ci(5, 10)
    assert ci.lo < ci.point < ci.hi
    assert 0 <= ci.lo and ci.hi <= 1
    # Degenerate all-success stays below 1 but high.
    ci2 = wilson_ci(10, 10)
    assert ci2.point == 1.0 and ci2.hi <= 1.0 and ci2.lo < 1.0


def test_bootstrap_ci_contains_true_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 1.0, size=200)
    ci = bootstrap_ci(lambda a: float(a.mean()), x, n_resamples=500, seed=1)
    assert ci.lo < 5.0 < ci.hi


def test_bootstrap_ci_paired_length_mismatch_raises():
    try:
        bootstrap_ci(lambda a, b: 0.0, np.zeros(3), np.zeros(4))
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched lengths")
