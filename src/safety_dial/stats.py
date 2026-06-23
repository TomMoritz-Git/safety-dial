"""Small, dependency-light statistics: AUC, Cohen's d, Wilson and bootstrap CIs.

All functions take and return plain floats / NumPy arrays so they can be unit
tested without a GPU or network.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    """A point estimate with a (lo, hi) confidence interval."""

    point: float
    lo: float
    hi: float

    def as_tuple(self) -> tuple[float, float, float]:
        """Return ``(point, lo, hi)``."""
        return (self.point, self.lo, self.hi)


def auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """Probability a random positive scores above a random negative.

    This is the Mann-Whitney U / ROC-AUC estimator with a half-credit tie
    correction. Returns ``nan`` if either group is empty.

    Args:
        scores_pos: Scores for the positive class (e.g. refusals).
        scores_neg: Scores for the negative class (e.g. complies).

    Returns:
        AUC in ``[0, 1]``, or ``nan`` if a class is empty.
    """
    pos = np.asarray(scores_pos, dtype=float)
    neg = np.asarray(scores_neg, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Rank-based formula: handles ties via average ranks.
    all_scores = np.concatenate([pos, neg])
    order = all_scores.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, all_scores.size + 1)
    # Average ranks within tie groups.
    _, inv, counts = np.unique(all_scores, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    ranks = avg[inv]
    rank_pos_sum = ranks[: pos.size].sum()
    u = rank_pos_sum - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Pooled-SD Cohen's d between two 1-D samples.

    Args:
        group_a: First sample.
        group_b: Second sample.

    Returns:
        Standardised mean difference ``(mean_a - mean_b) / pooled_sd``; 0.0 if
        the pooled SD is 0.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Interval:
    """Wilson score interval for a binomial proportion.

    Args:
        successes: Number of successes.
        n: Number of trials.
        z: Normal quantile (1.96 for 95%).

    Returns:
        ``Interval`` with the observed rate as the point estimate.
    """
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return Interval(point=p, lo=float(center - half), hi=float(center + half))


def bootstrap_ci(
    func: Callable[..., float],
    *arrays: np.ndarray,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI for a statistic of one or more paired arrays.

    All input arrays must share the same length ``n``; each resample draws the
    same row indices from every array (paired bootstrap).

    Args:
        func: Callable mapping the resampled arrays to a scalar statistic.
        *arrays: One or more equal-length NumPy arrays.
        n_resamples: Number of bootstrap resamples.
        alpha: Two-sided significance (0.05 -> 95% CI).
        seed: RNG seed for reproducibility.

    Returns:
        ``Interval`` with the point estimate on the full sample.
    """
    arrs = [np.asarray(a) for a in arrays]
    n = arrs[0].shape[0]
    if any(a.shape[0] != n for a in arrs):
        raise ValueError("all arrays must share the same length")
    point = float(func(*arrs))
    if n == 0:
        return Interval(point, float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        stats[i] = func(*(a[idx] for a in arrs))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return Interval(point=point, lo=float(lo), hi=float(hi))
