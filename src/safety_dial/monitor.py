"""The refusal axis as a *monitor*: read-side AUC, threshold, calibration gap.

The projection of an item onto the refusal axis is a cheap pre-generation
predictor of whether the model will refuse. Here we score how well it predicts
(within-topic and overall AUC), pick an operating threshold, and locate the
monitor<->action disagreements (over- and under-refusal).
"""

from dataclasses import dataclass

import numpy as np

from .stats import auc


def overall_auc(projections: np.ndarray, refused: np.ndarray) -> float:
    """AUC of projection predicting refusal over all items pooled."""
    proj = np.asarray(projections, float)
    ref = np.asarray(refused, bool)
    return auc(proj[ref], proj[~ref])


def within_topic_auc(
    projections: np.ndarray,
    refused: np.ndarray,
    scenario_ids: np.ndarray,
) -> tuple[float, int]:
    """AUC restricted to refuse/comply pairs within the same scenario ladder.

    This controls for topic entirely: a high score means the projection tracks
    *intent severity within a fixed topic*, not just which topic the request is
    about. Pairs are formed only between a refused and a complied item of the
    same scenario; the statistic is the fraction of such pairs where the refused
    item projects higher (ties count as half).

    Args:
        projections: ``[n]`` projection scalars.
        refused: ``[n]`` boolean refusal labels.
        scenario_ids: ``[n]`` scenario identifiers.

    Returns:
        ``(auc, n_pairs)``; ``auc`` is ``nan`` if no within-topic pair exists.
    """
    proj = np.asarray(projections, float)
    ref = np.asarray(refused, bool)
    sids = np.asarray(scenario_ids)
    wins = 0.0
    n_pairs = 0
    for sid in np.unique(sids):
        mask = sids == sid
        p_ref = proj[mask & ref]
        p_cmp = proj[mask & ~ref]
        for a in p_ref:
            for b in p_cmp:
                wins += (a > b) + 0.5 * (a == b)
                n_pairs += 1
    if n_pairs == 0:
        return float("nan"), 0
    return wins / n_pairs, n_pairs


def youden_threshold(projections: np.ndarray, refused: np.ndarray) -> float:
    """Operating threshold maximising Youden's J (TPR - FPR).

    Args:
        projections: ``[n]`` projection scalars.
        refused: ``[n]`` boolean refusal labels.

    Returns:
        The projection value of the best threshold; ``nan`` if a class is empty.
    """
    proj = np.asarray(projections, float)
    ref = np.asarray(refused, bool)
    pos, neg = ref.sum(), (~ref).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    # Candidate thresholds: each unique projection (predict refuse if proj >= t).
    best_t, best_j = float("nan"), -np.inf
    for t in np.unique(proj):
        pred = proj >= t
        tpr = (pred & ref).sum() / pos
        fpr = (pred & ~ref).sum() / neg
        if tpr - fpr > best_j:
            best_j, best_t = tpr - fpr, float(t)
    return best_t


@dataclass(frozen=True)
class CalibrationGap:
    """Counts of monitor<->action disagreements relative to a threshold."""

    threshold: float
    over_refusal: int  # refused but projection < threshold
    under_refusal: int  # complied but projection >= threshold
    n: int

    @property
    def over_rate(self) -> float:
        """Fraction of all items that are over-refusals."""
        return self.over_refusal / self.n if self.n else float("nan")

    @property
    def under_rate(self) -> float:
        """Fraction of all items that are under-refusals."""
        return self.under_refusal / self.n if self.n else float("nan")


def calibration_gap(
    projections: np.ndarray,
    refused: np.ndarray,
    threshold: float,
) -> CalibrationGap:
    """Locate over/under-refusals: where perception and action disagree.

    In-sample: the disagreements are counted against a fixed ``threshold``. For
    the reported rates we use :func:`cross_fit_gap` instead, which does not let
    the threshold see the items it is scored on.

    Args:
        projections: ``[n]`` projection scalars.
        refused: ``[n]`` boolean refusal labels.
        threshold: Operating threshold (e.g. from :func:`youden_threshold`).

    Returns:
        A :class:`CalibrationGap` summarising the disagreement counts.
    """
    proj = np.asarray(projections, float)
    ref = np.asarray(refused, bool)
    high = proj >= threshold
    over = int((ref & ~high).sum())
    under = int((~ref & high).sum())
    return CalibrationGap(threshold=threshold, over_refusal=over, under_refusal=under, n=ref.size)


def cross_fit_gap(
    projections: np.ndarray,
    refused: np.ndarray,
    *,
    k: int = 5,
    seed: int = 0,
) -> CalibrationGap:
    """Over/under-refusal counted out-of-sample via cross-fitted thresholds.

    The plain :func:`calibration_gap` fits the Youden threshold on the same
    items it then scores, so its disagreement rates are optimistic. Here the data
    is split into ``k`` label-stratified folds; for each fold the threshold is fit
    on the *other* folds and applied to the held-out one, so every item is judged
    by a threshold that never saw it. The reported ``threshold`` is still the
    full-data Youden operating point (what you would actually deploy); only the
    counts are cross-fitted.

    Args:
        projections: ``[n]`` projection scalars.
        refused: ``[n]`` boolean refusal labels.
        k: Number of folds (capped at the minority-class size).
        seed: RNG seed for the fold assignment.

    Returns:
        A :class:`CalibrationGap` whose counts are out-of-sample.
    """
    proj = np.asarray(projections, float)
    ref = np.asarray(refused, bool)
    n = ref.size
    full_tau = youden_threshold(proj, ref)
    pos, neg = int(ref.sum()), int((~ref).sum())
    if pos == 0 or neg == 0:
        return CalibrationGap(threshold=full_tau, over_refusal=0, under_refusal=0, n=n)
    k_eff = min(k, pos, neg)  # every fold must be able to hold both classes
    rng = np.random.default_rng(seed)
    fold = np.empty(n, dtype=int)
    for cls in (True, False):
        idx = rng.permutation(np.where(ref == cls)[0])
        fold[idx] = np.arange(idx.size) % k_eff
    over = under = 0
    for f in range(k_eff):
        test = fold == f
        tau = youden_threshold(proj[~test], ref[~test])
        if np.isnan(tau):
            tau = full_tau
        high = proj[test] >= tau
        over += int((ref[test] & ~high).sum())
        under += int((~ref[test] & high).sum())
    return CalibrationGap(threshold=full_tau, over_refusal=over, under_refusal=under, n=n)
