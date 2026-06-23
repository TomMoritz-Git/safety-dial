"""Refusal-direction extraction and held-out layer selection.

Pure NumPy: the GPU-side activation capture lives in :mod:`safety_dial.model`;
here we only manipulate the resulting vectors so the math is unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stats import cohens_d


def diff_of_means(harm_acts: np.ndarray, benign_acts: np.ndarray) -> np.ndarray:
    """Raw difference-of-means direction ``mean(harm) - mean(benign)``.

    The vector is returned unnormalised so that steering coefficients are in
    natural activation units.

    Args:
        harm_acts: ``[n_harm, hidden]`` activations of harmful anchors.
        benign_acts: ``[n_benign, hidden]`` activations of benign anchors.

    Returns:
        ``[hidden]`` direction pointing from benign toward harmful (= refuse).
    """
    return np.asarray(harm_acts, float).mean(0) - np.asarray(benign_acts, float).mean(0)


def unit(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` scaled to unit L2 norm (unchanged if it is the zero vector)."""
    v = np.asarray(vec, float)
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def project(acts: np.ndarray, benign_mean: np.ndarray, unit_dir: np.ndarray) -> np.ndarray:
    """Signed projection of activations onto the unit refusal axis.

    Args:
        acts: ``[..., hidden]`` activations.
        benign_mean: ``[hidden]`` origin (mean of benign anchors).
        unit_dir: ``[hidden]`` unit refusal direction.

    Returns:
        Projection scalars with the leading shape of ``acts``.
    """
    return (np.asarray(acts, float) - np.asarray(benign_mean, float)) @ np.asarray(unit_dir, float)


@dataclass(frozen=True)
class Direction:
    """A fitted refusal axis at one layer."""

    layer: int
    raw: np.ndarray  # diff-of-means, natural units (used for steering)
    unit: np.ndarray  # L2-normalised (used for projection)
    benign_mean: np.ndarray  # projection origin

    def project(self, acts: np.ndarray) -> np.ndarray:
        """Project activations onto this axis (see :func:`project`)."""
        return project(acts, self.benign_mean, self.unit)


def build_direction(layer: int, harm_acts: np.ndarray, benign_acts: np.ndarray) -> Direction:
    """Fit a :class:`Direction` from all anchors at ``layer``."""
    raw = diff_of_means(harm_acts, benign_acts)
    return Direction(
        layer=layer,
        raw=raw,
        unit=unit(raw),
        benign_mean=np.asarray(benign_acts, float).mean(0),
    )


def random_matched(reference: np.ndarray, seed: int) -> np.ndarray:
    """A random direction with the same L2 norm as ``reference``.

    Used as the steering control: if a norm-matched random vector does not move
    refusal, the real direction's effect is specific rather than a generic
    activation perturbation.

    Args:
        reference: Vector whose norm to match (the raw refusal direction).
        seed: RNG seed.

    Returns:
        A ``[hidden]`` random vector with ``||r|| == ||reference||``.
    """
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(np.asarray(reference).shape)
    norm = np.linalg.norm(r)
    return (r / norm * np.linalg.norm(reference)).astype(np.float32)


@dataclass(frozen=True)
class LayerScore:
    """Separation achieved by the direction fitted at one layer."""

    layer: int
    cohens_d: float


def layer_sweep(
    harm_by_layer: np.ndarray,
    benign_by_layer: np.ndarray,
    n_train: int,
) -> tuple[list[LayerScore], int]:
    """Score every layer by held-out anchor separation and pick the best.

    For each layer the direction is fitted on the first ``n_train`` anchors per
    class and scored by Cohen's d on the remaining (held-out) anchors. Selecting
    on a held-out split avoids cherry-picking the layer to the same data.

    Args:
        harm_by_layer: ``[n_layers, n_harm, hidden]`` harmful-anchor activations.
        benign_by_layer: ``[n_layers, n_benign, hidden]`` benign-anchor acts.
        n_train: Anchors per class used to fit the direction.

    Returns:
        A ``(scores, best_layer)`` tuple, where ``scores`` is one
        :class:`LayerScore` per layer and ``best_layer`` maximises Cohen's d.
    """
    harm = np.asarray(harm_by_layer, float)
    benign = np.asarray(benign_by_layer, float)
    n_layers = harm.shape[0]
    scores: list[LayerScore] = []
    for layer in range(n_layers):
        h_tr, h_val = harm[layer, :n_train], harm[layer, n_train:]
        b_tr, b_val = benign[layer, :n_train], benign[layer, n_train:]
        u = unit(diff_of_means(h_tr, b_tr))
        b_mean = b_tr.mean(0)
        d = cohens_d(project(h_val, b_mean, u), project(b_val, b_mean, u))
        scores.append(LayerScore(layer=layer, cohens_d=d))
    best = max(scores, key=lambda s: float("-inf") if np.isnan(s.cohens_d) else s.cohens_d)
    return scores, best.layer
