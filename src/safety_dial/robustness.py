"""Anchor-robustness: is the refusal direction a property of the model, or of 8 prompts?

The deployed direction is a diff-of-means over a small anchor set. To show the
read-side result does not hinge on that particular handful, we capture activations
for a larger anchor *pool* (``data/anchor_pool.json``) and for every graded item,
then -- purely in NumPy -- resample the anchors many ways and watch the monitor:

* **bootstrap** the direction (resample the pool with replacement) -> a CI on
  within-topic AUC that includes anchor-sampling uncertainty, not just item noise;
* **sweep** the anchor count N (4, 8, 16, 32) -> does AUC plateau, i.e. is N=8 on
  the flat of the curve;
* **cosine stability** -> how aligned are pool-resampled directions with the
  deployed 8-anchor direction.

Capture (:func:`capture_model`) needs the GPU and lives behind the same
``ModelRunner`` as the rest; the analysis functions below are pure and unit-tested.
The deployed direction and judged responses are never touched.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import config, monitor
from .extraction import diff_of_means, unit


def _robust_dir() -> Path:
    d = config.RESULTS_DIR / "robustness"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# Capture (GPU) -- forward passes only, no generation, no judge.
# --------------------------------------------------------------------------
def capture_model(spec, pool, ladders, refused_by_uid: dict[str, bool]) -> Path:
    """Capture pooled-anchor and graded activations for one model.

    Args:
        spec: Model to load.
        pool: ``Anchors`` with the expanded benign/harmful pool.
        ladders: Loaded ladder dataset.
        refused_by_uid: Map ``item.uid -> refused`` from the existing judged run,
            so robustness reuses the validated labels (no re-judge).

    Returns:
        Path to ``results/robustness/<key>.npz``.
    """
    from .data import all_items
    from .model import ModelRunner

    deployed = np.load(config.RESULTS_DIR / "directions" / f"{spec.key}.npz")
    layer = int(deployed["layer"])

    runner = ModelRunner.load(spec)
    try:
        harm = np.stack([runner.activations(p)[layer] for p in pool.harmful])  # [n_harm, H]
        benign = np.stack([runner.activations(p)[layer] for p in pool.benign])  # [n_benign, H]
        items = all_items(ladders)
        g_acts = np.stack([runner.activations(it.prompt)[layer] for it in items])  # [n_graded, H]
    finally:
        runner.unload()

    out = _robust_dir() / f"{spec.key}.npz"
    np.savez(
        out,
        layer=layer,
        harm_acts=harm.astype(np.float32),
        benign_acts=benign.astype(np.float32),
        graded_acts=g_acts.astype(np.float32),
        graded_scenario=np.array([it.scenario_id for it in items]),
        graded_level=np.array([it.level for it in items]),
        graded_safeguard=np.array([it.safeguard for it in items]),
        graded_refused=np.array([bool(refused_by_uid[it.uid]) for it in items]),
        deployed_unit=deployed["unit"].astype(np.float32),
    )
    return out


# --------------------------------------------------------------------------
# Analysis (pure NumPy).
# --------------------------------------------------------------------------
def _auc_from_anchors(
    harm: np.ndarray, benign: np.ndarray, g_acts: np.ndarray, refused: np.ndarray, sids: np.ndarray
) -> float:
    """Within-topic AUC of the diff-of-means direction built from given anchors."""
    u = unit(diff_of_means(harm, benign))
    b_mean = benign.mean(0)
    proj = (g_acts - b_mean) @ u
    val, _ = monitor.within_topic_auc(proj, refused, sids)
    return val


def bootstrap_auc(cap: dict, n_boot: int = 500, seed: int = 0) -> dict:
    """Bootstrap within-topic AUC over the full anchor pool (resample w/ replacement)."""
    rng = np.random.default_rng(seed)
    harm, benign = cap["harm_acts"], cap["benign_acts"]
    g, ref, sids = cap["graded_acts"], cap["graded_refused"], cap["graded_scenario"]
    nh, nb = harm.shape[0], benign.shape[0]
    point = _auc_from_anchors(harm, benign, g, ref, sids)
    stats = []
    for _ in range(n_boot):
        hi = rng.integers(0, nh, nh)
        bi = rng.integers(0, nb, nb)
        val = _auc_from_anchors(harm[hi], benign[bi], g, ref, sids)
        if not np.isnan(val):
            stats.append(val)
    lo, hi_ = np.quantile(stats, [0.025, 0.975]) if stats else (float("nan"), float("nan"))
    return {"point": float(point), "lo": float(lo), "hi": float(hi_), "n_boot": len(stats)}


def nsweep(cap: dict, sizes=(4, 8, 16, 32), reps: int = 200, seed: int = 0) -> pd.DataFrame:
    """Mean within-topic AUC as a function of anchors-per-class N (sampled w/o replacement)."""
    rng = np.random.default_rng(seed)
    harm, benign = cap["harm_acts"], cap["benign_acts"]
    g, ref, sids = cap["graded_acts"], cap["graded_refused"], cap["graded_scenario"]
    nh, nb = harm.shape[0], benign.shape[0]
    rows = []
    for n in sizes:
        n = min(n, nh, nb)
        vals = []
        for _ in range(reps):
            hi = rng.choice(nh, n, replace=False)
            bi = rng.choice(nb, n, replace=False)
            v = _auc_from_anchors(harm[hi], benign[bi], g, ref, sids)
            if not np.isnan(v):
                vals.append(v)
        arr = np.array(vals)
        rows.append(
            {
                "n_per_class": int(n),
                "mean_auc": float(arr.mean()),
                "lo": float(np.quantile(arr, 0.025)),
                "hi": float(np.quantile(arr, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def cosine_stability(cap: dict, n_boot: int = 500, seed: int = 0) -> dict:
    """Cosine between the deployed direction and bootstrap-resampled pool directions."""
    rng = np.random.default_rng(seed)
    harm, benign = cap["harm_acts"], cap["benign_acts"]
    deployed = unit(cap["deployed_unit"])
    nh, nb = harm.shape[0], benign.shape[0]
    c005 = unit(diff_of_means(harm, benign))  # full-pool direction
    pool_vs_deployed = float(c005 @ deployed)
    cosines = []
    for _ in range(n_boot):
        hi = rng.integers(0, nh, nh)
        bi = rng.integers(0, nb, nb)
        u = unit(diff_of_means(harm[hi], benign[bi]))
        cosines.append(float(u @ deployed))
    arr = np.array(cosines)
    return {
        "pool_vs_deployed": pool_vs_deployed,
        "mean_cos_to_deployed": float(arr.mean()),
        "lo": float(np.quantile(arr, 0.025)),
        "hi": float(np.quantile(arr, 0.975)),
    }


def load_capture(key: str) -> dict:
    """Load a captured ``.npz`` into a plain dict of arrays."""
    data = np.load(_robust_dir() / f"{key}.npz", allow_pickle=False)
    return {k: data[k] for k in data.files}
