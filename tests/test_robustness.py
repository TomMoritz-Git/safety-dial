import numpy as np

from safety_dial import robustness


def _synthetic_cap(seed=0):
    # Hidden dim 4; refusal lives on axis 0. Harmful anchors sit high on axis 0,
    # benign low; within each scenario the refused graded item projects higher.
    rng = np.random.default_rng(seed)
    H = 4
    harm = rng.normal(0, 0.3, (32, H))
    harm[:, 0] += 6.0
    benign = rng.normal(0, 0.3, (32, H))
    g_acts, refused, sids = [], [], []
    for sc in range(20):
        for level in range(5):
            ref = level >= 3
            v = rng.normal(0, 0.3, H)
            v[0] += 5.0 if ref else 0.5
            g_acts.append(v)
            refused.append(ref)
            sids.append(f"s{sc}")
    return {
        "harm_acts": harm,
        "benign_acts": benign,
        "graded_acts": np.array(g_acts),
        "graded_refused": np.array(refused),
        "graded_scenario": np.array(sids),
        "deployed_unit": np.eye(H)[0],  # deployed direction is the true axis
    }


def test_bootstrap_auc_high_and_bracketed():
    res = robustness.bootstrap_auc(_synthetic_cap(), n_boot=100)
    assert res["point"] > 0.95
    assert res["lo"] <= res["point"] <= res["hi"]


def test_nsweep_plateaus_high():
    df = robustness.nsweep(_synthetic_cap(), sizes=(4, 8, 16, 32), reps=40)
    assert list(df["n_per_class"]) == [4, 8, 16, 32]
    assert (df["mean_auc"] > 0.9).all()  # separable -> high at every N


def test_cosine_stability_near_one():
    res = robustness.cosine_stability(_synthetic_cap(), n_boot=100)
    assert res["pool_vs_deployed"] > 0.95
    assert res["mean_cos_to_deployed"] > 0.9
