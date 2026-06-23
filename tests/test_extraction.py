import numpy as np

from safety_dial.extraction import (
    build_direction,
    diff_of_means,
    layer_sweep,
    project,
    unit,
)


def test_diff_of_means_points_benign_to_harm():
    harm = np.array([[2.0, 0.0], [4.0, 0.0]])
    benign = np.array([[0.0, 1.0], [0.0, -1.0]])
    v = diff_of_means(harm, benign)
    assert np.allclose(v, [3.0, 0.0])


def test_unit_is_normalised_and_zero_safe():
    assert np.isclose(np.linalg.norm(unit(np.array([3.0, 4.0]))), 1.0)
    assert np.allclose(unit(np.zeros(3)), np.zeros(3))


def test_project_orders_along_axis():
    benign = np.array([[0.0, 0.0], [0.0, 0.0]])
    direction = build_direction(0, np.array([[1.0, 0.0]]), benign)
    proj = project(np.array([[2.0, 9.0], [-1.0, 9.0]]), direction.benign_mean, direction.unit)
    # Only the x-component matters; y is orthogonal to the axis.
    assert proj[0] > proj[1]
    assert np.isclose(proj[0], 2.0)


def test_layer_sweep_picks_the_separable_layer():
    rng = np.random.default_rng(0)
    n_per_class, hidden, n_layers = 8, 16, 4
    separable = 2
    harm = np.zeros((n_layers, n_per_class, hidden))
    benign = np.zeros((n_layers, n_per_class, hidden))
    for layer in range(n_layers):
        scale = 6.0 if layer == separable else 0.0
        harm[layer] = rng.normal(0, 0.5, (n_per_class, hidden))
        benign[layer] = rng.normal(0, 0.5, (n_per_class, hidden))
        harm[layer, :, 0] += scale  # signal only on the separable layer
    scores, best = layer_sweep(harm, benign, n_train=4)
    assert best == separable
    assert len(scores) == n_layers
    assert scores[separable].cohens_d > max(
        s.cohens_d for i, s in enumerate(scores) if i != separable
    )
