import jax
import numpy as np
import pytest

from tunamh_jax import alias


def test_table_encodes_the_target_distribution():
    rng = np.random.default_rng(0)
    weights = rng.exponential(size=257)
    table = alias.build(weights)
    recovered = np.asarray(alias.probabilities(table))
    np.testing.assert_allclose(recovered, weights / weights.sum(), atol=1e-12)


def test_uniform_weights_give_a_pure_table():
    table = alias.build(np.ones(16))
    np.testing.assert_allclose(np.asarray(table.prob), 1.0)


def test_sampling_matches_the_weights():
    rng = np.random.default_rng(1)
    weights = rng.exponential(size=12) + 0.05
    table = alias.build(weights)
    draws = np.asarray(alias.sample(jax.random.key(0), table, (400_000,)))
    counts = np.bincount(draws, minlength=12) / draws.size
    np.testing.assert_allclose(counts, weights / weights.sum(), atol=4e-3)


@pytest.mark.parametrize("bad", [np.zeros(4), np.array([1.0, -1.0]), np.array([])])
def test_rejects_invalid_weights(bad):
    with pytest.raises(ValueError):
        alias.build(bad)


def test_zero_weight_entries_are_never_drawn():
    weights = np.array([0.0, 3.0, 0.0, 1.0])
    table = alias.build(weights)
    draws = np.asarray(alias.sample(jax.random.key(3), table, (20_000,)))
    assert set(np.unique(draws).tolist()) <= {1, 3}
