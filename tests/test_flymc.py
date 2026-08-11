"""FlyMC-specific checks: the bound, and the vectorised brightness resampling.

The reference walked the dark set with geometric skips and applied a
Metropolis test to each visited point.  This port collapses that into one
Bernoulli per point.  The tests below confirm the collapsed flip probabilities
are the ones the two-stage procedure produces.
"""

import jax
import jax.numpy as jnp
import numpy as np

from conftest import make_dataset
from tunamh_jax.samplers import flymc


def _reference_log_ratio(x_signed, theta, a, b, c, temperature):
    """Literal transcription of the reference's ``log_bd_ratio``."""
    z = x_signed @ theta
    log_likelihood = 0.5 * (z - np.abs(z)) - np.log1p(np.exp(-np.abs(z)))
    log_bound = a * z**2 + b * z + c
    gap = (log_likelihood - log_bound) / temperature
    return gap + np.log1p(-np.exp(-gap))


def test_bound_is_a_genuine_lower_bound():
    """log B(z) <= log sigmoid(z) everywhere, so the gap never goes negative."""
    a, b, c = flymc.bound_coefficients()
    z = np.linspace(-60, 60, 200_001)
    log_sigmoid = np.where(z >= 0, -np.log1p(np.exp(-z)), z - np.log1p(np.exp(z)))
    assert np.all(a * z**2 + b * z + c <= log_sigmoid)


def test_bound_is_tight_at_the_expansion_point():
    a, b, c = flymc.bound_coefficients()
    for z in (1.5, -1.5):
        log_sigmoid = -np.log1p(np.exp(-z))
        assert abs((a * z**2 + b * z + c) - log_sigmoid) < 1e-5


def test_log_ratio_matches_the_reference_transcription():
    dataset = make_dataset(n=64, dim=4, seed=1)
    cfg = flymc.build(dataset, stepsize=0.01, temperature=0.9)
    theta = np.random.default_rng(0).normal(size=4)
    a, b, c = flymc.bound_coefficients()
    expected = _reference_log_ratio(np.asarray(cfg.x), theta, a, b, c, 0.9)
    got = flymc.log_bright_dark_ratio(cfg, jnp.asarray(theta))
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-10)


def test_brightness_flip_rates_match_the_two_stage_procedure():
    """Dark -> bright fires with probability min(q_db, exp(lbr)); bright -> dark
    with min(1, q_db * exp(-lbr)).  Those are the composites of the reference's
    geometric visit and its Metropolis test."""
    dataset = make_dataset(n=96, dim=3, seed=2)
    q_db = 0.3
    cfg = flymc.build(dataset, stepsize=0.01, q_db=q_db)
    theta = jnp.asarray(np.random.default_rng(1).normal(size=3))
    log_ratio = flymc.log_bright_dark_ratio(cfg, theta)

    all_dark = jnp.zeros(dataset.n_train, dtype=bool)
    all_bright = jnp.ones(dataset.n_train, dtype=bool)
    trials = 6_000
    resample = jax.vmap(
        lambda key, mask: flymc.resample_brightness(cfg, theta, mask, key, log_ratio),
        in_axes=(0, None),
    )
    keys = jax.random.split(jax.random.key(3), trials)

    lit = np.asarray(resample(keys, all_dark)).mean(axis=0)
    stayed = np.asarray(resample(keys, all_bright)).mean(axis=0)

    ratio = np.exp(np.asarray(log_ratio))
    expected_lit = np.minimum(q_db, ratio)
    expected_stayed = 1.0 - np.minimum(1.0, q_db / ratio)
    tolerance = 4.0 * np.sqrt(0.25 / trials)
    np.testing.assert_allclose(lit, expected_lit, atol=tolerance)
    np.testing.assert_allclose(stayed, expected_stayed, atol=tolerance)


def test_brightness_resampling_leaves_theta_alone():
    dataset = make_dataset(n=32, dim=2, seed=3)
    cfg = flymc.build(dataset, stepsize=0.01)
    state = flymc.init_state(cfg, jax.random.key(0))
    assert not bool(jnp.any(state.bright)), "the reference starts every point dark"
    new_state, stats = flymc.step(cfg, state, jax.random.key(1))
    assert int(stats.batch_size) == int(jnp.sum(new_state.bright))
