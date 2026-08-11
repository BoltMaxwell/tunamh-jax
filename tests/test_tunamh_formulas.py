"""The port rewrites the paper's per-index formulas; check the algebra holds.

``minibatch_terms`` uses ``(q + phi_low)/(q + c_i M)`` and
``log(q + phi_high) - log(q + phi_low)``.  The paper writes the keep
probability with explicit ``chi c_i C M^2`` and the log-ratio term as an
``artanh``; the Julia reference used ``log(1 + L phi/(lam M_i))``.  All three
must agree.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tunamh_jax.samplers.tunamh import minibatch_terms


def _draws(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    chi = 10 ** rng.uniform(-6, 0, size=n)
    total_c = 10 ** rng.uniform(0, 5, size=n)
    m = 10 ** rng.uniform(-4, 0, size=n)
    ci = total_c * rng.uniform(1e-5, 1e-3, size=n)
    # Assumption 1 keeps the energy difference inside +/- c_i M.
    energy_delta = ci * m * rng.uniform(-1.0, 1.0, size=n)
    return chi, total_c, m, ci, energy_delta


def test_keep_probability_matches_the_paper():
    chi, total_c, m, ci, energy_delta = _draws()
    q = chi * (total_c * m) * (ci * m)
    keep_prob, _ = minibatch_terms(jnp.asarray(q), jnp.asarray(ci * m), jnp.asarray(energy_delta))

    phi_low = 0.5 * (energy_delta + ci * m)
    expected = (chi * ci * total_c * m**2 + phi_low) / (chi * ci * total_c * m**2 + ci * m)
    np.testing.assert_allclose(np.asarray(keep_prob), expected, rtol=1e-12)
    assert np.all((np.asarray(keep_prob) >= 0) & (np.asarray(keep_prob) <= 1))


def test_log_ratio_term_matches_the_papers_artanh_form():
    chi, total_c, m, ci, energy_delta = _draws(seed=1)
    q = chi * (total_c * m) * (ci * m)
    _, contribution = minibatch_terms(
        jnp.asarray(q), jnp.asarray(ci * m), jnp.asarray(energy_delta)
    )

    # 2 artanh( (U_i(theta) - U_i(theta')) / (c_i M (1 + 2 chi C M)) )
    numerator = -energy_delta
    expected = 2.0 * np.arctanh(numerator / (ci * m * (1 + 2 * chi * total_c * m)))
    np.testing.assert_allclose(np.asarray(contribution), expected, rtol=1e-9, atol=1e-12)


def test_log_ratio_term_matches_the_julia_reference_form():
    chi, total_c, m, ci, energy_delta = _draws(seed=2)
    q = chi * (total_c * m) * (ci * m)
    _, contribution = minibatch_terms(
        jnp.asarray(q), jnp.asarray(ci * m), jnp.asarray(energy_delta)
    )

    big_l = m * total_c
    lam = chi * big_l**2
    mi = m * ci
    phi_low = 0.5 * (energy_delta + mi)
    phi_high = mi - phi_low
    expected = np.log1p(big_l / (lam * mi) * phi_high) - np.log1p(big_l / (lam * mi) * phi_low)
    np.testing.assert_allclose(np.asarray(contribution), expected, rtol=1e-9, atol=1e-12)


def test_symmetry_under_swapping_the_endpoints():
    """Swapping theta and theta' must negate the log-ratio; this is what makes
    TunaMH reversible, so it is worth pinning down."""
    chi, total_c, m, ci, energy_delta = _draws(seed=3)
    q = jnp.asarray(chi * (total_c * m) * (ci * m))
    ci_m = jnp.asarray(ci * m)
    _, forward = minibatch_terms(q, ci_m, jnp.asarray(energy_delta))
    _, backward = minibatch_terms(q, ci_m, jnp.asarray(-energy_delta))
    np.testing.assert_allclose(np.asarray(forward), -np.asarray(backward), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("ci_m", [0.0, 1e-300])
def test_zero_weight_points_are_inert(ci_m):
    keep_prob, contribution = minibatch_terms(
        jnp.asarray(0.0), jnp.asarray(ci_m), jnp.asarray(0.0)
    )
    assert np.isfinite(float(contribution))
    assert float(keep_prob) == 0.0 or float(contribution) == 0.0
