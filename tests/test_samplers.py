"""End-to-end sampler behaviour.

The headline property of TunaMH, TFMH and FlyMC is *exactness*: they must leave
the same posterior invariant as full-batch MH despite touching a handful of
data points per step.  These tests check that against a one-dimensional target
whose mean is available by quadrature, which is the only way to catch a
subtly-wrong acceptance ratio -- such a bug still produces a plausible-looking
chain.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import make_dataset, quadrature_posterior_moments
from tunamh_jax.runner import run_chain
from tunamh_jax.samplers import flymc, mh, tfmh, tunamh


@pytest.fixture(scope="module")
def problem():
    dataset = make_dataset(n=40, dim=1, seed=0)
    mean, std = quadrature_posterior_moments(dataset)
    return dataset, mean, std


def _chain(module, dataset, *, seed=0, n_samples, burnin, build_kwargs, chunk=64, **run_kwargs):
    cfg = module.build(dataset, **build_kwargs)
    key_init, key_chain = jax.random.split(jax.random.key(seed))
    state = module.init_state(cfg, key_init)
    step = partial(module.step, chunk=chunk) if module in (tunamh, tfmh) else module.step
    return run_chain(
        step,
        cfg,
        state,
        key_chain,
        n_samples=n_samples,
        burnin=burnin,
        interval=max(n_samples // 4, 1),
        warmup=False,
        **run_kwargs,
    )


# Settings chosen so every chain mixes well enough for a 5%-of-sigma tolerance.
EXACTNESS_CASES = {
    "mh": (mh, {}, 60_000, 1.0),
    "tunamh": (tunamh, {"chi": 1e-3}, 60_000, 1.0),
    "tfmh": (tfmh, {}, 200_000, 0.35),
    "flymc": (flymc, {"q_db": 0.5}, 60_000, 1.0),
}


@pytest.mark.parametrize("name", sorted(EXACTNESS_CASES))
def test_stationary_distribution_matches_quadrature(problem, name):
    dataset, exact_mean, exact_std = problem
    module, extra, n_samples, step_scale = EXACTNESS_CASES[name]

    result = _chain(
        module,
        dataset,
        n_samples=n_samples,
        burnin=n_samples // 10,
        build_kwargs={"stepsize": step_scale * exact_std, **extra},
    )

    estimate = float(result.theta_mean[0])
    assert abs(estimate - exact_mean) < 0.05 * exact_std, (
        f"{name}: posterior mean {estimate:.4f} vs quadrature {exact_mean:.4f} "
        f"(sigma={exact_std:.4f}, accept rate {result.accept_rate:.3f})"
    )
    assert 0.0 < result.accept_rate <= 1.0


def test_minibatch_samplers_touch_far_less_data_than_mh(problem):
    dataset, _, exact_std = problem
    full = _chain(mh, dataset, n_samples=2_000, burnin=0, build_kwargs={"stepsize": exact_std})
    mini = _chain(
        tunamh,
        dataset,
        n_samples=2_000,
        burnin=0,
        build_kwargs={"stepsize": 0.05 * exact_std, "chi": 1e-4},
    )
    assert full.avg_batch_size == dataset.n_train
    assert mini.avg_batch_size < 0.5 * dataset.n_train


def test_tunamh_batch_size_matches_its_poisson_rate(problem):
    """E[B] = chi C^2 M^2 + C M, with M = E||theta' - theta|| for the proposal."""
    dataset, _, _ = problem
    stepsize, chi = 0.01, 1e-3
    cfg = tunamh.build(dataset, stepsize=stepsize, chi=chi)
    result = _chain(
        tunamh, dataset, n_samples=20_000, burnin=0, build_kwargs={"stepsize": stepsize, "chi": chi}
    )

    total_c = float(cfg.total_c)
    # In one dimension the proposal displacement is |N(0, stepsize^2)|.
    m = stepsize * np.sqrt(2.0 / np.pi)
    expected = chi * (total_c * m) ** 2 + total_c * m
    assert result.avg_batch_size == pytest.approx(expected, rel=0.05)


def test_chunk_width_does_not_change_the_stationary_distribution(problem):
    """A narrow chunk forces many loop trips; the target must be unaffected."""
    dataset, exact_mean, exact_std = problem
    kwargs = dict(
        n_samples=60_000, burnin=6_000, build_kwargs={"stepsize": exact_std, "chi": 1e-3}
    )
    wide = _chain(tunamh, dataset, chunk=256, **kwargs)
    narrow = _chain(tunamh, dataset, chunk=3, seed=1, **kwargs)
    for result in (wide, narrow):
        assert abs(float(result.theta_mean[0]) - exact_mean) < 0.05 * exact_std


@pytest.mark.parametrize("module", [mh, tunamh, tfmh, flymc])
def test_step_is_deterministic_given_a_key(module):
    dataset = make_dataset(n=25, dim=3, seed=4)
    cfg = module.build(dataset, stepsize=0.05)
    state = module.init_state(cfg, jax.random.key(0))
    step = partial(module.step, chunk=32) if module in (tunamh, tfmh) else module.step
    first, stats_a = step(cfg, state, jax.random.key(7))
    second, stats_b = step(cfg, state, jax.random.key(7))
    np.testing.assert_array_equal(np.asarray(first.theta), np.asarray(second.theta))
    assert bool(stats_a.accepted) == bool(stats_b.accepted)


@pytest.mark.parametrize("module", [mh, tunamh, tfmh, flymc])
def test_state_only_ever_holds_theta_or_the_proposal(module):
    """A rejected step must leave theta untouched, bit for bit."""
    dataset = make_dataset(n=25, dim=3, seed=5)
    # A huge step is rejected essentially always.
    cfg = module.build(dataset, stepsize=1e4)
    state = module.init_state(cfg, jax.random.key(0))
    step = partial(module.step, chunk=32) if module in (tunamh, tfmh) else module.step
    for i in range(10):
        new_state, stats = step(cfg, state, jax.random.key(i))
        if not bool(stats.accepted):
            np.testing.assert_array_equal(np.asarray(new_state.theta), np.asarray(state.theta))
        state = new_state


@pytest.mark.parametrize("module", [mh, tunamh, tfmh, flymc])
def test_step_is_jittable_without_recompiling(module):
    dataset = make_dataset(n=25, dim=3, seed=6)
    cfg = module.build(dataset, stepsize=0.05)
    state = module.init_state(cfg, jax.random.key(0))
    step = partial(module.step, chunk=32) if module in (tunamh, tfmh) else module.step
    jitted = jax.jit(step).lower(cfg, state, jax.random.key(0)).compile()
    for i in range(5):
        state, _ = jitted(cfg, state, jax.random.key(i))
    assert jnp.all(jnp.isfinite(state.theta))
