import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import make_dataset
from tunamh_jax.runner import run_chain
from tunamh_jax.samplers import mh


@pytest.fixture
def setup():
    dataset = make_dataset(n=30, dim=2, seed=0)
    cfg = mh.build(dataset, stepsize=0.1)
    state = mh.init_state(cfg, jax.random.key(0))
    return cfg, state


def test_running_mean_equals_the_mean_of_the_stored_trace(setup):
    cfg, state = setup
    result = run_chain(
        mh.step, cfg, state, jax.random.key(1),
        n_samples=200, interval=50, store_samples=True, warmup=False,
    )
    assert result.samples.shape == (200, 2)
    np.testing.assert_allclose(
        np.asarray(result.theta_mean), np.asarray(result.samples).mean(axis=0), rtol=1e-12
    )


def test_burnin_states_are_excluded(setup):
    """Burn-in iterations must not enter the mean, the trace or the statistics."""
    cfg, state = setup
    result = run_chain(
        mh.step, cfg, state, jax.random.key(2),
        n_samples=100, burnin=50, interval=50, store_samples=True, warmup=False,
    )
    assert [c.iteration for c in result.checkpoints] == [50, 100]
    assert result.samples.shape == (100, 2)
    assert result.checkpoints[-1].data_used == pytest.approx(100 * 30)
    np.testing.assert_allclose(
        np.asarray(result.theta_mean), np.asarray(result.samples).mean(axis=0), rtol=1e-12
    )


def test_checkpoints_cover_the_run(setup):
    cfg, state = setup
    result = run_chain(
        mh.step, cfg, state, jax.random.key(3), n_samples=250, interval=100, warmup=False
    )
    assert [c.iteration for c in result.checkpoints] == [100, 200, 250]
    elapsed = [c.elapsed_seconds for c in result.checkpoints]
    data = [c.data_used for c in result.checkpoints]
    assert elapsed == sorted(elapsed)
    assert data == sorted(data)
    assert data[-1] == pytest.approx(250 * 30)


def test_monitor_is_called_with_the_running_mean(setup):
    cfg, state = setup
    seen = []

    def monitor(theta):
        seen.append(np.asarray(theta))
        return jnp.sum(theta)

    result = run_chain(
        mh.step, cfg, state, jax.random.key(4), n_samples=60, interval=20,
        monitor=monitor, store_samples=True, warmup=False,
    )
    samples = np.asarray(result.samples)
    assert len(seen) == 3
    for i, observed in enumerate(seen):
        np.testing.assert_allclose(observed, samples[: 20 * (i + 1)].mean(axis=0), rtol=1e-12)


def test_results_are_reproducible(setup):
    cfg, state = setup
    kwargs = dict(n_samples=120, interval=40, warmup=False)
    a = run_chain(mh.step, cfg, state, jax.random.key(5), **kwargs)
    b = run_chain(mh.step, cfg, state, jax.random.key(5), **kwargs)
    np.testing.assert_array_equal(np.asarray(a.theta_mean), np.asarray(b.theta_mean))
    assert a.accept_rate == b.accept_rate


def test_warmup_does_not_change_the_chain(setup):
    """The warm-up block is a throwaway; it must not consume the chain's keys."""
    cfg, state = setup
    kwargs = dict(n_samples=80, interval=40)
    cold = run_chain(mh.step, cfg, state, jax.random.key(6), warmup=False, **kwargs)
    warm = run_chain(mh.step, cfg, state, jax.random.key(6), warmup=True, **kwargs)
    assert warm.compile_seconds > 0.0
    np.testing.assert_array_equal(np.asarray(cold.theta_mean), np.asarray(warm.theta_mean))
    assert cold.accept_rate == warm.accept_rate
