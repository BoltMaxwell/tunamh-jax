"""Truncated Factorized Metropolis-Hastings (TFMH).

FMH factorises the acceptance decision into one Bernoulli test per data point
and rejects as soon as any factor fails, so a step usually touches only a
handful of points.  The truncated variant draws how many factors to test from
``Poisson(C M)`` using the same ``c_i / C`` index distribution as TunaMH.

Each drawn index rejects the proposal with probability
``max(0, U_i(θ') - U_i(θ)) / (c_i M)``.  The reference loop returned the moment
one index rejected.  Because the tests are independent and none of them affects
the others' draws, evaluating a whole chunk at once and then taking the *first*
index that would have rejected is exactly equivalent -- including for the
reported batch size, which counts indices examined up to and including the one
that rejected.  The chunk loop exits as soon as a chunk contains a rejection,
so the short-circuit is preserved at chunk granularity.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from tunamh_jax import alias, logistic
from tunamh_jax.data import Dataset
from tunamh_jax.runner import StepStats, random_walk_proposal


class Config(NamedTuple):
    x: jnp.ndarray  # (n, d)
    y_sign: jnp.ndarray  # (n,) in {-1, +1}
    c: jnp.ndarray  # (n,) Assumption 1 constants c_i
    total_c: jnp.ndarray  # C = sum_i c_i
    table: alias.AliasTable
    stepsize: jnp.ndarray
    temperature: jnp.ndarray


class State(NamedTuple):
    theta: jnp.ndarray  # (d,)


def build(
    dataset: Dataset,
    *,
    stepsize: float = 1e-4,
    temperature: float = 1.0,
    c1: float = 1.0,
) -> Config:
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    x = jnp.asarray(dataset.train_x, dtype)
    c = logistic.lipschitz_constants(x, temperature, c1)
    return Config(
        x=x,
        y_sign=jnp.asarray(2 * dataset.train_y - 1, dtype),
        c=c,
        total_c=jnp.sum(c),
        table=alias.build(jax.device_get(c), dtype),
        stepsize=jnp.asarray(stepsize, dtype),
        temperature=jnp.asarray(temperature, dtype),
    )


def init_state(cfg: Config, key: jax.Array) -> State:
    return State(theta=logistic.kaiming_uniform_init(key, cfg.x.shape[1], cfg.x.dtype))


def rejection_probability(ci_m: jnp.ndarray, energy_delta: jnp.ndarray) -> jnp.ndarray:
    """``max(0, U_i(θ') - U_i(θ)) / (c_i M)``, the per-factor rejection chance.

    Assumption 1 bounds this by 1.  Negative values are returned as-is because
    the uniform they are compared against is non-negative, so they never fire.
    """
    informative = ci_m > 0
    return jnp.where(informative, energy_delta / jnp.where(informative, ci_m, 1.0), 0.0)


class _Scan(NamedTuple):
    chunk_index: jnp.ndarray
    rejected: jnp.ndarray
    examined: jnp.ndarray


def _test_factors(
    cfg: Config,
    theta: jnp.ndarray,
    theta_prime: jnp.ndarray,
    m: jnp.ndarray,
    batch_size: jnp.ndarray,
    key: jax.Array,
    chunk: int,
) -> _Scan:
    offsets = jnp.arange(chunk)
    n_chunks = (batch_size + chunk - 1) // chunk

    def cond(carry: _Scan) -> jnp.ndarray:
        return (carry.chunk_index < n_chunks) & jnp.logical_not(carry.rejected)

    def body(carry: _Scan) -> _Scan:
        chunk_key = jax.random.fold_in(key, carry.chunk_index)
        key_index, key_coin = jax.random.split(chunk_key)
        valid = (carry.chunk_index * chunk + offsets) < batch_size

        idx = alias.sample(key_index, cfg.table, (chunk,))
        ci_m = m * cfg.c[idx]
        rows = cfg.x[idx]
        signs = cfg.y_sign[idx]
        energy_delta = (  # U_i(θ') - U_i(θ); only positive values can reject
            jax.nn.log_sigmoid(signs * (rows @ theta))
            - jax.nn.log_sigmoid(signs * (rows @ theta_prime))
        ) / cfg.temperature

        reject_prob = rejection_probability(ci_m, energy_delta)
        u = jax.random.uniform(key_coin, (chunk,), dtype=cfg.x.dtype)
        rejects = valid & (u < reject_prob)

        counted = jnp.where(
            jnp.any(rejects),
            # Stop counting at the index that rejected, as the reference did.
            jnp.argmax(rejects) + 1,
            jnp.sum(valid),
        )
        return _Scan(
            chunk_index=carry.chunk_index + 1,
            rejected=jnp.any(rejects),
            examined=carry.examined + counted.astype(carry.examined.dtype),
        )

    init = _Scan(
        chunk_index=jnp.zeros((), jnp.int32),
        rejected=jnp.zeros((), bool),
        examined=jnp.zeros((), jnp.int32),
    )
    return lax.while_loop(cond, body, init)


def step(
    cfg: Config, state: State, key: jax.Array, *, chunk: int = 512
) -> tuple[State, StepStats]:
    key_propose, key_poisson, key_batch = jax.random.split(key, 3)
    theta = state.theta

    theta_prime = random_walk_proposal(key_propose, theta, cfg.stepsize)
    m = jnp.linalg.norm(theta_prime - theta)
    batch_size = jax.random.poisson(key_poisson, m * cfg.total_c)

    outcome = _test_factors(cfg, theta, theta_prime, m, batch_size, key_batch, chunk)
    accepted = jnp.logical_not(outcome.rejected)
    return State(theta=jnp.where(accepted, theta_prime, theta)), StepStats(
        accepted=accepted,
        batch_size=outcome.examined,
        minibatch_size=outcome.examined,
    )
