"""Standard full-batch Metropolis-Hastings with a Gaussian random-walk proposal.

The baseline the minibatch methods are measured against: every step sums the
log-likelihood over all N data points.  In JAX that sum is a single ``(n, d) @
(d,)`` matvec, which is where this port differs most in character from the
reference's scalar Julia loop.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from tunamh_jax import logistic
from tunamh_jax.data import Dataset
from tunamh_jax.runner import StepStats, metropolis_accept, random_walk_proposal


class Config(NamedTuple):
    x: jnp.ndarray  # (n, d)
    y_sign: jnp.ndarray  # (n,) in {-1, +1}
    stepsize: jnp.ndarray
    temperature: jnp.ndarray


class State(NamedTuple):
    theta: jnp.ndarray  # (d,)


def build(dataset: Dataset, *, stepsize: float = 5e-3, temperature: float = 1.0) -> Config:
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    return Config(
        x=jnp.asarray(dataset.train_x, dtype),
        y_sign=jnp.asarray(2 * dataset.train_y - 1, dtype),
        stepsize=jnp.asarray(stepsize, dtype),
        temperature=jnp.asarray(temperature, dtype),
    )


def init_state(cfg: Config, key: jax.Array) -> State:
    return State(theta=logistic.kaiming_uniform_init(key, cfg.x.shape[1], cfg.x.dtype))


def step(cfg: Config, state: State, key: jax.Array) -> tuple[State, StepStats]:
    key_propose, key_accept = jax.random.split(key)
    theta_prime = random_walk_proposal(key_propose, state.theta, cfg.stepsize)

    log_ratio = logistic.log_likelihood(
        cfg.x, cfg.y_sign, theta_prime, cfg.temperature
    ) - logistic.log_likelihood(cfg.x, cfg.y_sign, state.theta, cfg.temperature)

    accepted = metropolis_accept(key_accept, log_ratio)
    theta = jnp.where(accepted, theta_prime, state.theta)
    n = jnp.asarray(cfg.x.shape[0], cfg.x.dtype)
    return State(theta=theta), StepStats(accepted=accepted, batch_size=n, minibatch_size=n)
