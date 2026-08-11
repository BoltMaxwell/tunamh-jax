"""FlyMC: Firefly Monte Carlo (Maclaurin & Adams, 2014).

Exact but *stateful*: an auxiliary Bernoulli variable per data point marks it
"bright" or "dark".  Dark points contribute only a cheap lower bound ``B_n(θ)``
to the joint target, so the MH ratio sums over the bright set alone.  The
per-point bound here is the quadratic Jaakkola-Jordan bound with ``xi = 1.5``,
matching the reference.

Labels are folded into the design matrix as ``x_i <- (2 y_i - 1) x_i`` so the
log-likelihood is just ``log sigmoid(x_i . θ)``.

Vectorisation note: the reference walked the dark set with geometric skips so
it evaluated the bound for only a ``q_db`` fraction of dark points.  Visiting
with probability ``q_db`` and then accepting with probability
``min(1, exp(lbr)/q_db)`` is the same Bernoulli as a single test against
``min(q_db, exp(lbr))``, so the flip is drawn here with one uniform per point
and the whole mask is updated at once.  The chain is distributionally identical
and the reported batch size (the bright count) is unchanged; what changes is
that this port evaluates the bound for all N points per step -- one matvec --
instead of skipping.  The bright-set sums that actually dominate the MH ratio
are the same either way.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from tunamh_jax import logistic
from tunamh_jax.data import Dataset
from tunamh_jax.runner import StepStats, metropolis_accept, random_walk_proposal

_EPS = 1e-6
_XI = 1.5


class Config(NamedTuple):
    x: jnp.ndarray  # (n, d), already multiplied by (2y - 1)
    quad: jnp.ndarray  # (d, d) = a * X^T X
    linear: jnp.ndarray  # (d,)  = b * sum_i x_i
    stepsize: jnp.ndarray
    temperature: jnp.ndarray
    q_db: jnp.ndarray  # bright -> dark proposal probability
    a: jnp.ndarray
    b: jnp.ndarray
    c: jnp.ndarray


class State(NamedTuple):
    theta: jnp.ndarray  # (d,)
    bright: jnp.ndarray  # (n,) bool


def bound_coefficients(xi: float = _XI, eps: float = _EPS) -> tuple[float, float, float]:
    """Coefficients of the quadratic lower bound ``log B(z) = a z^2 + b z + c``."""
    pexp, nexp = math.exp(xi), math.exp(-xi)
    f = pexp + nexp
    a = -0.25 / xi * (pexp - nexp) / (2 + f)
    b = 0.5
    c = -a * xi**2 - 0.5 * math.log(2 + f) - eps
    return a, b, c


def build(
    dataset: Dataset,
    *,
    stepsize: float = 2e-3,
    temperature: float = 1.0,
    q_db: float = 1e-1,
) -> Config:
    if not 0.0 < q_db <= 1.0:
        raise ValueError("q_db must lie in (0, 1]")
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    signs = (2 * dataset.train_y - 1).reshape(-1, 1)
    x = jnp.asarray(dataset.train_x * signs, dtype)
    a, b, c = bound_coefficients()
    return Config(
        x=x,
        quad=a * (x.T @ x),
        linear=b * jnp.sum(x, axis=0),
        stepsize=jnp.asarray(stepsize, dtype),
        temperature=jnp.asarray(temperature, dtype),
        q_db=jnp.asarray(q_db, dtype),
        a=jnp.asarray(a, dtype),
        b=jnp.asarray(b, dtype),
        c=jnp.asarray(c, dtype),
    )


def init_state(cfg: Config, key: jax.Array) -> State:
    """Start with every point dark, as the reference does."""
    return State(
        theta=logistic.kaiming_uniform_init(key, cfg.x.shape[1], cfg.x.dtype),
        bright=jnp.zeros(cfg.x.shape[0], dtype=bool),
    )


def log_bright_dark_ratio(cfg: Config, theta: jnp.ndarray) -> jnp.ndarray:
    """``log(L_i(θ)/B_i(θ) - 1)`` for every point: the weight a bright point adds."""
    z = cfg.x @ theta
    log_likelihood = jax.nn.log_sigmoid(z)
    log_bound = cfg.a * z**2 + cfg.b * z + cfg.c
    # The bound construction guarantees gap >= 0; the max absorbs rounding.
    gap = jnp.maximum((log_likelihood - log_bound) / cfg.temperature, 0.0)
    return gap + logistic.log1mexp(gap)


def resample_brightness(
    cfg: Config, theta: jnp.ndarray, bright: jnp.ndarray, key: jax.Array, log_ratio: jnp.ndarray
) -> jnp.ndarray:
    """One Gibbs sweep over the auxiliary variables at fixed ``θ``."""
    log_u = jnp.log(jax.random.uniform(key, bright.shape, dtype=cfg.x.dtype))
    log_q = jnp.log(cfg.q_db)
    to_dark = bright & (log_u < log_q - log_ratio)
    to_bright = jnp.logical_not(bright) & (log_u < jnp.minimum(log_q, log_ratio))
    return (bright & jnp.logical_not(to_dark)) | to_bright


def _log_target(
    cfg: Config, theta: jnp.ndarray, bright: jnp.ndarray, log_ratio: jnp.ndarray
) -> jnp.ndarray:
    """Bright-set log-likelihood plus the log of the product of all bounds.

    The ``Σ_i c`` part of the bound product is constant in θ and cancels in the
    MH ratio, so only the quadratic and linear terms are carried.
    """
    bright_sum = jnp.sum(jnp.where(bright, log_ratio, 0.0))
    bound_total = theta @ (cfg.quad @ theta) + cfg.linear @ theta
    return bright_sum + bound_total / cfg.temperature


def step(cfg: Config, state: State, key: jax.Array) -> tuple[State, StepStats]:
    key_propose, key_bright, key_accept = jax.random.split(key, 3)
    theta = state.theta

    theta_prime = random_walk_proposal(key_propose, theta, cfg.stepsize)

    # Auxiliary variables are resampled at the current θ, before the MH test.
    log_ratio_theta = log_bright_dark_ratio(cfg, theta)
    bright = resample_brightness(cfg, theta, state.bright, key_bright, log_ratio_theta)

    log_ratio_prime = log_bright_dark_ratio(cfg, theta_prime)
    log_mh = _log_target(cfg, theta_prime, bright, log_ratio_prime) - _log_target(
        cfg, theta, bright, log_ratio_theta
    )

    accepted = metropolis_accept(key_accept, log_mh)
    n_bright = jnp.sum(bright)
    return State(theta=jnp.where(accepted, theta_prime, theta), bright=bright), StepStats(
        accepted=accepted,
        batch_size=n_bright,
        minibatch_size=n_bright,
    )
