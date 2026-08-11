"""TunaMH: asymptotically optimal exact minibatch Metropolis-Hastings.

Algorithm 2 of Zhang, Cooper & De Sa (NeurIPS 2020).  With ``M = M(θ, θ')``,
``c_i`` the Assumption-1 Lipschitz constants and ``C = Σ c_i``, one step is:

1. propose ``θ' ~ q(·|θ)``;
2. draw a batch size ``B ~ Poisson(χ C² M² + C M)``;
3. draw ``B`` indices with ``P(i) = c_i / C`` and keep index ``i`` with
   probability ``(χ c_i C M² + φ_i⁻) / (χ c_i C M² + c_i M)``, where
   ``φ_i⁻ = ½(U_i(θ') - U_i(θ) + c_i M)``;
4. accept with probability ``min(1, r)`` where
   ``log r = Σ_{i∈I} 2 artanh( (U_i(θ) - U_i(θ')) / (c_i M (1 + 2χ C M)) )``.

Two rewrites of the paper's formulas are used here.  Writing
``φ_i⁺ = ½(U_i(θ) - U_i(θ') + c_i M)`` (so ``φ_i⁺ + φ_i⁻ = c_i M``) and
``q_i = χ C M c_i M``, the keep probability is exactly ``(q_i + φ_i⁻)/(q_i + c_i M)``
and the log-ratio term is exactly ``log(q_i + φ_i⁺) - log(q_i + φ_i⁻)``.  Both
are algebraically identical to the paper (and to the Julia reference, which
used ``log(1 + L φ/(λ M_i))``) but neither divides by a quantity that vanishes
as ``χ → 0`` or ``M → 0``, so they stay finite for small steps.

Shape handling: ``B`` is random, and JAX needs static shapes.  Rather than
truncate at some ``B_max`` -- which would silently break exactness -- the batch
is consumed in fixed-size chunks inside a ``fori_loop`` whose trip count
``ceil(B / chunk)`` is a traced value.  Every draw is used, so the sampler
remains exact, and with ``chunk`` at or above the typical ``B`` the loop runs
once per step.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from tunamh_jax import alias, logistic
from tunamh_jax.data import Dataset
from tunamh_jax.runner import StepStats, metropolis_accept, random_walk_proposal


class Config(NamedTuple):
    x: jnp.ndarray  # (n, d)
    y_sign: jnp.ndarray  # (n,) in {-1, +1}
    c: jnp.ndarray  # (n,) Assumption 1 constants c_i
    total_c: jnp.ndarray  # C = sum_i c_i
    table: alias.AliasTable  # samples i with probability c_i / C
    stepsize: jnp.ndarray
    temperature: jnp.ndarray
    chi: jnp.ndarray  # batch-size / convergence-rate dial


class State(NamedTuple):
    theta: jnp.ndarray  # (d,)


def build(
    dataset: Dataset,
    *,
    stepsize: float = 1e-3,
    chi: float = 1e-5,
    temperature: float = 1.0,
    c1: float = 1.0,
) -> Config:
    if chi <= 0:
        raise ValueError("chi must be positive; chi -> 0 degenerates the acceptance ratio")
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
        chi=jnp.asarray(chi, dtype),
    )


def init_state(cfg: Config, key: jax.Array) -> State:
    return State(theta=logistic.kaiming_uniform_init(key, cfg.x.shape[1], cfg.x.dtype))


def minibatch_terms(
    q: jnp.ndarray, ci_m: jnp.ndarray, energy_delta: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-index keep probability and log-ratio contribution.

    ``q = χ C M c_i M``, ``ci_m = c_i M`` and ``energy_delta = U_i(θ') - U_i(θ)``.
    Returns ``(keep_prob, contribution)`` where

        keep_prob    = (q + φ⁻) / (q + c_i M)
        contribution = log(q + φ⁺) - log(q + φ⁻)

    with ``φ∓ = ½(±(U_i(θ) - U_i(θ')) + c_i M)``.  See the module docstring for
    why these equal the paper's expressions.  Points with ``c_i = 0`` have a
    constant ``U_i``, contribute nothing, and are never kept.
    """
    informative = ci_m > 0
    # Assumption 1 bounds |energy_delta| by c_i M, so phi_low lands in
    # [0, c_i M]; the clip only absorbs rounding at the endpoints.
    phi_low = jnp.clip(0.5 * (energy_delta + ci_m), 0.0, ci_m)
    phi_high = ci_m - phi_low

    safe_low = jnp.where(informative, q + phi_low, 1.0)
    safe_high = jnp.where(informative, q + phi_high, 1.0)
    safe_total = jnp.where(informative, q + ci_m, 1.0)
    keep_prob = jnp.where(informative, safe_low / safe_total, 0.0)
    contribution = jnp.where(informative, jnp.log(safe_high) - jnp.log(safe_low), 0.0)
    return keep_prob, contribution


class _Batch(NamedTuple):
    log_ratio: jnp.ndarray
    kept: jnp.ndarray


def _accumulate(
    cfg: Config,
    theta: jnp.ndarray,
    theta_prime: jnp.ndarray,
    m: jnp.ndarray,
    cm: jnp.ndarray,
    batch_size: jnp.ndarray,
    key: jax.Array,
    chunk: int,
) -> _Batch:
    """Steps 3 and 4: form the minibatch and sum its log-ratio contributions."""
    dtype = cfg.x.dtype
    offsets = jnp.arange(chunk)

    def body(i, carry: _Batch) -> _Batch:
        chunk_key = jax.random.fold_in(key, i)
        key_index, key_keep = jax.random.split(chunk_key)
        # The last chunk is partly padding; padded lanes are computed then masked.
        valid = (i * chunk + offsets) < batch_size

        idx = alias.sample(key_index, cfg.table, (chunk,))
        ci_m = m * cfg.c[idx]  # c_i M
        q = cfg.chi * cm * ci_m  # χ C M c_i M

        rows = cfg.x[idx]
        signs = cfg.y_sign[idx]
        energy_delta = (  # U_i(θ') - U_i(θ)
            jax.nn.log_sigmoid(signs * (rows @ theta))
            - jax.nn.log_sigmoid(signs * (rows @ theta_prime))
        ) / cfg.temperature

        keep_prob, contribution = minibatch_terms(q, ci_m, energy_delta)
        u = jax.random.uniform(key_keep, (chunk,), dtype=dtype)
        keep = valid & (u <= keep_prob)

        return _Batch(
            log_ratio=carry.log_ratio + jnp.sum(jnp.where(keep, contribution, 0.0)),
            kept=carry.kept + jnp.sum(keep, dtype=carry.kept.dtype),
        )

    n_chunks = (batch_size + chunk - 1) // chunk
    init = _Batch(log_ratio=jnp.zeros((), dtype), kept=jnp.zeros((), jnp.int32))
    return lax.fori_loop(0, n_chunks, body, init)


def step(
    cfg: Config, state: State, key: jax.Array, *, chunk: int = 512
) -> tuple[State, StepStats]:
    key_propose, key_poisson, key_batch, key_accept = jax.random.split(key, 4)
    theta = state.theta

    theta_prime = random_walk_proposal(key_propose, theta, cfg.stepsize)
    m = jnp.linalg.norm(theta_prime - theta)  # M(θ, θ') for the Euclidean bound
    cm = m * cfg.total_c  # C M
    rate = cfg.chi * cm**2 + cm  # χ C² M² + C M
    batch_size = jax.random.poisson(key_poisson, rate)

    batch = _accumulate(cfg, theta, theta_prime, m, cm, batch_size, key_batch, chunk)
    accepted = metropolis_accept(key_accept, batch.log_ratio)
    return State(theta=jnp.where(accepted, theta_prime, theta)), StepStats(
        accepted=accepted,
        batch_size=batch_size,
        minibatch_size=batch.kept,
    )
