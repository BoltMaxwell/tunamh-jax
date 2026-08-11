"""Chain driver and the small pieces every sampler shares.

A sampler is a pure function ``step(cfg, state, key) -> (state, StepStats)``.
``run_chain`` scans that step inside a single jitted block, so the Python
interpreter is touched once per *checkpoint* rather than once per iteration.

Memory: the reference stored every sample (600k x 50 float64 = 240 MB for the
TFMH default).  Here the running sum needed for the posterior-mean estimate is
carried inside the scan, so the default run is O(1) in the number of
iterations; pass ``store_samples=True`` if you actually want the full trace.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax


class StepStats(NamedTuple):
    """What one transition cost and whether it moved."""

    accepted: jnp.ndarray  # bool scalar
    batch_size: jnp.ndarray  # per-datum energy evaluations this step
    minibatch_size: jnp.ndarray  # terms that entered the acceptance ratio


def random_walk_proposal(key: jax.Array, theta: jnp.ndarray, stepsize) -> jnp.ndarray:
    """Symmetric Gaussian random walk, so the proposal ratio q(θ|θ')/q(θ'|θ) is 1."""
    return theta + stepsize * jax.random.normal(key, theta.shape, theta.dtype)


def metropolis_accept(key: jax.Array, log_ratio: jnp.ndarray) -> jnp.ndarray:
    """``u < min(1, exp(log_ratio))``, tested in log space so it cannot overflow."""
    u = jax.random.uniform(key, (), dtype=jnp.result_type(log_ratio))
    return jnp.log(u) < log_ratio


@dataclass(frozen=True)
class Checkpoint:
    iteration: int
    elapsed_seconds: float
    data_used: float
    accept_rate: float
    monitor: float | None = None


@dataclass
class ChainResult:
    theta_mean: jnp.ndarray
    final_state: object
    accept_rate: float
    avg_batch_size: float
    avg_minibatch_size: float
    total_seconds: float
    compile_seconds: float
    checkpoints: list[Checkpoint] = field(default_factory=list)
    samples: jnp.ndarray | None = None

    @property
    def accuracy(self) -> float | None:
        return self.checkpoints[-1].monitor if self.checkpoints else None


class _Accum(NamedTuple):
    state: object
    theta_sum: jnp.ndarray
    n_accept: jnp.ndarray
    n_batch: jnp.ndarray
    n_minibatch: jnp.ndarray


@partial(jax.jit, static_argnums=(0, 4, 5))
def _run_block(step, cfg, state, key, n_steps: int, store_samples: bool):
    """Scan ``n_steps`` transitions, accumulating the running sum of theta."""
    real = jnp.result_type(state.theta)

    def body(carry: _Accum, step_key):
        new_state, stats = step(cfg, carry.state, step_key)
        carry = _Accum(
            state=new_state,
            theta_sum=carry.theta_sum + new_state.theta,
            n_accept=carry.n_accept + stats.accepted.astype(real),
            n_batch=carry.n_batch + stats.batch_size.astype(real),
            n_minibatch=carry.n_minibatch + stats.minibatch_size.astype(real),
        )
        return carry, (new_state.theta if store_samples else None)

    zero = jnp.zeros((), real)
    init = _Accum(state, jnp.zeros_like(state.theta), zero, zero, zero)
    return lax.scan(body, init, jax.random.split(key, n_steps))


def _block_sizes(total: int, interval: int) -> list[int]:
    if total <= 0:
        return []
    full, remainder = divmod(total, interval)
    return [interval] * full + ([remainder] if remainder else [])


def run_chain(
    step: Callable,
    cfg,
    state,
    key: jax.Array,
    *,
    n_samples: int,
    burnin: int = 0,
    interval: int = 100,
    monitor: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    store_samples: bool = False,
    warmup: bool = True,
    progress: Callable[[Checkpoint], None] | None = None,
) -> ChainResult:
    """Run ``burnin + n_samples`` transitions, checkpointing every ``interval``.

    Only post-burn-in states enter the running mean, the accuracy monitor and
    the reported acceptance rate.

    Timing excludes compilation: with ``warmup=True`` a throwaway block of the
    same shape is compiled and run first, and its cost is reported separately in
    ``compile_seconds``.
    """
    if interval <= 0:
        raise ValueError("interval must be positive")
    real = jnp.result_type(state.theta)

    compile_seconds = 0.0
    if warmup:
        # Derive the warm-up key by folding rather than splitting, so `key` is
        # not advanced and the chain matches a run with warmup=False bit for bit.
        warm_key = jax.random.fold_in(key, 0)
        started = time.perf_counter()
        jax.block_until_ready(_run_block(step, cfg, state, warm_key, interval, store_samples))
        compile_seconds = time.perf_counter() - started

    for size in _block_sizes(burnin, interval):
        key, block_key = jax.random.split(key)
        carry, _ = _run_block(step, cfg, state, block_key, size, False)
        state = carry.state

    theta_sum = jnp.zeros_like(state.theta)
    n_accept = jnp.zeros((), real)
    n_batch = jnp.zeros((), real)
    n_minibatch = jnp.zeros((), real)
    stored: list[jnp.ndarray] = []
    checkpoints: list[Checkpoint] = []
    elapsed = 0.0
    done = 0

    for size in _block_sizes(n_samples, interval):
        key, block_key = jax.random.split(key)
        started = time.perf_counter()
        carry, trace = _run_block(step, cfg, state, block_key, size, store_samples)
        jax.block_until_ready(carry)
        elapsed += time.perf_counter() - started

        state = carry.state
        theta_sum = theta_sum + carry.theta_sum
        n_accept = n_accept + carry.n_accept
        n_batch = n_batch + carry.n_batch
        n_minibatch = n_minibatch + carry.n_minibatch
        if store_samples:
            stored.append(trace)
        done += size

        theta_mean = theta_sum / done
        checkpoint = Checkpoint(
            iteration=done,
            elapsed_seconds=elapsed,
            data_used=float(n_batch),
            accept_rate=float(n_accept) / done,
            monitor=None if monitor is None else float(monitor(theta_mean)),
        )
        checkpoints.append(checkpoint)
        if progress is not None:
            progress(checkpoint)

    total = max(done, 1)
    return ChainResult(
        theta_mean=theta_sum / total,
        final_state=state,
        accept_rate=float(n_accept) / total,
        avg_batch_size=float(n_batch) / total,
        avg_minibatch_size=float(n_minibatch) / total,
        total_seconds=elapsed,
        compile_seconds=compile_seconds,
        checkpoints=checkpoints,
        samples=jnp.concatenate(stored, axis=0) if stored else None,
    )
