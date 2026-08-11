"""Alias-table sampling from a fixed categorical distribution.

TunaMH and TFMH draw each minibatch index from a distribution gamma_i = c_i / C
that is fixed for the whole chain, so it pays to build Vose's alias table once
on the host and then draw in O(1) per index rather than O(N).

The table is built in NumPy (host, once) and consumed as two device arrays.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class AliasTable(NamedTuple):
    """Vose alias table. ``prob[j]`` is the probability of keeping bucket ``j``."""

    prob: jnp.ndarray  # (n,) float
    alias: jnp.ndarray  # (n,) int32


def build(weights: np.ndarray, dtype: np.dtype | None = None) -> AliasTable:
    """Build an alias table for ``weights`` (non-negative, need not be normalised)."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size == 0:
        raise ValueError("weights must be non-empty")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and non-negative")
    total = w.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive value")

    n = w.size
    scaled = w * (n / total)
    prob = np.ones(n, dtype=np.float64)
    alias = np.arange(n, dtype=np.int32)

    small = [i for i in range(n) if scaled[i] < 1.0]
    large = [i for i in range(n) if scaled[i] >= 1.0]
    while small and large:
        s = small.pop()
        l = large.pop()
        prob[s] = scaled[s]
        alias[s] = l
        scaled[l] = (scaled[l] + scaled[s]) - 1.0
        (small if scaled[l] < 1.0 else large).append(l)
    # Whatever is left is 1.0 up to rounding; keep those buckets pure.
    for i in large + small:
        prob[i] = 1.0
        alias[i] = i

    if dtype is None:
        dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
    return AliasTable(prob=jnp.asarray(prob, dtype=dtype), alias=jnp.asarray(alias))


def sample(key: jax.Array, table: AliasTable, shape: tuple[int, ...]) -> jnp.ndarray:
    """Draw indices of the given ``shape`` from the categorical distribution."""
    k_bucket, k_coin = jax.random.split(key)
    j = jax.random.randint(k_bucket, shape, 0, table.prob.shape[0])
    u = jax.random.uniform(k_coin, shape, dtype=table.prob.dtype)
    return jnp.where(u < table.prob[j], j, table.alias[j])


def probabilities(table: AliasTable) -> jnp.ndarray:
    """Recover the categorical distribution the table encodes (for testing)."""
    n = table.prob.shape[0]
    kept = table.prob / n
    spilled = jnp.zeros(n, dtype=table.prob.dtype).at[table.alias].add((1.0 - table.prob) / n)
    return kept + spilled
