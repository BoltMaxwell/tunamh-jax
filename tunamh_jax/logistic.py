"""Binary logistic regression pieces shared by all samplers.

Design layout: ``X`` has shape ``(n, d)`` (one row per datum), which is the
transpose of the Julia reference's ``(d, n)``.  Row-major rows make the gathers
``X[idx]`` that dominate the minibatch samplers contiguous.

Labels are carried as *signs* ``s_i = 2 y_i - 1 in {-1, +1}`` because the
per-datum log-likelihood collapses to a single ``log_sigmoid`` call:

    log H(y_i, x_i . theta) = y log p + (1-y) log(1-p)
                            = log sigmoid(s_i * x_i . theta)

The Julia reference computed ``sigmoid`` and then took its log, which
overflows to ``-Inf`` once ``|x . theta|`` exceeds ~37.  ``log_sigmoid`` is the
same function evaluated stably, so it is used everywhere here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

LOG2 = 0.6931471805599453


def log_likelihood_terms(
    x: jnp.ndarray, y_sign: jnp.ndarray, theta: jnp.ndarray, temperature: float
) -> jnp.ndarray:
    """Per-datum tempered log-likelihood ``log H(y_i, x_i . theta) / T``."""
    return jax.nn.log_sigmoid(y_sign * (x @ theta)) / temperature


def log_likelihood(
    x: jnp.ndarray, y_sign: jnp.ndarray, theta: jnp.ndarray, temperature: float
) -> jnp.ndarray:
    """Total tempered log-likelihood over the rows of ``x``."""
    return jnp.sum(log_likelihood_terms(x, y_sign, theta, temperature))


def lipschitz_constants(x: jnp.ndarray, temperature: float, c1: float = 1.0) -> jnp.ndarray:
    """Assumption 1 constants ``c_i``.

    For ``U_i(theta) = -log H(y_i, x_i . theta) / T`` the gradient is
    ``(sigmoid(x_i . theta) - y_i) x_i / T``, whose norm is bounded by
    ``||x_i|| / T``.  So ``c_i = c1 * ||x_i|| / T`` is a valid Lipschitz
    constant and ``|U_i(theta) - U_i(theta')| <= c_i ||theta - theta'||``.
    """
    return c1 * jnp.linalg.norm(x, axis=1) / temperature


def kaiming_uniform_init(key: jax.Array, dim: int, dtype=None) -> jnp.ndarray:
    """PyTorch's default ``kaiming_uniform_(a=sqrt(5))`` bound, as in the reference."""
    a = jnp.sqrt(5.0)
    gain = jnp.sqrt(2.0 / (1.0 + a**2))
    bound = jnp.sqrt(3.0) * gain / jnp.sqrt(dim)
    if dtype is None:
        dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    return jax.random.uniform(key, (dim,), dtype=dtype, minval=-bound, maxval=bound)


def accuracy(theta: jnp.ndarray, test_x: jnp.ndarray, test_y: jnp.ndarray) -> jnp.ndarray:
    """Fraction of test points whose label matches ``sign(x . theta)``.

    Matches the reference's convention: ``predict > 0`` votes for class 1 and
    ``predict <= 0`` votes for class 0 (there is no bias term in this model).
    """
    predicted_positive = (test_x @ theta) > 0
    return jnp.mean(predicted_positive == (test_y == 1), dtype=test_x.dtype)


def log1mexp(x: jnp.ndarray) -> jnp.ndarray:
    """``log(1 - exp(-x))`` for ``x >= 0``, accurate at both ends of the range."""
    x = jnp.asarray(x)
    big = x > LOG2
    # Double-where keeps the unused branch off the NaN-producing inputs.
    x_big = jnp.where(big, x, 1.0)
    x_small = jnp.where(big, 1.0, x)
    return jnp.where(big, jnp.log1p(-jnp.exp(-x_big)), jnp.log(-jnp.expm1(-x_small)))
