import jax
import numpy as np
import pytest

import tunamh_jax  # noqa: F401  -- enables x64 at import
from tunamh_jax.data import Dataset


@pytest.fixture(scope="session", autouse=True)
def _require_x64():
    assert jax.config.jax_enable_x64, "the samplers assume float64"


def make_dataset(n: int = 40, dim: int = 1, seed: int = 0) -> Dataset:
    """A small, deliberately non-separable logistic problem.

    Both signs of ``(2y-1) x`` must appear or the likelihood does not decay and
    the posterior is improper, which would make the quadrature reference
    meaningless.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim))
    y = rng.integers(0, 2, size=n).astype(np.int32)
    test_x = rng.normal(size=(n, dim))
    test_y = rng.integers(0, 2, size=n).astype(np.int32)
    return Dataset(x, y, test_x, test_y)


def quadrature_posterior_moments(
    dataset: Dataset, temperature: float = 1.0, limit: float = 60.0
) -> tuple[float, float]:
    """Exact mean and standard deviation of a 1-D flat-prior logistic target."""
    assert dataset.train_x.shape[1] == 1
    grid = np.linspace(-limit, limit, 600_001)
    z = np.outer(grid, dataset.train_x[:, 0])
    signs = 2 * dataset.train_y - 1
    log_target = np.sum(-np.logaddexp(0.0, -signs * z), axis=1) / temperature
    weights = np.exp(log_target - log_target.max())
    norm = np.trapezoid(weights, grid)
    mean = np.trapezoid(weights * grid, grid) / norm
    second = np.trapezoid(weights * grid**2, grid) / norm
    return float(mean), float(np.sqrt(second - mean**2))
