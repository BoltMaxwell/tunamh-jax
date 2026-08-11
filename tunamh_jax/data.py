"""MNIST 7-vs-9 logistic-regression dataset, PCA-reduced.

This reproduces the reference pipeline: take the MNIST digits 7 and 9, relabel
them 0 and 1, flatten to 784 features scaled to [0, 1], fit PCA on the training
split, and project both splits onto the leading components.

All of this is host-side NumPy that runs once before any sampler starts, so it
deliberately lives outside every jitted region.
"""

from __future__ import annotations

import gzip
import hashlib
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_DATA_DIR = Path.home() / ".cache" / "tunamh-jax" / "mnist"

_MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)

_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


@dataclass(frozen=True)
class Dataset:
    """PCA-projected 7-vs-9 split. ``x`` is ``(n, d)``, ``y`` is ``(n,)`` in {0, 1}."""

    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray

    @property
    def dim(self) -> int:
        return self.train_x.shape[1]

    @property
    def n_train(self) -> int:
        return self.train_x.shape[0]

    @property
    def n_test(self) -> int:
        return self.test_x.shape[0]


def _download(name: str, data_dir: Path) -> Path:
    path = data_dir / name
    if path.exists():
        return path
    data_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    for mirror in _MIRRORS:
        url = mirror + name
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
        except Exception as exc:  # noqa: BLE001 - report every mirror that failed
            errors.append(f"{url}: {exc}")
            continue
        tmp = path.with_suffix(path.suffix + ".partial")
        tmp.write_bytes(payload)
        tmp.replace(path)
        return path
    raise RuntimeError(
        f"could not download {name}; tried:\n  " + "\n  ".join(errors) +
        f"\nPlace the four MNIST idx.gz files in {data_dir} manually to work offline."
    )


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic == 0x803:  # images
            rows, cols = struct.unpack(">II", handle.read(8))
            shape = (count, rows * cols)
        elif magic == 0x801:  # labels
            shape = (count,)
        else:
            raise ValueError(f"{path} is not an MNIST idx file (magic 0x{magic:x})")
        return np.frombuffer(handle.read(), dtype=np.uint8).reshape(shape)


def load_raw_mnist(data_dir: Path | str | None = None) -> dict[str, np.ndarray]:
    """Return the four raw MNIST arrays, downloading them on first use."""
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    return {key: _read_idx(_download(name, directory)) for key, name in _FILES.items()}


def _select_pair(
    images: np.ndarray, labels: np.ndarray, negative: int, positive: int
) -> tuple[np.ndarray, np.ndarray]:
    keep = np.flatnonzero((labels == negative) | (labels == positive))
    x = images[keep].astype(np.float64) / 255.0
    y = (labels[keep] == positive).astype(np.int32)
    return x, y


def fit_pca(
    x: np.ndarray, n_components: int, variance_ratio: float = 0.99
) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA on ``(n, d)`` data, returning ``(mean, components)``.

    ``components`` is ``(d, k)`` with columns ordered by descending explained
    variance.  ``variance_ratio`` mirrors ``MultivariateStats.jl``'s ``pratio``:
    the reference keeps the fewest components that explain that fraction of the
    total variance, capped at ``n_components``.  For MNIST 7-vs-9 the cap is
    what binds, so this defaults to a no-op, but keeping it makes the port
    faithful for other datasets.
    """
    mean = x.mean(axis=0)
    centered = x - mean
    # Economy SVD of the centered data: right singular vectors are the axes.
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    explained = singular_values**2
    total = explained.sum()
    if total <= 0:
        raise ValueError("data has zero variance; cannot fit PCA")
    cumulative = np.cumsum(explained) / total
    keep = min(n_components, int(np.searchsorted(cumulative, variance_ratio) + 1))
    return mean, vt[:keep].T


def load_mnist_pair(
    n_components: int = 50,
    *,
    negative_digit: int = 7,
    positive_digit: int = 9,
    variance_ratio: float = 0.99,
    data_dir: Path | str | None = None,
    cache: bool = True,
) -> Dataset:
    """Load the PCA-projected MNIST two-digit problem used in the paper."""
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    tag = hashlib.sha1(
        f"{negative_digit}v{positive_digit}-{n_components}-{variance_ratio}".encode()
    ).hexdigest()[:12]
    cache_path = directory / f"pca-{tag}.npz"
    if cache and cache_path.exists():
        with np.load(cache_path) as stored:
            return Dataset(**{k: stored[k] for k in ("train_x", "train_y", "test_x", "test_y")})

    raw = load_raw_mnist(directory)
    train_x, train_y = _select_pair(
        raw["train_images"], raw["train_labels"], negative_digit, positive_digit
    )
    test_x, test_y = _select_pair(
        raw["test_images"], raw["test_labels"], negative_digit, positive_digit
    )
    mean, components = fit_pca(train_x, n_components, variance_ratio)
    dataset = Dataset(
        train_x=(train_x - mean) @ components,
        train_y=train_y,
        test_x=(test_x - mean) @ components,
        test_y=test_y,
    )
    if cache:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **vars(dataset))
    return dataset


def synthetic_logistic(
    n: int, dim: int, seed: int = 0, scale: float = 1.0
) -> Dataset:
    """Small synthetic logistic problem, for tests and smoke runs without MNIST."""
    rng = np.random.default_rng(seed)
    theta_true = rng.normal(size=dim)
    train_x = scale * rng.normal(size=(n, dim))
    test_x = scale * rng.normal(size=(max(n // 4, 1), dim))

    def labels(x: np.ndarray) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-(x @ theta_true)))
        return (rng.uniform(size=x.shape[0]) < p).astype(np.int32)

    return Dataset(train_x, labels(train_x), test_x, labels(test_x))
