"""Command line entry point: ``tunamh <sampler> [options]``.

Mirrors the reference scripts (``julia tunamh.jl`` and friends), including
their per-sampler defaults, and adds JSON output for the checkpoint traces.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp

from tunamh_jax import data, logistic
from tunamh_jax.runner import Checkpoint, run_chain
from tunamh_jax.samplers import flymc, mh, tfmh, tunamh

# Per-sampler defaults, taken from the reference argument tables.  ``chunk`` is
# new here: it should sit just above the sampler's typical Poisson batch size so
# the minibatch loop runs once per step without wasting masked-out lanes.  At
# these stepsizes TunaMH draws ~500 indices per step and TFMH ~50.
DEFAULTS = {
    "mh": {"stepsize": 5e-3, "nsamples": 4_000, "interval": 100},
    "tunamh": {"stepsize": 1e-3, "nsamples": 45_000, "interval": 100, "chi": 1e-5, "chunk": 512},
    "tfmh": {"stepsize": 1e-4, "nsamples": 600_000, "interval": 10_000, "chunk": 64},
    "flymc": {"stepsize": 2e-3, "nsamples": 5_000, "interval": 500, "q_db": 1e-1},
}

MODULES = {"mh": mh, "tunamh": tunamh, "tfmh": tfmh, "flymc": flymc}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tunamh",
        description="Exact minibatch Metropolis-Hastings on MNIST 7-vs-9 logistic regression.",
    )
    parser.add_argument("sampler", choices=sorted(MODULES), help="which sampler to run")
    parser.add_argument("--pca-dim", type=int, default=50, help="PCA dimension (default: 50)")
    parser.add_argument("--stepsize", type=float, help="random-walk proposal scale")
    parser.add_argument("--temperature", "-T", type=float, default=1.0, help="target temperature")
    parser.add_argument("--nsamples", type=int, help="post-burn-in iterations")
    parser.add_argument("--burnin", type=int, default=0, help="discarded iterations")
    parser.add_argument("--interval", type=int, help="iterations between checkpoints")
    parser.add_argument(
        "--chi",
        "--lam",
        type=float,
        dest="chi",
        help="TunaMH batch-size / convergence-rate dial (reference calls it lam)",
    )
    parser.add_argument("--q-db", type=float, help="FlyMC bright-to-dark proposal probability")
    parser.add_argument(
        "--chunk", type=int, help="minibatch chunk width for TunaMH/TFMH"
    )
    parser.add_argument("--seed", type=int, default=222, help="PRNG seed (default: 222)")
    parser.add_argument("--data-dir", type=Path, help="where MNIST idx.gz files live")
    parser.add_argument(
        "--synthetic",
        type=int,
        metavar="N",
        help="skip MNIST and use an N-point synthetic logistic problem",
    )
    parser.add_argument("--store-samples", action="store_true", help="keep the full trace")
    parser.add_argument("--out", type=Path, help="write results as JSON to this path")
    parser.add_argument("--quiet", action="store_true", help="suppress per-checkpoint output")
    return parser


def _resolve(args: argparse.Namespace) -> dict:
    settings = dict(DEFAULTS[args.sampler])
    for name in ("stepsize", "nsamples", "interval", "chi", "q_db", "chunk"):
        value = getattr(args, name, None)
        if value is not None:
            settings[name] = value
    return settings


def run(args: argparse.Namespace) -> dict:
    settings = _resolve(args)
    module = MODULES[args.sampler]

    if args.synthetic:
        dataset = data.synthetic_logistic(args.synthetic, args.pca_dim, seed=args.seed)
    else:
        dataset = data.load_mnist_pair(args.pca_dim, data_dir=args.data_dir)
    if not args.quiet:
        print(
            f"{args.sampler}: n_train={dataset.n_train} n_test={dataset.n_test} "
            f"dim={dataset.dim} device={jax.devices()[0].device_kind}"
        )

    build_kwargs = {"stepsize": settings["stepsize"], "temperature": args.temperature}
    if args.sampler == "tunamh":
        build_kwargs["chi"] = settings["chi"]
    if args.sampler == "flymc":
        build_kwargs["q_db"] = settings["q_db"]
    cfg = module.build(dataset, **build_kwargs)

    key = jax.random.key(args.seed)
    key_init, key_chain = jax.random.split(key)
    state = module.init_state(cfg, key_init)

    step = module.step
    if args.sampler in ("tunamh", "tfmh"):
        step = partial(module.step, chunk=settings["chunk"])

    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    test_x = jnp.asarray(dataset.test_x, dtype)
    test_y = jnp.asarray(dataset.test_y)
    monitor = jax.jit(lambda theta: logistic.accuracy(theta, test_x, test_y))

    def report(checkpoint: Checkpoint) -> None:
        print(
            f"  iter {checkpoint.iteration:>8}  {checkpoint.elapsed_seconds:7.2f}s  "
            f"acc={checkpoint.monitor:.4f}  accept={checkpoint.accept_rate:.4f}  "
            f"data={checkpoint.data_used:.3g}"
        )

    result = run_chain(
        step,
        cfg,
        state,
        key_chain,
        n_samples=settings["nsamples"],
        burnin=args.burnin,
        interval=settings["interval"],
        monitor=monitor,
        store_samples=args.store_samples,
        progress=None if args.quiet else report,
    )

    summary = {
        "sampler": args.sampler,
        "settings": settings | {"temperature": args.temperature, "burnin": args.burnin},
        "accuracy": result.accuracy,
        "runtime_seconds": result.total_seconds,
        "compile_seconds": result.compile_seconds,
        "avg_batch_size": result.avg_batch_size,
        "avg_minibatch_size": result.avg_minibatch_size,
        "avg_accept_prob": result.accept_rate,
        "checkpoints": [vars(c) for c in result.checkpoints],
    }
    print(f"accuracy: {summary['accuracy']}")
    print(f"runtime: {summary['runtime_seconds']}")
    print(f"Avg Batch Size: {summary['avg_batch_size']}")
    print(f"Avg Acceptance Prob: {summary['avg_accept_prob']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out}")
    return summary


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
