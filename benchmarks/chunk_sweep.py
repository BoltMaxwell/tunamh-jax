"""Throughput vs. minibatch chunk width for TunaMH and TFMH.

``chunk`` is the one tuning knob this port adds.  The Poisson batch size B is
random, so the minibatch is consumed in fixed-width chunks inside a traced-trip
loop.  Too narrow and the loop runs many times per step; too wide and most
lanes are masked-out padding.  The sweep below is how the CLI defaults were
picked.

Timings are steady-state: each configuration is compiled and run once as a
warm-up (reported separately) before the measured blocks, and every block is
synchronised with ``block_until_ready``.

    python benchmarks/chunk_sweep.py --sampler tunamh
"""

from __future__ import annotations

import argparse
from functools import partial

import jax

from tunamh_jax import data
from tunamh_jax.runner import run_chain
from tunamh_jax.samplers import tfmh, tunamh

MODULES = {"tunamh": tunamh, "tfmh": tfmh}
STEPSIZES = {"tunamh": 1e-3, "tfmh": 1e-4}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampler", choices=sorted(MODULES), default="tunamh")
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--iters", type=int, default=20_000)
    parser.add_argument("--stepsize", type=float)
    parser.add_argument("--chi", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=222)
    parser.add_argument(
        "--chunks", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024, 2048]
    )
    args = parser.parse_args()

    module = MODULES[args.sampler]
    dataset = data.load_mnist_pair(args.pca_dim)
    stepsize = args.stepsize if args.stepsize is not None else STEPSIZES[args.sampler]
    build_kwargs = {"stepsize": stepsize}
    if args.sampler == "tunamh":
        build_kwargs["chi"] = args.chi
    cfg = module.build(dataset, **build_kwargs)

    key_init, key_chain = jax.random.split(jax.random.key(args.seed))
    state = module.init_state(cfg, key_init)

    device = jax.devices()[0]
    print(
        f"{args.sampler} on {device.device_kind} ({device.platform}), "
        f"n={dataset.n_train} d={dataset.dim} stepsize={stepsize:g}, "
        f"{args.iters} iterations per configuration"
    )
    print(f"{'chunk':>7} {'iters/s':>10} {'seconds':>9} {'compile':>9} {'avg batch':>10}")

    for chunk in args.chunks:
        result = run_chain(
            partial(module.step, chunk=chunk),
            cfg,
            state,
            key_chain,
            n_samples=args.iters,
            interval=args.iters,
            warmup=True,
        )
        print(
            f"{chunk:>7} {args.iters / result.total_seconds:>10.0f} "
            f"{result.total_seconds:>9.2f} {result.compile_seconds:>9.2f} "
            f"{result.avg_batch_size:>10.1f}"
        )


if __name__ == "__main__":
    main()
