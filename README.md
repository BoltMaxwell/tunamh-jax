# tunamh-jax

A JAX reimplementation of **TunaMH** and the three exact-MH samplers it is
benchmarked against, on the paper's MNIST 7-vs-9 logistic regression task.

> Ruqi Zhang, A. Feder Cooper, Christopher De Sa.
> [*Asymptotically Optimal Exact Minibatch Metropolis-Hastings*](https://arxiv.org/abs/2006.11677),
> NeurIPS 2020 (Spotlight).

> [!NOTE]
> This repository began as the authors' Julia reference implementation. The port
> to JAX was carried out with [Claude Code](https://claude.com/claude-code),
> which rewrote the samplers, built the test suite, and produced the benchmark
> numbers below. The Julia sources are no longer present; they remain in this
> repository's git history.

The port is not a transliteration. Metropolis-Hastings is a sequential algorithm
with a *data-dependent, random* amount of work per step, which is close to the
worst case for a compiler that wants static shapes — so most of the interesting
content here is in how that was reconciled with XLA without giving up the
exactness guarantee that the paper is about. Those decisions are documented in
[Port decisions](#port-decisions) below.

| sampler  | what it is                                | exact? |
|----------|-------------------------------------------|--------|
| `mh`     | full-batch random-walk Metropolis-Hastings | yes    |
| `tunamh` | TunaMH, Algorithm 2 of the paper           | yes    |
| `tfmh`   | Truncated Factorized MH                    | yes    |
| `flymc`  | Firefly Monte Carlo (stateful)             | yes    |

## Install

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

For GPU, install the matching `jax[cuda12]` wheel in place of the CPU default.

## Run

```bash
.venv/bin/tunamh tunamh                          # paper defaults
.venv/bin/tunamh tunamh --chi 1e-4 --nsamples 10000
.venv/bin/tunamh mh --out results/mh.json        # JSON checkpoint trace
.venv/bin/tunamh tunamh --synthetic 5000         # no MNIST download needed
```

Every flag from the reference scripts is preserved, including `--lam` as an alias
for `--chi`. MNIST is downloaded once to `~/.cache/tunamh-jax/mnist` and the PCA
projection is cached beside it; `--data-dir` points at an existing copy of the four
`idx.gz` files for offline work.

## Results

Paper defaults, measured on an Apple M-series CPU (12,214 train / 2,037 test points,
50 PCA dimensions, seed 222):

| sampler  | iterations | avg batch size | accept rate | test accuracy | runtime |
|----------|-----------:|---------------:|------------:|--------------:|--------:|
| `mh`     |      4,000 |         12,214 |       0.667 |        0.9485 |   1.6 s |
| `tunamh` |     45,000 |            504 |       0.625 |        0.9470 |   5.7 s |
| `tfmh`   |    600,000 |             40 |       0.620 |        0.9372 |  18.6 s |
| `flymc`  |      5,000 |          2,035 |       0.643 |        0.9421 |   3.1 s |

TunaMH reaches full-batch MH's accuracy while touching 24x less data per step, which
is the paper's headline claim.

Runtimes exclude compilation, which is reported separately as `compile_seconds`. They
are **not** comparable to the Julia reference's published numbers: the original was not
re-run as part of this port, so no cross-language speedup is claimed here.

## Port decisions

### Random batch sizes against static shapes

This is the central problem. TunaMH draws `B ~ Poisson(chi C^2 M^2 + C M)` indices per
step and TFMH draws `B ~ Poisson(C M)`, but XLA needs shapes known at trace time.

The tempting fix is to cap `B` at some `B_max`, sample that many, and mask the overflow.
It is also wrong: truncating the Poisson tail changes the stationary distribution, and
exactness is the entire reason to prefer these methods over the inexact ones the paper
argues against in Section 2.1. The failure would be quiet — the chain still looks fine.

Instead the minibatch is consumed in fixed-width chunks inside a loop whose trip count
`ceil(B / chunk)` is a *traced* value (`lax.fori_loop` for TunaMH, `lax.while_loop` for
TFMH so it can still short-circuit on the first rejecting factor). Every draw is used, so
the sampler stays exact for any `chunk`; the parameter buys throughput only. The test
suite asserts that `chunk=3` and `chunk=256` target the same distribution, and the
measured average batch size is identical to one decimal place across every width tried.

### Picking the chunk width

Set it just above the sampler's typical batch size: too narrow and the loop runs many
times per step, too wide and most lanes are masked-out padding. Measured over 20,000
TunaMH iterations at default settings, where the average batch is 504
(`python benchmarks/chunk_sweep.py`):

| chunk    | 64   | 128  | 256  | 512  | 1024 | 2048 |
|----------|-----:|-----:|-----:|-----:|-----:|-----:|
| iters/s  | 5501 | 7376 | 7293 | 7930 | 7201 | 2778 |

Hence the defaults: 512 for TunaMH, 64 for TFMH (whose much smaller stepsize gives it a
typical batch of ~40).

### Rewritten acceptance algebra

The paper writes the TunaMH log-ratio term as
`2 artanh((U_i(θ) − U_i(θ')) / (c_i M (1 + 2 chi C M)))`, and the Julia reference computed
`log(1 + L φ / (lam M_i))`. Both divide by a quantity that vanishes as `chi → 0` or
`M → 0`, which is exactly the small-step regime these samplers run in.

Writing `φ± = ½(±(U_i(θ) − U_i(θ')) + c_i M)` and `q_i = chi C M c_i M`, the same term is
exactly `log(q_i + φ⁺) − log(q_i + φ⁻)`, and the keep probability is exactly
`(q_i + φ⁻) / (q_i + c_i M)`. Both are sums of non-negative quantities. Those are the
forms used here, and `tests/test_tunamh_formulas.py` checks all three formulations agree
numerically so the rewrite is pinned down rather than trusted.

### Numerically safe log-likelihoods

The reference computed `sigmoid(z)` and then took its log, which underflows to `-Inf`
once `|z|` exceeds about 37 and poisons the acceptance ratio. Labels are carried here as
signs `s_i = 2 y_i − 1`, which collapses the per-datum log-likelihood to a single
`log_sigmoid(s_i x_i · θ)` — the same function, evaluated stably. FlyMC's
`log(1 − exp(−gap))` goes through a piecewise `log1mexp` for the same reason.

### float64 throughout

These samplers sum hundreds of log-ratio terms and then exponentiate. In float32 the
rounding is enough to flip accept/reject decisions, which perturbs the stationary
distribution — so x64 is enabled at import rather than left to the caller.
`TUNAMH_JAX_X32=1` opts out for throughput experiments that can accept the loss.

### FlyMC's dark set is vectorised

The reference walked the dark points with geometric skips, evaluating the bound for only
a `q_db` fraction of them. Visiting with probability `q_db` and then accepting with
probability `min(1, exp(lbr)/q_db)` is the same Bernoulli as a single test against
`min(q_db, exp(lbr))`, so the flip is drawn here with one uniform per point and the whole
mask updates at once. The chain is distributionally identical — `tests/test_flymc.py`
measures the empirical flip rates against both expressions — and the reported bright
count is unchanged. The trade is real, though: this port pays one dense matvec per step
instead of skip-sampling, which is the right call on a vector unit and the wrong one in a
scalar loop.

### Chain driver, memory, and timing

A sampler is a pure `step(cfg, state, key) -> (state, StepStats)`. `run_chain` scans it,
so Python runs once per *checkpoint* rather than once per iteration.

The reference stored every sample — 240 MB for the TFMH default. Only the running sum
needed for the posterior-mean estimate is carried inside the scan here, so a run is O(1)
in iteration count; `--store-samples` still gives the full trace when you want it.

Timing excludes compilation: a throwaway block of identical shape is compiled and run
first, and its cost is reported as `compile_seconds`. The warm-up derives its key by
folding rather than splitting, so it does not perturb the chain that follows.

### A burn-in bug in the reference

`mh.jl`, `tfmh.jl` and `flymc.jl` computed their checkpoint accuracy over
`samples[:, 1:i]` regardless of burn-in, double-counting when `burnin > 0`. All three
shipped with `burnin = 0`, so it never fired. Here burn-in is excluded from the running
mean, the stored trace, and every reported statistic, uniformly across samplers.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The load-bearing tests are in `tests/test_samplers.py`. Each sampler runs against a
one-dimensional logistic target whose posterior mean is computed by quadrature, and must
recover it to within 5% of a posterior standard deviation. This matters because a wrong
acceptance ratio still produces a chain that looks entirely plausible — trace plots,
acceptance rates, and test accuracy all stay believable while the stationary distribution
is quietly wrong. Comparing against an independently computed target is the only check
that catches it.

`tests/test_logistic.py` additionally verifies Assumption 1 empirically — that
`c_i = ||x_i|| / T` really does bound `|U_i(θ) − U_i(θ')| / ||θ − θ'||` — since all three
minibatch samplers are built on that bound holding.

## Layout

```
tunamh_jax/
  alias.py            Vose alias table: O(1) draws from the fixed c_i / C distribution
  data.py             MNIST 7-vs-9 loading, PCA, synthetic fallback (host-side, once)
  logistic.py         shared model pieces: log-likelihood, Lipschitz constants, init
  runner.py           StepStats, proposal/accept helpers, lax.scan chain driver
  samplers/           mh.py, tunamh.py, tfmh.py, flymc.py
  cli.py
benchmarks/chunk_sweep.py
tests/
```

## Citation

```bibtex
@inproceedings{zhang2020tunamh,
  title     = {Asymptotically Optimal Exact Minibatch Metropolis-Hastings},
  author    = {Zhang, Ruqi and Cooper, A. Feder and De Sa, Christopher},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2020},
}
```
