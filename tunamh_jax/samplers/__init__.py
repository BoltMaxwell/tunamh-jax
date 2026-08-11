"""The four samplers compared in the paper.

* :mod:`tunamh_jax.samplers.mh` -- standard full-batch Metropolis-Hastings.
* :mod:`tunamh_jax.samplers.tunamh` -- TunaMH (Algorithm 2), exact minibatch MH.
* :mod:`tunamh_jax.samplers.tfmh` -- Truncated Factorized MH, exact minibatch MH.
* :mod:`tunamh_jax.samplers.flymc` -- FlyMC, exact but stateful.

Each module exposes ``Config``, ``State``, ``build``, ``init_state`` and
``step``.  ``step(cfg, state, key) -> (state, StepStats)`` is pure and jittable;
the minibatch samplers additionally take a static ``chunk`` keyword.
"""

from tunamh_jax.samplers import flymc, mh, tfmh, tunamh

__all__ = ["flymc", "mh", "tfmh", "tunamh"]

REGISTRY = {"mh": mh, "tunamh": tunamh, "tfmh": tfmh, "flymc": flymc}
