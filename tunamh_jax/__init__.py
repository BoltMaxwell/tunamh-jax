"""JAX implementation of TunaMH and baseline minibatch Metropolis-Hastings samplers.

Reference: Zhang, Cooper & De Sa, "Asymptotically Optimal Exact Minibatch
Metropolis-Hastings", NeurIPS 2020.

Every sampler here accumulates log-acceptance ratios over hundreds of terms and
then exponentiates them, so single precision is not a safe default: float32
rounding perturbs accept/reject decisions and therefore the stationary
distribution.  x64 is enabled at import time and the whole package assumes it.
Set ``TUNAMH_JAX_X32=1`` to opt out (only sensible for throughput experiments
where you accept the loss of exactness).
"""

import os as _os

if _os.environ.get("TUNAMH_JAX_X32", "") != "1":
    import jax as _jax

    _jax.config.update("jax_enable_x64", True)

from tunamh_jax import alias, data, logistic, runner, samplers

__all__ = ["alias", "data", "logistic", "runner", "samplers"]
